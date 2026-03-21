import json
import re
import traceback
import math
import pickle
import torch
import gc
import os
import time
import numpy as np
import random

from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from utils import safe_parse_scene, inherit_props_by_id, get_sft_model, init_wandb, create_vllm_engine, destroy_vllm_engine, is_high_quality_sample
from dataset import process_scene_sample, format_with_chat_template, create_full_scene_from_before_and_added, RejSFTSceneDataCollator, ensure_order_of_keys_for_sg_input_dict
from eval import eval_scene_before_after_with_delta, compute_dss_score, compute_size_l2_dist
from train import CustomTrainerCallback
from dataset import create_instruction_from_scene

def normalize_completion(completion_text):
	"""Normalize a completion string for comparison: parse JSON and enforce canonical key order."""
	try:
		unescaped = completion_text.replace('\\"', '"')
		parsed = json.loads(unescaped)

		if parsed.get("objects") is not None:
			# Full scene format
			parsed = ensure_order_of_keys_for_sg_input_dict(parsed)
		else:
			# Single object format — enforce fixed key order
			ordered = {}
			for key in ["desc", "size", "pos", "rot"]:
				if key in parsed:
					ordered[key] = parsed[key]
			for key in parsed:
				if key not in ordered:
					ordered[key] = parsed[key]
			parsed = ordered

		return json.dumps(parsed)
	except:
		return None


def is_gt_match(completion_text, gt_completion_text):
	"""Check if a generated completion matches the ground truth (normalized JSON comparison)."""
	norm_gen = normalize_completion(completion_text)
	norm_gt = normalize_completion(gt_completion_text)
	if norm_gen is None or norm_gt is None:
		return False
	return norm_gen == norm_gt


def score_completion_with_metrics(completion_text, scene_before_with_assets, sg_output_add, orig_prompt, sampling_engine):
	"""Returns dict with all metrics + is_high_quality flag"""
	result = {
		"is_high_quality": False,
		"txt_dss_score": 0.0,
		"delta_pbl_loss": float('inf'),
		"txt_pms_sampled_score": 0.0,
		"size_l2_dist": float('inf'),
		"is_valid_json": False,
	}
	
	unescaped = completion_text.replace('\\"', '"')
	scene_after = safe_parse_scene(unescaped)
	if scene_after is None:
		return result
	
	result["is_valid_json"] = True
	
	try:
		if scene_after.get("objects") is None:
			scene_after = create_full_scene_from_before_and_added(scene_before_with_assets, scene_after)

		# if scene_after.get("jid") is not None:
		# import pdb
		# pdb.set_trace()
		
		scene_after_with_assets = sampling_engine.sample_last_asset(scene_after, is_greedy_sampling=True)
		inherit_props_by_id(scene_before_with_assets, scene_after_with_assets)
		scene_after_with_assets["objects"][-1]["prompt"] = orig_prompt
		
		metrics = eval_scene_before_after_with_delta(scene_before_with_assets, scene_after_with_assets)
		
		new_obj_desc = scene_after_with_assets["objects"][-1].get("desc")
		gt_obj_desc = sg_output_add.get("desc")
		txt_dss_score = compute_dss_score(new_obj_desc, gt_obj_desc, sampling_engine)
		
		new_obj_size = scene_after_with_assets["objects"][-1].get("size")
		gt_obj_size = sg_output_add.get("size")
		size_l2_dist = compute_size_l2_dist(new_obj_size, gt_obj_size)
		
		result.update({
			"delta_pbl_loss": metrics["delta_pbl_loss"],
			"txt_pms_sampled_score": metrics["txt_pms_sampled_score"],
			"txt_dss_score": txt_dss_score,
			"size_l2_dist": size_l2_dist,
		})
		
		is_high_quality = is_high_quality_sample(
			metrics["delta_pbl_loss"],
			metrics["txt_pms_sampled_score"],
			size_l2_dist,
			txt_dss_score,
			desc=new_obj_desc
		)
		result["is_high_quality"] = is_high_quality
		
	except Exception as e:
		print(f"error scoring completion: {e}")
		traceback.print_exc()
	
	return result


def score_single_gt_sample(args_tuple):
	"""Worker for scoring one GT sample."""
	pdata, sampling_engine = args_tuple

	scene_before = json.loads(pdata["sg_input_with_jids"])
	scene_before_with_assets = sampling_engine.sample_all_assets(scene_before, is_greedy_sampling=True)
	sg_output_add = json.loads(pdata["completion_gt"])

	metrics_dict = score_completion_with_metrics(
		pdata["sg_output_add_with_jids"],
		scene_before_with_assets,
		sg_output_add,
		pdata["orig_prompt"],
		sampling_engine,
	)
	return metrics_dict


def score_gt_samples(prompt_data, sampling_engine, num_workers=1):
	"""Threaded GT scoring, same pattern as score_and_filter."""
	print(f"scoring GT samples (num_workers={num_workers})...")

	work_items = [(pdata, sampling_engine) for pdata in prompt_data]

	if num_workers > 1:
		from concurrent.futures import ThreadPoolExecutor
		with ThreadPoolExecutor(max_workers=num_workers) as pool:
			gt_metrics_all = list(pool.map(score_single_gt_sample, work_items))
	else:
		gt_metrics_all = [score_single_gt_sample(item) for item in work_items]

	print("finished scoring GT samples")

	return gt_metrics_all


def compute_gt_pass_rate(prompt_data, sampling_engine, tokenizer, num_workers=1):
	"""Compute GT pass rate using threaded scoring."""
	gt_metrics_all = score_gt_samples(prompt_data, sampling_engine, num_workers)

	gt_pass_count = sum(1 for m in gt_metrics_all if m["is_high_quality"])
	gt_pass_rate = gt_pass_count / len(prompt_data) if prompt_data else 0.0
	gt_metrics_summary = compute_metrics_summary(gt_metrics_all)

	return gt_pass_rate, gt_pass_count, gt_metrics_summary


def compute_metrics_summary(all_metrics):
	"""Compute mean/std for all metric fields"""
	valid_metrics = [m for m in all_metrics if m["is_valid_json"]]
	
	if not valid_metrics:
		return {
			"txt_dss_score_mean": 0.0, "txt_dss_score_std": 0.0,
			"delta_pbl_loss_mean": 0.0, "delta_pbl_loss_std": 0.0,
			"txt_pms_sampled_mean": 0.0, "txt_pms_sampled_std": 0.0,
			"size_l2_dist_mean": 0.0, "size_l2_dist_std": 0.0,
			"valid_json_rate": 0.0,
		}
	
	txt_dss = [m["txt_dss_score"] for m in valid_metrics]
	delta_pbl = [m["delta_pbl_loss"] for m in valid_metrics if m["delta_pbl_loss"] != float('inf')]
	txt_pms = [m["txt_pms_sampled_score"] for m in valid_metrics]
	size_l2 = [m["size_l2_dist"] for m in valid_metrics if m["size_l2_dist"] != float('inf')]
	
	return {
		"txt_dss_score_mean": np.mean(txt_dss) if txt_dss else 0.0,
		"txt_dss_score_std": np.std(txt_dss) if txt_dss else 0.0,
		"delta_pbl_loss_mean": np.mean(delta_pbl) if delta_pbl else 0.0,
		"delta_pbl_loss_std": np.std(delta_pbl) if delta_pbl else 0.0,
		"txt_pms_sampled_mean": np.mean(txt_pms) if txt_pms else 0.0,
		"txt_pms_sampled_std": np.std(txt_pms) if txt_pms else 0.0,
		"size_l2_dist_mean": np.mean(size_l2) if size_l2 else 0.0,
		"size_l2_dist_std": np.std(size_l2) if size_l2 else 0.0,
		"valid_json_rate": len(valid_metrics) / len(all_metrics),
	}


# ======================================================================
# Unified generation: prepare prompts, generate completions, score+filter
# ======================================================================

def prepare_prompt_data(dataset, tokenizer, sampling_engine, args, max_seq_length, format_prompts=False):
	"""Prepare prompt data from dataset. If format_prompts=True, also store formatted prompts (for vLLM).
	Also stores sg_input/sg_output_add with jids so scoring can use GT asset assignments."""
	all_prompts_meta = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
	all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

	prompt_data = []
	for idx in range(len(dataset)):
		sample = dataset[idx]
		full_instr, completion_gt, orig_prompt, instr_sample = process_scene_sample(
			sample, tokenizer, max_seq_length, all_prompts_meta, all_assets_metadata_simple_descs,
			args.do_simple_descs, args.do_augm, args.do_full_sg_outputs
		)
		pdata = {
			"full_instr": full_instr,
			"completion_gt": completion_gt,
			"orig_prompt": orig_prompt,
			"sg_input_with_jids": instr_sample["sg_input_with_jids"],
			"sg_output_add_with_jids": instr_sample["sg_output_add_with_jids"],
		}
		if format_prompts:
			pdata["formatted_prompt"] = format_with_chat_template(tokenizer, full_instr)
		prompt_data.append(pdata)

	print(f"prepared {len(prompt_data)} prompts for generation")
	return prompt_data


def generate_completions_vllm(llm_engine, sampling_params, tokenizer, prompt_data):
	"""Generate N completions per prompt using vLLM. Returns list of list of completion strings."""
	all_formatted_prompts = [p["formatted_prompt"] for p in prompt_data]

	chunk_size = 128
	all_outputs = []
	for idx, i in enumerate(range(0, len(all_formatted_prompts), chunk_size)):
		chunk = all_formatted_prompts[i:i+chunk_size]
		print(f"chunk {idx+1}/{(len(all_formatted_prompts) + chunk_size - 1) // chunk_size} starting forward pass through vLLM")
		outputs = llm_engine.generate(chunk, sampling_params, use_tqdm=False)
		print("finished forward pass through vLLM")
		all_outputs.extend(outputs)
		print(f"generated {min(i+chunk_size, len(all_formatted_prompts))}/{len(all_formatted_prompts)} prompts")

	# Convert to list of list of completion strings
	completions_per_prompt = []
	for output in all_outputs:
		completions = [tokenizer.decode(gen.token_ids, skip_special_tokens=True) for gen in output.outputs]
		completions_per_prompt.append(completions)

	return completions_per_prompt


def generate_completions_standard(model, tokenizer, accelerator, prompt_data, args):
	"""Generate N completions per prompt using model.generate(). Returns list of list of completion strings."""
	from test import get_batch_input_ids

	gen_kwargs = {
		"max_new_tokens": 128,
		"pad_token_id": tokenizer.pad_token_id,
		"do_sample": True,
		"temperature": 0.7,
		"top_k": 50,
		"top_p": 0.95,
	}

	model.eval()
	torch.use_deterministic_algorithms(False)

	completions_per_prompt = []
	print(f"generating completions with standard model.generate()... at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
	for idx, pdata in enumerate(prompt_data):
		input_ids, attention_mask = get_batch_input_ids([pdata["full_instr"]], tokenizer)
		input_ids = input_ids.repeat(args.rej_num_gen, 1).to(accelerator.device)
		attention_mask = attention_mask.repeat(args.rej_num_gen, 1).to(accelerator.device)

		with torch.inference_mode():
			outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

		generated_ids = outputs[:, input_ids.shape[-1]:]
		completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
		completions_per_prompt.append(completions)

	print(f"finished generating completions at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

	torch.use_deterministic_algorithms(True)
	model.train()

	return completions_per_prompt


def _score_single_prompt(args_tuple):
	"""Worker function for parallel scoring. Scores all N completions for one prompt.
	Uses sg_input_with_jids for scene_before asset sampling to avoid variance from random asset assignment."""
	pdata, completions, sampling_engine, tokenizer = args_tuple

	# formatted_prompt = pdata.get("formatted_prompt") or format_with_chat_template(tokenizer, pdata["full_instr"])

	# Use the jid version for scene_before so existing objects get their GT assets
	scene_before = json.loads(pdata["sg_input_with_jids"])
	scene_before_with_assets = sampling_engine.sample_all_assets(scene_before, is_greedy_sampling=True)
	sg_output_add = json.loads(pdata["completion_gt"])

	prompt_metrics = []
	prompt_accepted_completions = []
	prompt_gt_matches = 0

	for completion_text in completions:
		metrics_dict = score_completion_with_metrics(completion_text, scene_before_with_assets, sg_output_add, pdata["orig_prompt"], sampling_engine)
		prompt_metrics.append(metrics_dict)

		if metrics_dict["is_high_quality"]:
			prompt_accepted_completions.append(completion_text)

		if is_gt_match(completion_text, pdata["completion_gt"]):
			prompt_gt_matches += 1

	return {
		"full_instr": pdata["full_instr"],
		"metrics": prompt_metrics,
		"accepted_completions": prompt_accepted_completions,
		"gt_matches": prompt_gt_matches,
	}


def score_and_filter(prompt_data, completions_per_prompt, sampling_engine, tokenizer, max_samples_per_prompt=None, num_workers=1):
	"""Score all completions, filter high-quality ones, and compute GT match stats.
	Set num_workers > 1 to parallelize scoring across prompts using threads."""
	
	print(f"scoring generated candidates and filtering (num_workers={num_workers})...")

	work_items = [
		(pdata, completions, sampling_engine, tokenizer)
		for pdata, completions in zip(prompt_data, completions_per_prompt)
	]

	if num_workers > 1:
		from concurrent.futures import ThreadPoolExecutor
		with ThreadPoolExecutor(max_workers=num_workers) as pool:
			results = list(pool.map(_score_single_prompt, work_items))
	else:
		results = [_score_single_prompt(item) for item in work_items]

	print("finished scoring and filtering generated candidates")

	# Aggregate results
	all_instructions = []
	all_completions = []
	all_candidate_metrics = []
	gt_match_counts_per_prompt = []
	total_gt_matches = 0
	n_total_candidates = 0
	n_passed_filters = 0  # ← NEW: count before capping
	n_accepted = 0
	n_fallback = 0

	for pdata, result in zip(prompt_data, results):
		n_completions = len(result["metrics"])
		n_total_candidates += n_completions
		all_candidate_metrics.extend(result["metrics"])

		n_passed_filters += len(result["accepted_completions"])

		# Step 1: Deduplicate accepted completions
		unique_accepted = []
		seen_normalized = set()
		for comp in result["accepted_completions"]:
			norm = normalize_completion(comp)
			if norm and norm not in seen_normalized:
				unique_accepted.append(comp)
				seen_normalized.add(norm)
		
		# Step 2: Cap or fallback
		if len(unique_accepted) == 0:
			# No accepted samples: use GT fallback
			samples_to_use = [pdata["completion_gt"]]
			n_fallback += 1
		else:
			if max_samples_per_prompt is None:
				# Old behavior: keep all unique accepted samples
				samples_to_use = unique_accepted
			else:
				# Has accepted: cap at max_samples_per_prompt
				n_to_sample = min(len(unique_accepted), max_samples_per_prompt)
				samples_to_use = random.sample(unique_accepted, n_to_sample)
			n_accepted += len(samples_to_use)

		# Add to training set
		for comp in samples_to_use:
			all_instructions.append(pdata["full_instr"])
			all_completions.append(comp)

		gt_match_counts_per_prompt.append(result["gt_matches"])
		total_gt_matches += result["gt_matches"]

	# accept_rate = n_accepted / n_total_candidates if n_total_candidates > 0 else 0
	filter_pass_rate = n_passed_filters / n_total_candidates if n_total_candidates > 0 else 0
	kept_rate = n_accepted / n_total_candidates if n_total_candidates > 0 else 0
	metrics_summary = compute_metrics_summary(all_candidate_metrics)

	# GT match stats
	n_prompts = len(prompt_data)
	n_per_prompt = len(completions_per_prompt[0]) if completions_per_prompt else 0
	gt_match_rate = total_gt_matches / n_total_candidates if n_total_candidates > 0 else 0
	gt_match_at_least_once = sum(1 for c in gt_match_counts_per_prompt if c > 0)
	gt_match_coverage = gt_match_at_least_once / n_prompts if n_prompts > 0 else 0
	avg_gt_matches_per_prompt = np.mean(gt_match_counts_per_prompt) if gt_match_counts_per_prompt else 0

	gt_match_stats = {
		"gt_match_rate": gt_match_rate,
		"gt_match_coverage": gt_match_coverage,
		"gt_match_at_least_once": gt_match_at_least_once,
		"avg_gt_matches_per_prompt": avg_gt_matches_per_prompt,
		"gt_match_counts_per_prompt": gt_match_counts_per_prompt,
	}

	print(f"rejection sampling kept_rate: {n_accepted}/{n_total_candidates} model-generated accepted ({kept_rate:.2%})")
	print(f"fallback samples (GT): {n_fallback}/{n_prompts}")
	print(f"total training samples: {len(all_instructions)} ({n_accepted} accepted + {n_fallback} fallback)")
	print(f"unique prompts with model-generated samples: {len(set(all_instructions)) - n_fallback}/{n_prompts}")
	print(f"GT match rate: {total_gt_matches}/{n_total_candidates} ({gt_match_rate:.2%})")
	print(f"GT match coverage: {gt_match_at_least_once}/{n_prompts} prompts have >=1 GT match ({gt_match_coverage:.2%})")
	print(f"Avg GT matches per prompt: {avg_gt_matches_per_prompt:.2f}/{n_per_prompt}")

	return all_instructions, all_completions, filter_pass_rate, kept_rate, metrics_summary, gt_match_stats, n_passed_filters, n_accepted, n_fallback


def print_epoch_stats(epoch, filter_pass_rate, kept_rate, n_passed_filters, n_accepted, n_total, n_unique, n_dataset, n_fallback, n_all, metrics_summary, gt_metrics_summary, gt_match_stats, n_per_prompt):
	"""Print detailed epoch statistics."""
	print(f"\n{'='*80}")
	print(f"EPOCH {epoch+1} GENERATION STATISTICS")
	print(f"{'='*80}")
	print(f"Filter pass rate: {filter_pass_rate:.2%} ({n_passed_filters}/{n_total} candidates passed filters)")
	print(f"Accepted: {n_accepted}/{n_total} ({kept_rate:.2%})")
	print(f"Unique prompts with ≥1 accepted: {n_unique}/{n_dataset}")
	print(f"Fallback samples (0 accepted): {n_fallback}")
	print(f"Total training samples: {n_all}")
	print(f"\nMETRIC DISTRIBUTIONS (all {n_total} candidates):")
	print(f"  Valid JSON rate:      {metrics_summary['valid_json_rate']:.2%}")
	print(f"  txt_dss_score:        mean={metrics_summary['txt_dss_score_mean']:.3f}, std={metrics_summary['txt_dss_score_std']:.3f}")
	print(f"  delta_pbl_loss:       mean={metrics_summary['delta_pbl_loss_mean']:.3f}, std={metrics_summary['delta_pbl_loss_std']:.3f}")
	print(f"  txt_pms_sampled:      mean={metrics_summary['txt_pms_sampled_mean']:.3f}, std={metrics_summary['txt_pms_sampled_std']:.3f}")
	print(f"  size_l2_dist:         mean={metrics_summary['size_l2_dist_mean']:.3f}, std={metrics_summary['size_l2_dist_std']:.3f}")
	print(f"\nGROUND TRUTH METRIC DISTRIBUTIONS:")
	print(f"  Valid JSON rate:      {gt_metrics_summary['valid_json_rate']:.2%}")
	print(f"  txt_dss_score:        mean={gt_metrics_summary['txt_dss_score_mean']:.3f}, std={gt_metrics_summary['txt_dss_score_std']:.3f}")
	print(f"  delta_pbl_loss:       mean={gt_metrics_summary['delta_pbl_loss_mean']:.3f}, std={gt_metrics_summary['delta_pbl_loss_std']:.3f}")
	print(f"  txt_pms_sampled:      mean={gt_metrics_summary['txt_pms_sampled_mean']:.3f}, std={gt_metrics_summary['txt_pms_sampled_std']:.3f}")
	print(f"  size_l2_dist:         mean={gt_metrics_summary['size_l2_dist_mean']:.3f}, std={gt_metrics_summary['size_l2_dist_std']:.3f}")
	print(f"  pass_rate:             {gt_metrics_summary.get('pass_rate', 0.0):.2%}")
	print(f"\nGT MATCH STATISTICS:")
	print(f"  GT match rate:        {gt_match_stats['gt_match_rate']:.2%} ({int(gt_match_stats['gt_match_rate'] * n_total)}/{n_total})")
	print(f"  GT match coverage:    {gt_match_stats['gt_match_coverage']:.2%} ({gt_match_stats['gt_match_at_least_once']}/{n_dataset} prompts)")
	print(f"  Avg GT matches/prompt: {gt_match_stats['avg_gt_matches_per_prompt']:.2f}/{n_per_prompt}")
	print(f"{'='*80}\n")

def get_epoch_training_subset(dataset_train, epoch, n_samples_per_epoch=2000, use_fixed_subset=False):
	"""Rotate through dataset across epochs"""
	n_total = len(dataset_train)
	
	if use_fixed_subset:
		# Always return the same first n_samples_per_epoch scenes
		indices = list(range(min(n_samples_per_epoch, n_total)))
		return dataset_train.select(indices)
	
	# Original behavior: Deterministic rotation
	start_idx = (epoch * n_samples_per_epoch) % n_total
	indices = list(range(start_idx, min(start_idx + n_samples_per_epoch, n_total)))
	
	# Wrap around if needed
	if len(indices) < n_samples_per_epoch and n_total > n_samples_per_epoch:
		indices += list(range(n_samples_per_epoch - len(indices)))
	
	return dataset_train.select(indices)

def do_rej_training(model_id, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args):

	accelerator.wait_for_everyone()
	gc.collect()
	torch.cuda.empty_cache()

	model, max_seq_length, lora_rank, lora_alpha = get_sft_model(model_id, args, accelerator)

	tokenizer.padding_side = "left"
	if "qwen" in args.llm:
		model.config.use_cache = False

	learning_rate = args.rej_learning_rate

	if "llama" in model_id:
		response_template = "<|start_header_id|>user<|end_header_id|>\n\n"
	else:
		response_template = "<|im_start|>assistant\n"

	if args.use_vllm:
		from vllm import SamplingParams
		sampling_params = SamplingParams(
			n=args.rej_num_gen,
			max_tokens=128,
			temperature=0.7,
			top_k=50,
			top_p=0.95,
			detokenize=False,
		)

	init_wandb(args, accelerator)

	if accelerator.is_main_process:
		print("=" * 95)
		print(f"Rejection Sampling SFT: starting training (Approach A: sequential vLLM → purge → multi-GPU SFT)")
		print(f"RUN ID (wandb): {args.run_id}")
		print(f"JOB ID (sherlock): {args.jid}")
		print(f"number of processes (accelerate): {accelerator.num_processes}")
		if args.use_lora:
			print(f"lora rank: {lora_rank}, lora alpha: {lora_alpha}")
			model.print_trainable_parameters()
		else:
			print("full number of params to train:", model.num_parameters())
		print(f"num generations per prompt: {args.rej_num_gen}")
		print(f"learning rate: {learning_rate}")
		print(f"len of training dataset: {len(dataset_train)}")
		print(f"max rounds: {args.epochs}")
		print(f"per device batch size: {args.dvc_batch_size}")
		print(f"gas steps: {args.gas_steps}")
		print(f"using vLLM: {args.use_vllm}")
		print(f"sanity check: {args.do_sanity_check}")
		print(f"augmentation: {args.do_augm}")
		print(f"cpus for eval scoring: {args.num_eval_workers}")
		print(f"n samples per epoch: {args.rej_samples_per_epoch}")
		print("=" * 95)

	# for sanity check: use a small subset
	if args.do_sanity_check:
		n_gen_samples = min(50, len(dataset_train))
		dataset_train = dataset_train.select(range(n_gen_samples))
		if accelerator.is_main_process:
			print(f"SANITY CHECK: using {n_gen_samples} train samples for generation (instead of {len(dataset_train)})")
	else:
		dataset_train = dataset_train

	pth_rej_dataset = os.getenv("PTH_SFT_REJ_DATASET")
	if accelerator.is_main_process:
		os.makedirs(pth_rej_dataset, exist_ok=True)

	n_rounds = args.epochs
	best_val_delta_pbl_loss = float("inf")
	
	for epoch in range(n_rounds):

		if accelerator.is_main_process:
			print(f"\n{'='*80}")
			print(f"REJ-SFT ROUND {epoch+1}/{n_rounds}")
			print(f"{'='*80}")

		if not args.do_augm:
			np.random.seed(1234)
			random.seed(1234)
			torch.manual_seed(1234)

		if not args.do_sanity_check:
			dataset_train_gen_epoch = get_epoch_training_subset(dataset_train, epoch, n_samples_per_epoch=args.rej_samples_per_epoch, use_fixed_subset=args.rej_use_fixed_subset)
		else:
			dataset_train_gen_epoch = dataset_train

		# ======================================================================
		# PHASE 1: Generate + filter on main process, save to disk
		# ======================================================================
		epoch_pkl_path = os.path.join(pth_rej_dataset, f"epoch-{epoch+1}.pkl")

		if accelerator.is_main_process:

			# --- Prepare prompts ---
			prompt_data = prepare_prompt_data(dataset_train_gen_epoch, tokenizer, sampling_engine, args, max_seq_length, format_prompts=args.use_vllm)

			# --- Generate completions (vLLM or standard) ---
			if args.use_vllm:
				if epoch == 0:
					vllm_model_path = f"./ckpts/{args.test_ckpt}"
				else:
					vllm_model_path = f"./ckpts/{args.jid}/checkpoint-last"

				print(f"loading vLLM from: {vllm_model_path}")
				gc.collect()
				torch.cuda.empty_cache()
				time.sleep(3.0)

				# llm_engine = create_vllm_engine(vllm_model_path, max_seq_length)
				llm_engine = create_vllm_engine(vllm_model_path, max_seq_length, device="cuda:1")

				completions_per_prompt = generate_completions_vllm(llm_engine, sampling_params, tokenizer, prompt_data)

				# purge vLLM
				destroy_vllm_engine(llm_engine)
			else:
				completions_per_prompt = generate_completions_standard(model, tokenizer, accelerator, prompt_data, args)

			# --- Score, filter, compute GT match ---
			all_instructions, all_completions, filter_pass_rate, kept_rate, metrics_summary, gt_match_stats, n_passed_filters, n_accepted, n_fallback = score_and_filter(
				prompt_data, completions_per_prompt, sampling_engine, tokenizer, 
				max_samples_per_prompt=args.rej_max_samples_per_prompt,
				num_workers=args.num_eval_workers
			)

			# --- GT pass rate ---
			gt_pass_rate, gt_pass_count, gt_metrics_summary = compute_gt_pass_rate(prompt_data, sampling_engine, tokenizer, num_workers=args.num_eval_workers)

			n_total = args.rej_num_gen * len(dataset_train_gen_epoch)
			n_unique_with_accepted = len(set(all_instructions)) - n_fallback  # Prompts with model-generated samples
			avg_samples_per_scene = len(all_completions) / len(dataset_train_gen_epoch)

			print_epoch_stats(
				epoch, filter_pass_rate, kept_rate, n_passed_filters, n_accepted, n_total,
				n_unique_with_accepted, len(dataset_train_gen_epoch), n_fallback, len(all_instructions),
				metrics_summary, gt_metrics_summary, gt_match_stats, args.rej_num_gen
			)

			# --- Save to disk ---
			epoch_data = {
				"instructions": all_instructions,
				"completions": all_completions,
				"filter_pass_rate": filter_pass_rate,
    			"kept_rate": kept_rate,
				"metrics_summary": metrics_summary,
				"n_accepted": n_accepted,
				"n_fallback": n_fallback,
				"n_passed_filters": n_passed_filters,
    			"n_unique_with_accepted": n_unique_with_accepted,
    			"avg_samples_per_scene": avg_samples_per_scene,
				"gt_pass_rate": gt_pass_rate,
				"gt_pass_count": gt_pass_count,
				"gt_metrics_summary": gt_metrics_summary,
				"gt_match_stats": gt_match_stats,
			}
			with open(epoch_pkl_path, "wb") as f:
				pickle.dump(epoch_data, f)
			print(f"saved {len(all_instructions)} samples to {epoch_pkl_path} ({n_accepted} accepted + {n_fallback} fallback)")

			gc.collect()
			torch.cuda.empty_cache()

		accelerator.wait_for_everyone()

		# ======================================================================
		# PHASE 2: All processes load pickle and do multi-GPU SFT
		# ======================================================================
		with open(epoch_pkl_path, "rb") as f:
			epoch_data = pickle.load(f)
		all_instructions = epoch_data["instructions"]
		all_completions = epoch_data["completions"]
		filter_pass_rate = epoch_data["filter_pass_rate"]
		kept_rate = epoch_data["kept_rate"]
		metrics_summary = epoch_data["metrics_summary"]
		n_accepted = epoch_data["n_accepted"]
		n_fallback = epoch_data["n_fallback"]
		gt_match_stats = epoch_data.get("gt_match_stats", {})

		n_unique_with_accepted = epoch_data.get("n_unique_with_accepted")
		avg_samples_per_scene = epoch_data.get("avg_samples_per_scene")
		saturation_rate = avg_samples_per_scene / args.rej_max_samples_per_prompt if args.rej_max_samples_per_prompt else None

		if len(all_instructions) == 0:
			print(f"round {epoch+1}: no training samples, skipping SFT step")
			continue

		filtered_dataset = Dataset.from_dict({
			"instruction": all_instructions,
			"completion": all_completions,
		})
		filtered_dataset = filtered_dataset.shuffle(seed=args.seed + epoch)

		if accelerator.is_main_process:
			print(f"filtered dataset size: {len(filtered_dataset)} ({n_accepted} accepted + {n_fallback} fallback)")

		data_collator = RejSFTSceneDataCollator(
			response_template=response_template,
			tokenizer=tokenizer,
			max_seq_length=max_seq_length,
			do_simple_descs=args.do_simple_descs,
			do_augm=False if args.do_sanity_check else args.do_augm,
			do_full_sg_outputs=args.do_full_sg_outputs,
		)

		per_device_batch_size = args.dvc_batch_size
		gas_steps = args.gas_steps
		samples_per_step = per_device_batch_size * accelerator.num_processes * gas_steps
		steps_per_epoch = max(1, math.ceil(len(filtered_dataset) / samples_per_step))

		sft_config = SFTConfig(
			output_dir=f"./ckpts/{args.jid}",
			logging_dir=f"./logs/{args.jid}",
			max_seq_length=max_seq_length,
			packing=False,
			bf16=True,
			lr_scheduler_type="cosine",
			learning_rate=learning_rate,
			warmup_steps=0,
			max_steps=-1,
			num_train_epochs=1,
			per_device_train_batch_size=per_device_batch_size,
			gradient_accumulation_steps=gas_steps,
			save_strategy="no",
			logging_steps=steps_per_epoch,
			logging_strategy="epoch",
			eval_strategy="epoch",
			greater_is_better=False,
			dataset_kwargs={'skip_prepare_dataset': True},
			remove_unused_columns=False,
			report_to=[],
		)

		trainer = SFTTrainer(
			model=model,
			train_dataset=filtered_dataset,
			eval_dataset=dataset_val,
			processing_class=tokenizer,
			data_collator=data_collator,
			args=sft_config,
		)

		trainer.model = accelerator.prepare(trainer.model)

		callback = CustomTrainerCallback(
			n_samples_snippet=min(500, len(dataset_test) if not args.do_sanity_check else 4),
			trainer=trainer,
			dataset_train=dataset_train,
			dataset_val=dataset_val,
			dataset_test=dataset_test,
			sampling_engine=sampling_engine,
			dvc=dvc,
			cli_args=args,
			accelerator=accelerator,
			is_sft_training=True,
			epoch_offset=epoch,
			initial_best_val=best_val_delta_pbl_loss,
		)
		trainer.add_callback(callback)

		trainer.train()

		# Propagate the best val loss seen so far to the next round's callback
		best_val_delta_pbl_loss = callback.best_val_delta_pbl_loss

		if accelerator.is_main_process:
			print(f"round {epoch+1}: SFT step complete")

		if accelerator.is_main_process and args.use_wandb:
			import wandb
			gt_metrics_summary = epoch_data.get("gt_metrics_summary", {})
			wandb.log({
				"rej_sft/filter_pass_rate": filter_pass_rate,
    			"rej_sft/kept_rate": kept_rate,
				"rej_sft/n_accepted": n_accepted,
				"rej_sft/n_fallback": n_fallback,
				"rej_sft/n_unique_prompts_with_accepted": n_unique_with_accepted, # Number of unique prompts (from the generation pool) that had at least one model-generated sample pass quality filters.
				"rej_sft/coverage_rate": n_unique_with_accepted / len(dataset_train_gen_epoch), # Percentage of prompts in the generation pool that had at least one accepted model-generated sample (i.e., didn't need GT fallback).
				"rej_sft/avg_samples_per_scene": avg_samples_per_scene, # Average number of accepted samples contributed to the training set per prompt in the generation pool (accounts for capping and fallback).
				"rej_sft/saturation_rate": saturation_rate if saturation_rate is not None else 0.0,

				"rej_sft/txt_dss_score_mean": metrics_summary['txt_dss_score_mean'],
				"rej_sft/delta_pbl_loss_mean": metrics_summary['delta_pbl_loss_mean'],
				"rej_sft/txt_pms_sampled_mean": metrics_summary['txt_pms_sampled_mean'],
				"rej_sft/size_l2_dist_mean": metrics_summary['size_l2_dist_mean'],
				"rej_sft/valid_json_rate": metrics_summary['valid_json_rate'],

				"rej_sft/unique_completions_ratio": len(set(all_completions)) / len(all_completions), # Fraction of training samples that are unique after deduplication (1.0 = no duplicates, 0.5 = half were duplicates).
				"rej_sft/avg_completions_per_prompt": len(all_completions) / len(set(all_instructions)), # Average number of training samples contributed per unique prompt (accounts for prompts with multiple accepted samples due to capping).
				"rej_sft/train_scenes_ratio": len(set(all_instructions)) / len(dataset_train_gen_epoch), # Fraction of the generation pool that appears in the final training set (should be ~1.0 since fallback ensures all prompts are included, but accounting for augmentation creating different instruction strings).
				
				"rej_sft/gt_pass_rate": epoch_data.get("gt_pass_rate", 0.0),
				"rej_sft/gt_pass_count": epoch_data.get("gt_pass_count", 0),
				"rej_sft/gt_txt_dss_score_mean": gt_metrics_summary.get('txt_dss_score_mean', 0.0),
				"rej_sft/gt_delta_pbl_loss_mean": gt_metrics_summary.get('delta_pbl_loss_mean', 0.0),
				"rej_sft/gt_txt_pms_sampled_mean": gt_metrics_summary.get('txt_pms_sampled_mean', 0.0),
				"rej_sft/gt_size_l2_dist_mean": gt_metrics_summary.get('size_l2_dist_mean', 0.0),
				"rej_sft/gt_valid_json_rate": gt_metrics_summary.get('valid_json_rate', 0.0),
				"rej_sft/gt_match_rate": gt_match_stats.get("gt_match_rate", 0.0),
				"rej_sft/gt_match_coverage": gt_match_stats.get("gt_match_coverage", 0.0),
				"rej_sft/gt_match_at_least_once": gt_match_stats.get("gt_match_at_least_once", 0),
				"rej_sft/avg_gt_matches_per_prompt": gt_match_stats.get("avg_gt_matches_per_prompt", 0.0),
				
				"rej_sft/round": epoch + 1,
			})

		accelerator.wait_for_everyone()
		print(f"\nround {epoch+1} complete, resuming...\n")

	if accelerator.is_main_process:
		print("REJECTION SAMPLING SFT TRAINING FINISHED!")


def verify_augmentation_determinism(dataset_train, tokenizer, args):
    """Verify that do_augm controls determinism correctly - testing through process_scene_sample"""
    
    all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
    all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
    max_seq_length = 3000
    
    # Take first 5 samples from TRAIN split (important - augmentation only on train)
    test_samples = dataset_train.select(range(5))
    
    print("\n" + "="*80)
    print("AUGMENTATION DETERMINISM TEST (via process_scene_sample)")
    print("="*80)
    
    # Test with do_augm=False (should be deterministic across epochs)
    print("\n1. Testing do_augm=False (should produce identical results across 'epochs'):")
    results_no_augm = []
    for epoch in range(3):
        print(f"\n  Epoch {epoch+1}:")
        
        # Set seeds once per epoch, only if not do_augm (matching your training loop)
        if not (False == True):  # simulating do_augm=False
            np.random.seed(1234)
            random.seed(1234)
            torch.manual_seed(1234)
        
        epoch_results = []
        for sample in test_samples:
            full_instr, completion, orig_prompt, instr_sample = process_scene_sample(
                sample, tokenizer, max_seq_length, all_prompts, 
                all_assets_metadata_simple_descs,
                do_simple_descs=args.do_simple_descs,
                do_augm=False,  # do_augm=False
                do_full_sg_outputs=args.do_full_sg_outputs
            )
            epoch_results.append({
                'full_instr': full_instr,
                'completion': completion,
                'orig_prompt': orig_prompt,
                'sg_input_with_jids': instr_sample['sg_input_with_jids'],
                'sg_output_add_with_jids': instr_sample['sg_output_add_with_jids'],
            })
        results_no_augm.append(epoch_results)
    
    # Check if all epochs are identical
    all_identical = all(
        results_no_augm[0][i] == results_no_augm[j][i]
        for i in range(len(test_samples))
        for j in range(1, 3)
    )
    print(f"\n  ✓ All epochs identical: {all_identical}")
    if not all_identical:
        print("  ✗ WARNING: Results differ across epochs with do_augm=False!")
        for i in range(len(test_samples)):
            for j in range(1, 3):
                if results_no_augm[0][i] != results_no_augm[j][i]:
                    print(f"    Sample {i}: Epoch 0 vs Epoch {j} differ")
                    # Show what differs
                    for key in results_no_augm[0][i].keys():
                        if results_no_augm[0][i][key] != results_no_augm[j][i][key]:
                            print(f"      Key '{key}' differs")
                            if key in ['sg_input_with_jids', 'sg_output_add_with_jids']:
                                # Parse and show first object position for debugging
                                obj0 = json.loads(results_no_augm[0][i][key])
                                objJ = json.loads(results_no_augm[j][i][key])
                                if obj0.get('pos'):
                                    print(f"        Epoch 0 pos: {obj0.get('pos')}")
                                    print(f"        Epoch {j} pos: {objJ.get('pos')}")
    
    # Test with do_augm=True (should be different across epochs)
    print("\n2. Testing do_augm=True (should produce different results across 'epochs'):")
    results_with_augm = []
    for epoch in range(3):
        print(f"\n  Epoch {epoch+1}:")
        
        # No seed setting when do_augm=True
        
        epoch_results = []
        for sample in test_samples:
            full_instr, completion, orig_prompt, instr_sample = process_scene_sample(
                sample, tokenizer, max_seq_length, all_prompts,
                all_assets_metadata_simple_descs,
                do_simple_descs=args.do_simple_descs,
                do_augm=True,  # do_augm=True
                do_full_sg_outputs=args.do_full_sg_outputs
            )
            epoch_results.append({
                'sg_input_with_jids': instr_sample['sg_input_with_jids'],
                'sg_output_add_with_jids': instr_sample['sg_output_add_with_jids'],
            })
        results_with_augm.append(epoch_results)
    
    # Check if epochs are different (they should be due to augmentation)
    any_different = any(
        results_with_augm[0][i] != results_with_augm[j][i]
        for i in range(len(test_samples))
        for j in range(1, 3)
    )
    print(f"\n  ✓ Results differ across epochs: {any_different}")
    if not any_different:
        print("  ✗ WARNING: All results identical with do_augm=True - augmentation not working!")
    else:
        # Show example of differences
        for i in range(min(2, len(test_samples))):
            obj0 = json.loads(results_with_augm[0][i]['sg_output_add_with_jids'])
            obj1 = json.loads(results_with_augm[1][i]['sg_output_add_with_jids'])
            if obj0.get('pos') != obj1.get('pos'):
                print(f"    Sample {i} - position differs (good!):")
                print(f"      Epoch 0: {obj0.get('pos')}")
                print(f"      Epoch 1: {obj1.get('pos')}")
    
    print("\n" + "="*80 + "\n")