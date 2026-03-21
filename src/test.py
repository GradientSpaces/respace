from datetime import datetime
import time
import random
import torch
import json
import wandb
import numpy as np
import re
import math
import os
from filelock import FileLock
import gc
import pdb
from dotenv import load_dotenv
from pathlib import Path
import uuid
import pprint
import traceback
import copy
from tqdm import tqdm
import threading
from collections import defaultdict
import pickle

from src.eval import eval_scene, eval_scene_before_after_with_delta, build_eval_cache_room, compute_dss_score, compute_size_l2_dist, compute_pms_score
from src.dataset import build_full_instruction_from_prompt, load_train_val_test_datasets, create_full_scene_from_before_and_added, clean_copy_of_objects, process_scene_sample, format_with_chat_template
from src.sample import AssetRetrievalModule
from src.utils import set_seeds, remove_and_recreate_folder, safe_parse_scene, inherit_props_by_id, compute_fid_scores, get_system_prompt_sgllm, get_test_instrs_all, get_model, is_high_quality_sample
from src.viz import render_full_scene_and_export_with_gif

def print_and_log_metric(dataset_split, epoch, metric_label, metric_val, use_wandb):
	label = f"[ {dataset_split} ] {'FINAL' if epoch is None else ''} {metric_label}"
	print(f"{label}: {metric_val}")
	if use_wandb: 
		wandb.log({label: metric_val, "epoch": epoch})

def get_batch_input_ids(queries, tokenizer):
	batch_messages = [
		[{ "role": "system", "content": get_system_prompt_sgllm() },
		{ "role": "user", "content": query }] for query in queries
	]

	original_padding_side = tokenizer.padding_side
	tokenizer.padding_side = "left"
	
	encoded = tokenizer.apply_chat_template(
		batch_messages,
		truncation=True,
		padding=True, 
		add_generation_prompt=True, 
		return_tensors="pt",
		return_attention_mask=True,
		return_dict=True
	)
	
	tokenizer.padding_side = original_padding_side

	return encoded.input_ids, encoded.attention_mask

def write_dict_to_file(pth_file, data_for_key, idx_key=None):
	if os.path.exists(pth_file):
		with open(pth_file, 'r') as f:
			data = json.load(f)
	else:
		data = {}

	if idx_key is None:
		idx_key = len(data)

	data[idx_key] = data_for_key
	
	with open(pth_file, 'w') as f:
		json.dump(data, f, indent=4)

def get_sample_outputs_batch(batch_instrs, model, tokenizer, max_seq_length, accelerator, n_best_of_n_llm, return_logits=False, temp=None, vllm_engine=None):

	temp = temp if temp is not None else (0.9 if n_best_of_n_llm > 1 else 0.7)
	# print("temp inside get_sample_outputs_batch is ", temp)

	# ---- vLLM path ----
	if vllm_engine is not None:
		from vllm import SamplingParams

		# format prompts as strings directly, same as run_test_fast
		formatted_prompts = [format_with_chat_template(tokenizer, instr) for instr in batch_instrs]

		# compute max_tokens dynamically based on longest input (same logic as HF path)
		max_input_len = max(len(tokenizer.encode(p)) for p in formatted_prompts)
		max_tokens = min(max_seq_length - max_input_len, 150)
		if max_tokens <= 0:
			max_tokens = 10  # minimal fallback, will likely produce invalid JSON

		sampling_params = SamplingParams(
			n=n_best_of_n_llm,
			max_tokens=max_tokens,
			temperature=temp,
			top_k=50,
			top_p=0.95,
			detokenize=False,
		)

		print("vllm sampling with num of prompts: ", len(formatted_prompts), " max_new_tokens: ", max_tokens)

		# Chunk prompts to avoid CUDA OOM: limit total concurrent sequences (prompts × n)
		max_concurrent_seqs = 64
		chunk_size = max(1, max_concurrent_seqs // n_best_of_n_llm)
		if len(formatted_prompts) <= chunk_size:
			outputs = vllm_engine.generate(formatted_prompts, sampling_params)
		else:
			outputs = []
			for chunk_start in range(0, len(formatted_prompts), chunk_size):
				chunk = formatted_prompts[chunk_start:chunk_start + chunk_size]
				outputs.extend(vllm_engine.generate(chunk, sampling_params))

		# decode manually, same as run_test_fast
		split_responses = []
		for output in outputs:
			if n_best_of_n_llm > 1:
				responses = [tokenizer.decode(o.token_ids, skip_special_tokens=True) for o in output.outputs]
				split_responses.append(responses)
			else:
				split_responses.append(tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True))

		return split_responses

	# ---- HF generate path (unchanged) ----
	batch_input_ids, batch_attention_masks = get_batch_input_ids(batch_instrs, tokenizer)
	
	all_input_ids = []
	all_attention_masks = []
	num_return_sequences = []

	for i in range(len(batch_input_ids)):
		input_ids = batch_input_ids[i]
		attention_mask = batch_attention_masks[i]
		all_input_ids.append(input_ids.repeat(n_best_of_n_llm, 1))
		all_attention_masks.append(attention_mask.repeat(n_best_of_n_llm, 1))
		num_return_sequences.append(n_best_of_n_llm)

	all_input_ids = torch.cat(all_input_ids, dim=0).to(accelerator.device)
	all_attention_masks = torch.cat(all_attention_masks, dim=0).to(accelerator.device)

	is_greedy_sampling = False
	max_tokens = min(max_seq_length - all_input_ids.shape[-1], 150)

	print(f"\nnucleus sampling... batch_input_ids.shape: {batch_input_ids.shape} max_new_tokens: {max_tokens}")

	gen_kwargs = {
		"max_new_tokens": max_tokens,
		"pad_token_id": tokenizer.pad_token_id,
		"attention_mask": all_attention_masks,
		"do_sample": not is_greedy_sampling,
		"temperature": temp,
		"top_k": None if is_greedy_sampling else 50,
		"top_p": None if is_greedy_sampling else 0.95
	}

	torch.use_deterministic_algorithms(False)
	with torch.inference_mode():
		with accelerator.no_sync(model):
			outputs = model.generate(
				input_ids=all_input_ids,
				output_logits=return_logits,
				return_dict_in_generate=return_logits,
				**gen_kwargs
			)

	if return_logits:
		all_output_ids = outputs.sequences[:, all_input_ids.shape[-1]:]
		all_responses = tokenizer.batch_decode(all_output_ids, skip_special_tokens=True)
		all_logits = torch.stack(outputs.scores, dim=1)
	else:
		all_responses = tokenizer.batch_decode(outputs[:, all_input_ids.shape[-1]:], skip_special_tokens=True)

	torch.use_deterministic_algorithms(True)

	split_responses, split_output_ids, split_logits = [], [], []
	idx = 0
	for n in num_return_sequences:
		if n > 1:
			split_responses.append(all_responses[idx:idx + n])
			if return_logits:
				split_output_ids.append(all_output_ids[idx:idx + n])
				split_logits.append(all_logits[idx:idx + n])
		else:
			split_responses.append(all_responses[idx])
			if return_logits:
				split_output_ids.append(all_output_ids[idx])
				split_logits.append(all_logits[idx])
		idx += n

	if return_logits:
		return split_responses, split_output_ids, split_logits
	return split_responses

def prepare_batch(tokenizer, max_seq_length, dataset_split, batch_samples, all_test_instrs, all_prompts, all_assets_metadata_simple_descs, do_simple_descs):
	batch_instr_samples = []
	batch_full_instrs = []
	for sample in batch_samples:
		if dataset_split != "test":
			_, _, _, instr_sample = process_scene_sample(sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, do_augm=False, do_full_sg_outputs=False)
			# instr_sample = create_instruction_from_scene(sample, all_prompts, all_assets_metadata_simple_descs, do_simple_descs)
		else:
			# pick hardcoded sample with fixed random seed for fair comparison
			instr_sample = all_test_instrs.get(sample.get("pth_orig_file"))[1234]
			
			# remove jids from sg_input
			sg_input = json.loads(instr_sample.get("sg_input"))
			sg_input["objects"] = clean_copy_of_objects(sg_input.get("objects"))
			instr_sample["sg_input"] = json.dumps(sg_input)

			# remove jids from sg_output_add
			sg_output_add = json.loads(instr_sample.get("sg_output_add"))
			sg_output_add = clean_copy_of_objects(sg_output_add)
			instr_sample["sg_output_add"] = json.dumps(sg_output_add)
		
		full_instruction = build_full_instruction_from_prompt(instr_sample.get("prompt"), instr_sample.get("sg_input"))

		batch_instr_samples.append(instr_sample)
		batch_full_instrs.append(full_instruction)
	
	return batch_instr_samples, batch_full_instrs

def print_scene_error(exc, dataset_split):
	print("")
	if exc is not None: 
		print(exc)
	print(f"[ {dataset_split} ] ⛔️ COULD NOT EVALUATE GENERATED SCENE")
	print("")

def print_scene_success(dataset_split):
	print("")
	print(f"[ {dataset_split} ] ✅ VALID GENERATED SCENE")
	print("")

def init_best_result():
	return {
		'is_valid_scene_pbl': None,
		'scene': None,

		'total_oob_loss': np.inf,
		'total_mbl_loss': np.inf,
		'total_pbl_loss': np.inf,

		'delta_oob_loss': np.inf,
		'delta_mbl_loss': np.inf,
		'delta_pbl_loss': np.inf,
	}


def run_test_for_addition(scene_after, scene_before_with_assets, n_best_of_n_assets, sampling_engine=None, gpu_lock=None, before_metrics=None, eval_cache=None, idx=None):
	start_time = time.time()

	best_result = init_best_result()

	if scene_after.get("objects") is None:
		scene_after = create_full_scene_from_before_and_added(scene_before_with_assets, scene_after)
		# print(f"[ bon_idx={idx} ] create_full_scene_from_before_and_added() took {time.time() - start_time:.2f} seconds")

	for _ in range(n_best_of_n_assets):

		# sample only last object for sample_scene and inherit other samples from scene_before
		if sampling_engine is not None:
			if gpu_lock is not None:
				with gpu_lock:
					scene_after_with_assets = sampling_engine.sample_last_asset(scene_after, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False))
					inherit_props_by_id(scene_before_with_assets, scene_after_with_assets)
			else:
				scene_after_with_assets = sampling_engine.sample_last_asset(scene_after, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False))
				inherit_props_by_id(scene_before_with_assets, scene_after_with_assets)
		else:
			scene_after_with_assets = scene_after

		# print(f"[ bon_idx={idx} ] sample_last_asset() took {time.time() - start_time:.2f} seconds")

		result = eval_scene_before_after_with_delta(scene_before_with_assets, scene_after_with_assets, is_debug=False, before_metrics=before_metrics, eval_cache=eval_cache)
		# result_old = eval_scene_before_after_with_delta_old(scene_before_with_assets, scene_after_with_assets, is_debug=False, before_metrics=before_metrics, eval_cache=eval_cache)

		if result['delta_pbl_loss'] < best_result['delta_pbl_loss']:
			best_result = result

		# print(f"[ bon_idx={idx} ] eval_scene_before_after_with_delta() took {time.time() - start_time:.2f} seconds - delta_pbl_loss: {result['delta_pbl_loss']} - txt_pms_sampled_score: {result.get('txt_pms_sampled_score')} - size_l2_dist: {result.get('size_l2_dist')}")
		# print(f"[ bon_idx={idx} ] OLD eval_scene_before_after_with_delta_old() took {time.time() - start_time:.2f} seconds - delta_pbl_loss: {result_old['delta_pbl_loss']} - txt_pms_sampled_score: {result_old.get('txt_pms_sampled_score')} - size_l2_dist: {result_old.get('size_l2_dist')}")

	return best_result

def process_and_render_result(best_result, pth_viz_output, filename=None, prompt=None, do_renderings=False, show_bboxes_also=False):
	# print("running process_and_render_result")
	if not best_result['scene']:
		return
		
	if do_renderings:
		render_full_scene_and_export_with_gif(best_result['scene'], filename=filename if filename else "current", pth_output=pth_viz_output, create_gif=False, show_bboxes_also=show_bboxes_also)
	
	if prompt is not None:
		scene_metrics = copy.deepcopy(best_result)
		scene_metrics['prompt'] = prompt

		write_dict_to_file(pth_viz_output / "metrics.json", json.dumps(scene_metrics))
		write_dict_to_file(pth_viz_output / "scene.json", json.dumps(best_result['scene']))	

def run_bon_test_for_addition(
	prompt, responses, scene_before_with_assets, n_best_of_n_assets,
	sampling_engine, pth_viz_output=None, filename=None,
	do_rendering_with_object_count=False, do_renderings=False,
	num_workers=1, eval_cache=None, before_metrics=None
):
	best_result = init_best_result()
	responses = responses if isinstance(responses, list) else [responses]

	gpu_lock = threading.Lock() if num_workers > 1 else None

	# pre-compute room mesh + scene_before metrics once (shared across all BoN responses)
	if eval_cache is None:
		eval_cache = build_eval_cache_room(scene_before_with_assets)
	if before_metrics is None:
		before_metrics = eval_scene(scene_before_with_assets, is_debug=False, eval_cache=eval_cache)

	def _process_one_response(args):
		start_time = time.time()
		idx, response = args
		scene_after = safe_parse_scene(response)
		if not scene_after:
			return None
		try:
			if scene_after.get("objects") is None:
				scene_after["prompt"] = prompt
			else:
				scene_after["objects"][-1]["prompt"] = prompt

			result = run_test_for_addition(scene_after, scene_before_with_assets, n_best_of_n_assets, sampling_engine, gpu_lock=gpu_lock, before_metrics=before_metrics, eval_cache=eval_cache, idx=idx)

			if pth_viz_output and do_rendering_with_object_count:
				n_objs = len(scene_before_with_assets.get("objects", [])) + 1
				process_and_render_result(result, pth_viz_output / f"samples-n-{n_objs}", 
					filename=f"n{n_objs}-idx{idx}-dpbl{str(round(result.get('delta_pbl_loss'), 4))}", 
					do_renderings=do_renderings, show_bboxes_also=True)
				pth_json = pth_viz_output / f"samples-n-{n_objs}" / f"n{n_objs}-idx{idx}-dpbl{str(round(result.get('delta_pbl_loss'), 4))}.json"
				with pth_json.open("w") as f:
					json.dump(result.get("scene"), f, indent=4)

			# if len(responses) > 1:
			print(f"[ bon_idx={idx} ] run test for addition ... took {time.time() - start_time:.2f} seconds")

			return (idx, result)
		except Exception:
			traceback.print_exc()
			print("some error during run_bon_test_for_addition. skipping current sample")
			return None

	indexed_responses = list(enumerate(responses))
	
	if num_workers > 1 and len(responses) > 1:
		from concurrent.futures import ThreadPoolExecutor
		with ThreadPoolExecutor(max_workers=num_workers) as pool:
			raw_results = list(pool.map(_process_one_response, indexed_responses))
	else:
		raw_results = [_process_one_response(item) for item in indexed_responses]

	all_results = [r for r in raw_results if r is not None]

	if all_results:
		max_pms_score = max(r[1].get('txt_pms_sampled_score', 0) for r in all_results)
		best_pms_results = [r for r in all_results if r[1].get('txt_pms_sampled_score', 0) == max_pms_score]
		best_idx, best_result = min(best_pms_results, key=lambda x: x[1].get('delta_pbl_loss', float('inf')))
		best_result["best_idx"] = best_idx

	if pth_viz_output and filename is not None:
		process_and_render_result(best_result, pth_viz_output, filename=filename, do_renderings=do_renderings)

	return best_result

def run_instr(prompt, scene_before_with_assets, batch_full_instrs, model, tokenizer, max_seq_length, accelerator, best_of_n, n_best_of_n_assets, sampling_engine, pth_viz_output, do_rendering_with_object_count, temp=None, vllm_engine=None, num_workers=1):
	
	# pth_export = Path(pth_output) / "best-of-n" / str(idx)

	# get best-of-n responses for each sample
	responses = get_sample_outputs_batch(batch_full_instrs, model, tokenizer, max_seq_length, accelerator, best_of_n, return_logits=False, temp=temp, vllm_engine=vllm_engine)

	best_result = run_bon_test_for_addition(prompt, responses[0], scene_before_with_assets, n_best_of_n_assets, sampling_engine, pth_viz_output=pth_viz_output, do_rendering_with_object_count=do_rendering_with_object_count, do_renderings=True, num_workers=num_workers)
	
	# process_and_render_result(best_result, pth_export / "current", filename="current", prompt=prompt, do_renderings=True)

	return best_result

def initialize_file(filepath, process_index):
	if process_index == 0:
		if os.path.exists(filepath):
			os.remove(filepath)
		if os.path.exists(filepath + ".lock"):
			os.remove(filepath + ".lock")
		print(f"idx [{process_index}]: deleted existing file(s)!")

def write_metrics_to_file(metrics, pth_file, process_index, num_processes):
	print(f"idx [{process_index}] wants to write to metrics file!")
	
	lock_path = pth_file + ".lock"
	with FileLock(lock_path):
		write_dict_to_file(pth_file, metrics, str(process_index))
	print(f"idx [{process_index}] has finished writing metrics file!")
	
	while True:
		with FileLock(lock_path):
			with open(pth_file, 'r') as f:
				all_metrics = json.load(f)
			if len(all_metrics) == num_processes:
				print(f"idx [{process_index}] ok lengths match!")
				return all_metrics  # Return while still holding the lock
			print(f"idx [{process_index}] still waiting for other process to finish (lengths do not match)")
		time.sleep(5.0)

def aggregate_metrics(all_metrics):

	aggregated = {
		"num_scenes": 0,
		"num_scenes_loss_metrics": 0,

		"num_valid_instrs_by_pbl_sum": 0,
		"num_high_quality_sum": 0,

		"scene_total_oob_loss_sum": 0,
		"scene_total_mbl_loss_sum": 0,
		"scene_total_pbl_loss_sum": 0,

		"scene_delta_oob_loss_sum": 0,
		"scene_delta_mbl_loss_sum": 0,
		"scene_delta_pbl_loss_sum": 0,

		"scene_size_l2_dist_sum": 0,
		"scene_size_m3_vol_sum": 0,

		"txt_pms_score_sum": 0,
		"txt_pms_sampled_score_sum": 0,
		"txt_dss_score_sum": 0,
	}
	
	for metrics in all_metrics.values():
		aggregated["num_valid_instrs_by_pbl_sum"] += metrics["subset_num_valid_instrs_by_pbl"]
		aggregated["num_high_quality_sum"] += metrics.get("subset_num_high_quality", 0)
		aggregated["num_scenes"] += metrics["subset_num_scenes"]
		aggregated["num_scenes_loss_metrics"] += metrics["subset_num_scenes_loss_metrics"]

		aggregated["scene_total_oob_loss_sum"] += metrics["subset_scene_total_oob_loss_sum"]
		aggregated["scene_total_mbl_loss_sum"] += metrics["subset_scene_total_mbl_loss_sum"]
		aggregated["scene_total_pbl_loss_sum"] += metrics["subset_scene_total_pbl_loss_sum"]

		aggregated["scene_delta_oob_loss_sum"] += metrics["subset_scene_delta_oob_loss_sum"]
		aggregated["scene_delta_mbl_loss_sum"] += metrics["subset_scene_delta_mbl_loss_sum"]
		aggregated["scene_delta_pbl_loss_sum"] += metrics["subset_scene_delta_pbl_loss_sum"]

		aggregated["scene_size_l2_dist_sum"] += metrics["subset_scene_size_l2_dist_sum"]
		aggregated["scene_size_m3_vol_sum"] += metrics["subset_scene_size_m3_vol_sum"]

		aggregated["txt_pms_score_sum"] += metrics["subset_txt_pms_score_sum"]
		aggregated["txt_pms_sampled_score_sum"] += metrics["subset_txt_pms_sampled_score_sum"]
		aggregated["txt_dss_score_sum"] += metrics["subset_txt_dss_score_sum"]

	aggregated["num_valid_instrs_by_pbl_ratio"] = aggregated["num_valid_instrs_by_pbl_sum"] / aggregated["num_scenes"]
	aggregated["num_valid_instrs_by_json_ratio"] = aggregated["num_scenes_loss_metrics"] / aggregated["num_scenes"]
	aggregated["filter_pass_rate"] = aggregated["num_high_quality_sum"] / aggregated["num_scenes_loss_metrics"] if aggregated["num_scenes_loss_metrics"] > 0 else 0
	
	if aggregated["num_scenes_loss_metrics"] > 0:
		aggregated["scene_total_oob_loss"] = aggregated["scene_total_oob_loss_sum"] / aggregated["num_scenes_loss_metrics"]
		aggregated["scene_total_mbl_loss"] = aggregated["scene_total_mbl_loss_sum"] / aggregated["num_scenes_loss_metrics"]
		aggregated["scene_total_pbl_loss"] = aggregated["scene_total_pbl_loss_sum"] / aggregated["num_scenes_loss_metrics"]

		aggregated["scene_delta_oob_loss"] = aggregated["scene_delta_oob_loss_sum"] / aggregated["num_scenes_loss_metrics"]
		aggregated["scene_delta_mbl_loss"] = aggregated["scene_delta_mbl_loss_sum"] / aggregated["num_scenes_loss_metrics"]
		aggregated["scene_delta_pbl_loss"] = aggregated["scene_delta_pbl_loss_sum"] / aggregated["num_scenes_loss_metrics"]

		aggregated["scene_size_l2_dist"] = aggregated["scene_size_l2_dist_sum"] / aggregated["num_scenes_loss_metrics"]
		aggregated["scene_size_m3_vol"] = aggregated["scene_size_m3_vol_sum"] / aggregated["num_scenes_loss_metrics"]

		aggregated["txt_pms_score"] = aggregated["txt_pms_score_sum"] / aggregated["num_scenes_loss_metrics"]
		aggregated["txt_pms_sampled_score"] = aggregated["txt_pms_sampled_score_sum"] / aggregated["num_scenes_loss_metrics"]
		aggregated["txt_dss_score"] = aggregated["txt_dss_score_sum"] / aggregated["num_scenes_loss_metrics"]
	else:
		aggregated["scene_total_oob_loss"] = float('inf')
		aggregated["scene_total_mbl_loss"] = float('inf')
		aggregated["scene_total_pbl_loss"] = float('inf')

		aggregated["scene_delta_oob_loss"] = float('inf')
		aggregated["scene_delta_mbl_loss"] = float('inf')
		aggregated["scene_delta_pbl_loss"] = float('inf')

		aggregated["scene_size_l2_dist"] = float('inf')
		aggregated["scene_size_m3_vol"] = float('inf')

		aggregated["txt_pms_score"] = float('inf')
		aggregated["txt_pms_sampled_score"] = float('inf')
		aggregated["txt_dss_score"] = float('inf')
	
	return aggregated

def _score_test_sample(args_tuple):
	"""Worker for scoring one test sample's generated response (for threaded scoring in run_test_fast)."""
	instr_sample, sample_response, sampling_engine, n_best_of_n_assets, gpu_lock = args_tuple

	result = {
		"success": False,
		"is_valid_scene_pbl": False,
		"is_high_quality": False,
		"total_oob_loss": None,
		"total_mbl_loss": None,
		"total_pbl_loss": None,
		"delta_oob_loss": None,
		"delta_mbl_loss": None,
		"delta_pbl_loss": None,
		"size_l2_dist": None,
		"size_m3_vol": None,
		"txt_pms_score": None,
		"txt_pms_sampled_score": None,
		"txt_dss_score": None,
		"desc_length": None,
	}

	try:
		scene_before = json.loads(instr_sample.get("sg_input"))
		
		if gpu_lock is not None:
			with gpu_lock:
				scene_before_with_assets = sampling_engine.sample_all_assets(scene_before, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False))
		else:
			scene_before_with_assets = sampling_engine.sample_all_assets(scene_before, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False))

		best_result = run_bon_test_for_addition(instr_sample.get("prompt"), sample_response, scene_before_with_assets, n_best_of_n_assets, sampling_engine, num_workers=1)

		if best_result.get("scene") is None:
			print(f"_score_test_sample: 'scene' in best_result is None")
			return result

		new_obj_desc = best_result.get("scene").get("objects")[-1].get("desc")
		gt_obj_desc = json.loads(instr_sample.get("sg_output_add")).get("desc")
		txt_dss_score = compute_dss_score(new_obj_desc, gt_obj_desc, sampling_engine)

		new_obj_size = best_result.get("scene").get("objects")[-1].get("size")
		gt_obj_size = json.loads(instr_sample.get("sg_output_add")).get("size")
		size_l2_dist = compute_size_l2_dist(new_obj_size, gt_obj_size)
		size_m3_vol = math.prod(new_obj_size)

		if best_result["delta_pbl_loss"] < 0.0:
			print(f"_score_test_sample: delta_pbl is negative ({best_result['delta_pbl_loss']})")
			return result
		
		is_high_quality = is_high_quality_sample(
			best_result.get("delta_pbl_loss", float('inf')),
			best_result.get("txt_pms_sampled_score", 0.0),
			size_l2_dist,
			txt_dss_score,
			desc=new_obj_desc
		)

		result["success"] = True
		result["is_valid_scene_pbl"] = best_result.get("is_valid_scene_pbl", False)
		result["is_high_quality"] = is_high_quality
		result["total_oob_loss"] = best_result.get("total_oob_loss")
		result["total_mbl_loss"] = best_result.get("total_mbl_loss")
		result["total_pbl_loss"] = best_result.get("total_pbl_loss")
		result["delta_oob_loss"] = best_result.get("delta_oob_loss")
		result["delta_mbl_loss"] = best_result.get("delta_mbl_loss")
		result["delta_pbl_loss"] = best_result.get("delta_pbl_loss")
		result["size_l2_dist"] = size_l2_dist
		result["size_m3_vol"] = size_m3_vol
		result["txt_pms_score"] = best_result.get("txt_pms_score")
		result["txt_pms_sampled_score"] = best_result.get("txt_pms_sampled_score")
		result["txt_dss_score"] = txt_dss_score
		result["desc_length"] = len(new_obj_desc.split()) if new_obj_desc else 0

	except Exception as exc:
		print(f"_score_test_sample error: {exc}")
		traceback.print_exc()

	return result


def run_test_fast(llm_engine, tokenizer, accelerator, dvc, dataset_split, room_type, dataset, max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, args, do_print=False, epoch=None, num_workers=1):
	"""Fast evaluation using vLLM for batch generation + threaded scoring. Runs on main process only."""
	print(f"\n[ {dataset_split} ] running fast tests (vLLM + threaded scoring, num_workers={num_workers})...\n")

	total_samples = len(dataset)
	all_test_instrs = get_test_instrs_all(room_type)

	# ---- Step 1: Prepare all prompts ----
	all_instr_samples = []
	all_formatted_prompts = []

	random.seed(1234)
	np.random.seed(1234)

	for idx in range(total_samples):
		batch_instrs, batch_full_instrs = prepare_batch(tokenizer, max_seq_length, dataset_split, dataset.select(range(idx, idx + 1)), all_test_instrs, all_prompts, all_assets_metadata_simple_descs, do_simple_descs)
		all_instr_samples.append(batch_instrs[0])
		all_formatted_prompts.append(format_with_chat_template(tokenizer, batch_full_instrs[0]))

	print(f"prepared {len(all_formatted_prompts)} prompts for vLLM generation")

	# ---- Step 2: Batch generate with vLLM ----
	from vllm import SamplingParams
	eval_sampling_params = SamplingParams(
		n=1,
		max_tokens=150,
		temperature=0.7,
		top_k=50,
		top_p=0.95,
		detokenize=False,
	)

	chunk_size = 128
	all_outputs = []
	for chunk_idx, i in enumerate(range(0, len(all_formatted_prompts), chunk_size)):
		chunk = all_formatted_prompts[i:i+chunk_size]
		print(f"chunk {chunk_idx+1}/{(len(all_formatted_prompts) + chunk_size - 1) // chunk_size} starting forward pass through vLLM")
		outputs = llm_engine.generate(chunk, eval_sampling_params, use_tqdm=False)
		print("finished forward pass through vLLM")
		all_outputs.extend(outputs)
		print(f"generated {min(i+chunk_size, len(all_formatted_prompts))}/{len(all_formatted_prompts)} prompts")

	# Decode responses (n=1, so single response per prompt)
	all_responses = []
	for output in all_outputs:
		response = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)
		all_responses.append(response)

	print(f"decoded {len(all_responses)} responses")

	gpu_lock = threading.Lock() if num_workers > 1 else None

	# ---- Step 3: Score in parallel ----
	n_best_of_n_assets = 1
	work_items = [
    	(instr_sample, response, sampling_engine, n_best_of_n_assets, gpu_lock)
    	for instr_sample, response in zip(all_instr_samples, all_responses)
	]

	print(f"scoring {len(work_items)} samples (num_workers={num_workers})...")
	start_time = time.time()

	if num_workers > 1:
		from concurrent.futures import ThreadPoolExecutor
		with ThreadPoolExecutor(max_workers=num_workers) as pool:
			scored_results = list(pool.map(_score_test_sample, work_items))
	else:
		scored_results = [_score_test_sample(item) for item in work_items]

	elapsed = time.time() - start_time
	print(f"finished scoring {len(scored_results)} samples in {elapsed:.1f}s")

	# ---- Step 4: Aggregate metrics ----
	num_valid_instrs_by_pbl = 0
	num_high_quality = 0
	all_total_oob, all_total_mbl, all_total_pbl = [], [], []
	all_delta_oob, all_delta_mbl, all_delta_pbl = [], [], []
	all_size_l2, all_size_m3 = [], []
	all_pms, all_pms_sampled, all_dss = [], [], []
	all_desc_lengths = []

	for r in scored_results:
		if not r["success"]:
			continue
		if r["is_valid_scene_pbl"]:
			num_valid_instrs_by_pbl += 1
		if r["is_high_quality"]:
			num_high_quality += 1
		all_total_oob.append(r["total_oob_loss"])
		all_total_mbl.append(r["total_mbl_loss"])
		all_total_pbl.append(r["total_pbl_loss"])
		all_delta_oob.append(r["delta_oob_loss"])
		all_delta_mbl.append(r["delta_mbl_loss"])
		all_delta_pbl.append(r["delta_pbl_loss"])
		all_size_l2.append(r["size_l2_dist"])
		all_size_m3.append(r["size_m3_vol"])
		all_pms.append(r["txt_pms_score"])
		all_pms_sampled.append(r["txt_pms_sampled_score"])
		all_dss.append(r["txt_dss_score"])
		if r["desc_length"] is not None:
			all_desc_lengths.append(r["desc_length"])

	n_scored = len(all_total_oob)

	aggregated_metrics = {
		"num_scenes": total_samples,
		"num_scenes_loss_metrics": n_scored,
		"num_valid_instrs_by_pbl_sum": num_valid_instrs_by_pbl,
		"num_valid_instrs_by_pbl_ratio": num_valid_instrs_by_pbl / total_samples if total_samples > 0 else 0,
		"num_valid_instrs_by_json_ratio": n_scored / total_samples if total_samples > 0 else 0,
		"filter_pass_rate": num_high_quality / n_scored if n_scored > 0 else 0,
	}

	if n_scored > 0:
		aggregated_metrics["scene_total_oob_loss"] = np.mean(all_total_oob)
		aggregated_metrics["scene_total_mbl_loss"] = np.mean(all_total_mbl)
		aggregated_metrics["scene_total_pbl_loss"] = np.mean(all_total_pbl)
		aggregated_metrics["scene_delta_oob_loss"] = np.mean(all_delta_oob)
		aggregated_metrics["scene_delta_mbl_loss"] = np.mean(all_delta_mbl)
		aggregated_metrics["scene_delta_pbl_loss"] = np.mean(all_delta_pbl)
		aggregated_metrics["scene_size_l2_dist"] = np.mean(all_size_l2)
		aggregated_metrics["scene_size_m3_vol"] = np.mean(all_size_m3)
		aggregated_metrics["txt_pms_score"] = np.mean(all_pms)
		aggregated_metrics["txt_pms_sampled_score"] = np.mean(all_pms_sampled)
		aggregated_metrics["txt_dss_score"] = np.mean(all_dss)
		aggregated_metrics["desc_length_avg"] = np.mean(all_desc_lengths) if len(all_desc_lengths) > 0 else 0
	else:
		aggregated_metrics["scene_total_oob_loss"] = float('inf')
		aggregated_metrics["scene_total_mbl_loss"] = float('inf')
		aggregated_metrics["scene_total_pbl_loss"] = float('inf')
		aggregated_metrics["scene_delta_oob_loss"] = float('inf')
		aggregated_metrics["scene_delta_mbl_loss"] = float('inf')
		aggregated_metrics["scene_delta_pbl_loss"] = float('inf')
		aggregated_metrics["scene_size_l2_dist"] = float('inf')
		aggregated_metrics["scene_size_m3_vol"] = float('inf')
		aggregated_metrics["txt_pms_score"] = float('inf')
		aggregated_metrics["txt_pms_sampled_score"] = float('inf')
		aggregated_metrics["txt_dss_score"] = float('inf')
		aggregated_metrics["desc_length_avg"] = 0

	# ---- Step 5: FID scores + Print + Log ----
	pth_viz_output = Path(f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/run-test-subset-{dataset_split}")

	compute_fid_scores("diag", f"3d-front-train-instr-scenes-{room_type}-diag", f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/3d-front-train-instr-scenes-{room_type}/diag", f"{pth_viz_output}/diag", args.do_renderings, aggregated_metrics, dataset_res=1024)
	compute_fid_scores("top", f"3d-front-train-instr-scenes-{room_type}-top", f"{os.getenv('PTH_EVAL_VIZ_CACHE')})/3d-front-train-instr-scenes-{room_type}/top", f"{pth_viz_output}/top", args.do_renderings, aggregated_metrics, dataset_res=1024)

	print("")
	print(f"==== eval for [ {dataset_split} ] dataset ({aggregated_metrics['num_scenes']} samples) ====")
	print("")
	print_and_log_metric(dataset_split, epoch, "num_valid_instrs_by_pbl_ratio", aggregated_metrics["num_valid_instrs_by_pbl_ratio"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "num_valid_instrs_by_json_ratio", aggregated_metrics["num_valid_instrs_by_json_ratio"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "filter_pass_rate", aggregated_metrics["filter_pass_rate"], args.use_wandb)
	print("")
	print_and_log_metric(dataset_split, epoch, "scene_total_oob_loss", aggregated_metrics["scene_total_oob_loss"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "scene_total_mbl_loss", aggregated_metrics["scene_total_mbl_loss"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "scene_total_pbl_loss", aggregated_metrics["scene_total_pbl_loss"], args.use_wandb)
	print("")
	print_and_log_metric(dataset_split, epoch, "scene_delta_oob_loss", aggregated_metrics["scene_delta_oob_loss"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "scene_delta_mbl_loss", aggregated_metrics["scene_delta_mbl_loss"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "scene_delta_pbl_loss", aggregated_metrics["scene_delta_pbl_loss"], args.use_wandb)
	print("")
	print_and_log_metric(dataset_split, epoch, "scene_size_l2_dist", aggregated_metrics["scene_size_l2_dist"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "scene_size_m3_vol", aggregated_metrics["scene_size_m3_vol"], args.use_wandb)
	print("")
	print_and_log_metric(dataset_split, epoch, "txt_pms_score", aggregated_metrics["txt_pms_score"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "txt_pms_sampled_score", aggregated_metrics["txt_pms_sampled_score"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "txt_dss_score", aggregated_metrics["txt_dss_score"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "desc_length_avg", aggregated_metrics["desc_length_avg"], args.use_wandb)
	print("")
	print_and_log_metric(dataset_split, epoch, "fid_score_diag", aggregated_metrics["fid_score_diag"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "fid_clip_score_diag", aggregated_metrics["fid_clip_score_diag"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "kid_score_diag", aggregated_metrics["kid_score_diag"], args.use_wandb)
	print("")
	print_and_log_metric(dataset_split, epoch, "fid_score_top", aggregated_metrics["fid_score_top"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "fid_clip_score_top", aggregated_metrics["fid_clip_score_top"], args.use_wandb)
	print_and_log_metric(dataset_split, epoch, "kid_score_top", aggregated_metrics["kid_score_top"], args.use_wandb)
	print("")
	print("==== ============================================================== ====")

	# write final metrics to file
	pth_final_metrics = f"./ckpts/{args.jid}/metrics_{dataset_split}_final.json"
	initialize_file(pth_final_metrics, 0)
	write_dict_to_file(pth_final_metrics, aggregated_metrics)

	gc.collect()
	torch.cuda.empty_cache()

	return aggregated_metrics


def run_test(model, tokenizer, accelerator, dvc, dataset_split, room_type, dataset, max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, args, n_best_of_n_llm=1, n_best_of_n_assets=1, do_print=False, epoch=None):
	print(f"\n[ {dataset_split} ] running tests...\n")

	total_samples = len(dataset)
	num_processes = accelerator.num_processes
	process_index = accelerator.process_index
	samples_per_process = (total_samples + num_processes - 1) // num_processes
	start_idx = process_index * samples_per_process
	end_idx = min(start_idx + samples_per_process, total_samples)
	subset_dataset = dataset.select(range(start_idx, end_idx))
	batch_size = min(len(subset_dataset), args.test_bs)

	pth_metrics = f"./ckpts/{args.jid}/metrics_{dataset_split}.json"
	initialize_file(pth_metrics, accelerator.process_index)

	subset_num_valid_instrs_by_pbl = torch.zeros(1, device=accelerator.device)
	subset_num_high_quality = torch.zeros(1, device=accelerator.device)
	all_subset_scene_total_oob_loss, all_subset_scene_total_mbl_loss, all_subset_scene_total_pbl_loss = [], [], []
	all_subset_scene_delta_oob_loss, all_subset_scene_delta_mbl_loss, all_subset_scene_delta_pbl_loss = [], [], []
	all_subset_scene_size_l2_dist, all_subset_scene_size_m3_vol = [], []
	all_subset_txt_pms_score, all_subset_txt_pms_sampled_score, all_subset_txt_dss_score = [], [], []

	all_test_instrs = get_test_instrs_all(room_type)

	idx = 0
	for batch_idx in range(0, len(subset_dataset), batch_size):
		print("\n==========================================")
		print(f"idx [{accelerator.process_index}] — {dataset_split} — epoch {epoch} — batch {int((batch_idx/batch_size) + 1)}/{int(math.ceil(len(subset_dataset)/batch_size))}")
		print("============================================")
		
		# sample instructions and bring into input prompt style
		end_idx = min(batch_idx + batch_size, len(subset_dataset))
		batch_instrs, batch_full_instrs = prepare_batch(tokenizer, max_seq_length, dataset_split, subset_dataset.select(range(batch_idx, end_idx)), all_test_instrs, all_prompts, all_assets_metadata_simple_descs, do_simple_descs)

		# make forward passes
		batch_responses = get_sample_outputs_batch(batch_full_instrs, model, tokenizer, max_seq_length, accelerator, n_best_of_n_llm, return_logits=False)

		# viz folder
		pth_viz_output = Path(f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/run-test-subset-{dataset_split}")
		
		for sample, sample_response in zip(batch_instrs, batch_responses):
			try:
				scene_before = json.loads(sample.get("sg_input"))
				scene_before_with_assets = sampling_engine.sample_all_assets(scene_before, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False))

				# print(f"n_objects_query: {sample.get('n_objects_query')}, n_objects_full: {sample.get('n_objects_full')}, response ===", sample_response, "===")
				best_result = run_bon_test_for_addition(sample.get("prompt"), sample_response, scene_before_with_assets, n_best_of_n_assets, sampling_engine, pth_viz_output=pth_viz_output, filename=idx, do_renderings=args.do_renderings)

				if best_result.get("scene") is None:
					print_scene_error("'scene' in best_result is None. can not evaluate generated scene", dataset_split)
					print(f"> generated response: {sample_response}")
					print(f"> GT response: {sample.get('sg_output_add')}")
					continue

				new_obj_desc = best_result.get("scene").get("objects")[-1].get("desc")
				gt_obj_desc = json.loads(sample.get("sg_output_add")).get("desc")
				txt_dss_score = compute_dss_score(new_obj_desc, gt_obj_desc, sampling_engine)

				new_obj_size = best_result.get("scene").get("objects")[-1].get("size")
				gt_obj_size = json.loads(sample.get("sg_output_add")).get("size")
				size_l2_dist = compute_size_l2_dist(new_obj_size, gt_obj_size)
				size_m3_vol = math.prod(new_obj_size)

				if best_result["delta_pbl_loss"] < 0.0:
					print_scene_error("best delta_pbl is negative... model failed to complete scene (objects before addition are not consistent)", dataset_split)
					print(best_result.get("delta_pbl_loss"))
					continue
				else:
					if best_result.get("is_valid_scene_pbl"):
						subset_num_valid_instrs_by_pbl += 1
				
				is_high_quality = is_high_quality_sample(
					best_result.get("delta_pbl_loss", float('inf')),
					best_result.get("txt_pms_sampled_score", 0.0),
					size_l2_dist,
					txt_dss_score,
					desc=new_obj_desc
				)
				if is_high_quality:
					subset_num_high_quality += 1

				if do_print:
					print(f"[ {dataset_split} ] is_valid_scene_pbl: {best_result.get('is_valid_scene_pbl')}")
					print("")
					print(f"[ {dataset_split} ] total_oob_loss: {best_result.get('total_oob_loss')}")
					print(f"[ {dataset_split} ] total_mbl_loss: {best_result.get('total_mbl_loss')}")
					print(f"[ {dataset_split} ] total_pbl_loss: {best_result.get('total_pbl_loss')}")
					print("")
					print(f"[ {dataset_split} ] delta_oob_loss: {best_result.get('delta_oob_loss')}")
					print(f"[ {dataset_split} ] delta_mbl_loss: {best_result.get('delta_mbl_loss')}")
					print(f"[ {dataset_split} ] delta_pbl_loss: {best_result.get('delta_pbl_loss')}")
					print("")
					print(f"[ {dataset_split} ] size_l2_dist: {size_l2_dist}")
					print(f"[ {dataset_split} ] size_m3_vol: {size_m3_vol}")
					print("")
					print(f"[ {dataset_split} ] txt_pms_score: {best_result.get('txt_pms_score')}")
					print(f"[ {dataset_split} ] txt_dss_score: {txt_dss_score}")

				all_subset_scene_total_oob_loss.append(torch.tensor(best_result.get("total_oob_loss"), device=accelerator.device))
				all_subset_scene_total_mbl_loss.append(torch.tensor(best_result.get("total_mbl_loss"), device=accelerator.device))
				all_subset_scene_total_pbl_loss.append(torch.tensor(best_result.get("total_pbl_loss"), device=accelerator.device))

				all_subset_scene_delta_oob_loss.append(torch.tensor(best_result.get("delta_oob_loss"), device=accelerator.device))
				all_subset_scene_delta_mbl_loss.append(torch.tensor(best_result.get("delta_mbl_loss"), device=accelerator.device))
				all_subset_scene_delta_pbl_loss.append(torch.tensor(best_result.get("delta_pbl_loss"), device=accelerator.device))

				all_subset_scene_size_l2_dist.append(torch.tensor(size_l2_dist, device=accelerator.device))
				all_subset_scene_size_m3_vol.append(torch.tensor(size_m3_vol, device=accelerator.device))
				
				all_subset_txt_pms_score.append(torch.tensor(best_result.get('txt_pms_score'), device=accelerator.device))
				all_subset_txt_pms_sampled_score.append(torch.tensor(best_result.get('txt_pms_sampled_score'), device=accelerator.device))
				all_subset_txt_dss_score.append(torch.tensor(txt_dss_score, device=accelerator.device))

			except Exception as exc:
				print(traceback.format_exc())
				print(f"> generated response: {sample_response}")
				print(f"> GT response: {sample.get('sg_output_add')}")
				print_scene_error(exc, dataset_split)

			idx += 1

		gc.collect()
		torch.cuda.empty_cache()

	print(f"idx [{accelerator.process_index}] finished while loop for all batches")

	metrics = {
		"subset_num_valid_instrs_by_pbl": subset_num_valid_instrs_by_pbl.item(),
		"subset_num_high_quality": subset_num_high_quality.item(),
		"subset_num_scenes": len(subset_dataset),
		"subset_num_scenes_loss_metrics": len(all_subset_scene_total_oob_loss),
		
		"subset_scene_total_oob_loss_sum": torch.stack(all_subset_scene_total_oob_loss).sum().item() if len(all_subset_scene_total_oob_loss) > 0 else float('inf'),
		"subset_scene_total_mbl_loss_sum": torch.stack(all_subset_scene_total_mbl_loss).sum().item() if len(all_subset_scene_total_mbl_loss) > 0 else float('inf'),
		"subset_scene_total_pbl_loss_sum": torch.stack(all_subset_scene_total_pbl_loss).sum().item() if len(all_subset_scene_total_pbl_loss) > 0 else float('inf'),

		"subset_scene_delta_oob_loss_sum": torch.stack(all_subset_scene_delta_oob_loss).sum().item() if len(all_subset_scene_delta_oob_loss) > 0 else float('inf'),
		"subset_scene_delta_mbl_loss_sum": torch.stack(all_subset_scene_delta_mbl_loss).sum().item() if len(all_subset_scene_delta_mbl_loss) > 0 else float('inf'),
		"subset_scene_delta_pbl_loss_sum": torch.stack(all_subset_scene_delta_pbl_loss).sum().item() if len(all_subset_scene_delta_pbl_loss) > 0 else float('inf'),

		"subset_scene_size_l2_dist_sum": torch.stack(all_subset_scene_size_l2_dist).sum().item() if len(all_subset_scene_size_l2_dist) > 0 else float('inf'),
		"subset_scene_size_m3_vol_sum": torch.stack(all_subset_scene_size_m3_vol).sum().item() if len(all_subset_scene_size_m3_vol) > 0 else float('inf'),

		"subset_txt_pms_score_sum": torch.stack(all_subset_txt_pms_score).sum().item() if len(all_subset_txt_pms_score) > 0 else float('inf'),
		"subset_txt_pms_sampled_score_sum": torch.stack(all_subset_txt_pms_sampled_score).sum().item() if len(all_subset_txt_pms_sampled_score) > 0 else float('inf'),
		"subset_txt_dss_score_sum": torch.stack(all_subset_txt_dss_score).sum().item() if len(all_subset_txt_dss_score) > 0 else float('inf'),
	}

	# wait and aggregate on all processes equally
	all_metrics = write_metrics_to_file(metrics, pth_metrics, accelerator.process_index, accelerator.num_processes)
	aggregated_metrics = aggregate_metrics(all_metrics)

	if accelerator.is_main_process:

		print(f"idx [{accelerator.process_index}] aggregated metrics: {aggregated_metrics}")
		
		compute_fid_scores("diag", f"3d-front-train-instr-scenes-{room_type}-diag", f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/3d-front-train-instr-scenes-{room_type}/diag", f"{pth_viz_output}/diag", args.do_renderings, aggregated_metrics, dataset_res=1024)
		compute_fid_scores("top", f"3d-front-train-instr-scenes-{room_type}-top", f"{os.getenv('PTH_EVAL_VIZ_CACHE')})/3d-front-train-instr-scenes-{room_type}/top", f"{pth_viz_output}/top", args.do_renderings, aggregated_metrics, dataset_res=1024)

		print("")
		print(f"==== eval for [ {dataset_split} ] dataset ({aggregated_metrics['num_scenes']} samples) ====")
		print("")
		print_and_log_metric(dataset_split, epoch, "num_valid_instrs_by_pbl_ratio", aggregated_metrics["num_valid_instrs_by_pbl_ratio"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "num_valid_instrs_by_json_ratio", aggregated_metrics["num_valid_instrs_by_json_ratio"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "filter_pass_rate", aggregated_metrics["filter_pass_rate"], args.use_wandb)
		print("")
		print_and_log_metric(dataset_split, epoch, "scene_total_oob_loss", aggregated_metrics["scene_total_oob_loss"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "scene_total_mbl_loss", aggregated_metrics["scene_total_mbl_loss"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "scene_total_pbl_loss", aggregated_metrics["scene_total_pbl_loss"], args.use_wandb)
		print("")
		print_and_log_metric(dataset_split, epoch, "scene_delta_oob_loss", aggregated_metrics["scene_delta_oob_loss"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "scene_delta_mbl_loss", aggregated_metrics["scene_delta_mbl_loss"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "scene_delta_pbl_loss", aggregated_metrics["scene_delta_pbl_loss"], args.use_wandb)
		print("")
		print_and_log_metric(dataset_split, epoch, "scene_size_l2_dist", aggregated_metrics["scene_size_l2_dist"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "scene_size_m3_vol", aggregated_metrics["scene_size_m3_vol"], args.use_wandb)
		print("")
		print_and_log_metric(dataset_split, epoch, "txt_pms_score", aggregated_metrics["txt_pms_score"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "txt_pms_sampled_score", aggregated_metrics["txt_pms_sampled_score"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "txt_dss_score", aggregated_metrics["txt_dss_score"], args.use_wandb)
		print("")
		print_and_log_metric(dataset_split, epoch, "fid_score_diag", aggregated_metrics["fid_score_diag"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "fid_clip_score_diag", aggregated_metrics["fid_clip_score_diag"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "kid_score_diag", aggregated_metrics["kid_score_diag"], args.use_wandb)
		print("")
		print_and_log_metric(dataset_split, epoch, "fid_score_top", aggregated_metrics["fid_score_top"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "fid_clip_score_top", aggregated_metrics["fid_clip_score_top"], args.use_wandb)
		print_and_log_metric(dataset_split, epoch, "kid_score_top", aggregated_metrics["kid_score_top"], args.use_wandb)
		print("")
		print("==== ============================================================== ====")

		# write final metrics to file
		pth_final_metrics = f"./ckpts/{args.jid}/metrics_{dataset_split}_final.json"
		initialize_file(pth_final_metrics, accelerator.process_index)
		write_dict_to_file(pth_final_metrics, aggregated_metrics)

	print(f"idx [{accelerator.process_index}] finished with run_test() for now!")
	accelerator.wait_for_everyone()
	
	return aggregated_metrics

def compute_dataset_statistics(dataset, all_prompts, all_test_instrs, tokenizer, max_seq_length, all_assets_metadata_simple_descs, sampling_engine, split, n_max, room_type):

	n_best_of_n_assets = 1
	
	num_valid_instrs_by_pbl = torch.zeros(1)

	all_scene_total_oob_loss = []
	all_scene_total_mbl_loss = []
	all_scene_total_pbl_loss = []

	all_scene_delta_oob_loss = []
	all_scene_delta_mbl_loss = []
	all_scene_delta_pbl_loss = []

	all_txt_pms_score = []
	all_txt_dss_score = []

	pth_viz_output = Path(f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/dataset-{split}-sanity-n-{str(n_max)}")
	remove_and_recreate_folder(pth_viz_output)
	
	for idx, sample in tqdm(enumerate(dataset)):		
		if split != "test":
			_, _, _, instr_sample = process_scene_sample(sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False, do_augm=False, do_full_sg_outputs=False, do_keep_jids=True)
		else:
			instr_sample = all_test_instrs.get(sample.get("pth_orig_file"))[1234]

		scene_before_gt_with_assets = json.loads(instr_sample.get("sg_input"))
		scene_before_gt_with_assets = sampling_engine.sample_all_assets(scene_before_gt_with_assets, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False))

		sg_output_add = instr_sample.get("sg_output_add")

		scene_after_gt_with_assets = json.loads(sg_output_add)
		scene_after_gt_with_assets["prompt"] = instr_sample.get("prompt")

		best_result = run_test_for_addition(scene_after_gt_with_assets, scene_before_gt_with_assets, n_best_of_n_assets=1, sampling_engine=sampling_engine)
		
		# since we compare GT with GT
		txt_dss_score = 1.0

		if best_result.get("is_valid_scene_pbl"):
			num_valid_instrs_by_pbl += 1

		all_scene_total_oob_loss.append(torch.tensor(best_result.get("total_oob_loss")))
		all_scene_total_mbl_loss.append(torch.tensor(best_result.get("total_mbl_loss")))
		all_scene_total_pbl_loss.append(torch.tensor(best_result.get("total_pbl_loss")))

		all_scene_delta_oob_loss.append(torch.tensor(best_result.get("delta_oob_loss")))
		all_scene_delta_mbl_loss.append(torch.tensor(best_result.get("delta_mbl_loss")))
		all_scene_delta_pbl_loss.append(torch.tensor(best_result.get("delta_pbl_loss")))
		
		all_txt_pms_score.append(torch.tensor(best_result.get("txt_pms_score")))
		all_txt_dss_score.append(torch.tensor(txt_dss_score))
	
	metrics = {
		"num_scenes": len(dataset),
		"num_valid_instrs_by_pbl": num_valid_instrs_by_pbl.item(),
		"num_scenes_loss_metrics": len(all_scene_total_oob_loss),

		"total_obb_loss": torch.stack(all_scene_total_oob_loss).mean().item() if len(all_scene_total_oob_loss) > 1 else float('inf'),
		"total_mbl_loss": torch.stack(all_scene_total_mbl_loss).mean().item() if len(all_scene_total_mbl_loss) > 1 else float('inf'),
		"total_pbl_loss": torch.stack(all_scene_total_pbl_loss).mean().item() if len(all_scene_total_pbl_loss) > 1 else float('inf'),

		"delta_obb_loss": torch.stack(all_scene_delta_oob_loss).mean().item() if len(all_scene_delta_oob_loss) > 1 else float('inf'),
		"delta_mbl_loss": torch.stack(all_scene_delta_mbl_loss).mean().item() if len(all_scene_delta_mbl_loss) > 1 else float('inf'),
		"delta_pbl_loss": torch.stack(all_scene_delta_pbl_loss).mean().item() if len(all_scene_delta_pbl_loss) > 1 else float('inf'),
		
		"txt_pms_score": torch.stack(all_txt_pms_score).mean().item() if len(all_txt_pms_score) > 1 else float('inf'),
		"txt_dss_score": torch.stack(all_txt_dss_score).mean().item() if len(all_txt_dss_score) > 1 else float('inf'),
	}

	metrics["num_valid_instrs_by_pbl_ratio"] = metrics["num_valid_instrs_by_pbl"] / metrics["num_scenes"]

	compute_fid_scores("diag", f"3d-front-train-instr-scenes-{room_type}-diag", f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/3d-front-train-instr-scenes-{room_type}/diag", str(pth_viz_output / "diag"), True, metrics, dataset_res=1024)
	compute_fid_scores("top", f"3d-front-train-instr-scenes-{room_type}-top", f"{os.getenv('PTH_EVAL_VIZ_CACHE')})/3d-front-train-instr-scenes-{room_type}/top", str(pth_viz_output / "top"), True, metrics, dataset_res=1024)
	
	return metrics

def compute_multi_seed_statistics(room_type, seeds=[1234, 5678, 9012]):

	set_seeds(1234)
	all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
	dataset_train, dataset_val, dataset_test = load_train_val_test_datasets(room_type=room_type, use_cached_dataset=False, seed=1234)
	all_test_instrs = get_test_instrs_all(room_type)
	all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
	
	sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=1234, accelerator=None, do_print=False, is_sft_training=False)
	
	model, tokenizer, max_seq_length = get_model("meta-llama/Llama-3.2-1B-Instruct", use_gpu=False, accelerator=None)	

	all_stats = []
	for split, dataset in [("train", dataset_train), ("val", dataset_val), ("test", dataset_test)]:
		
		for seed in seeds:
			print(f"\ncomputing {split} statistics for seed {seed}...")
			
			set_seeds(seed)
			n_max = 500
			dataset_subset = dataset.select(range(min(len(dataset), n_max)))
		
			stats = compute_dataset_statistics(dataset_subset, all_prompts, all_test_instrs, tokenizer, max_seq_length, all_assets_metadata_simple_descs, sampling_engine, split, n_max, room_type)
			all_stats.append(stats)

	pprint.pprint(pprint.pprint(all_stats))

	# save stats to json
	with open(f"all_dataset_stats_{room_type}.json", 'w') as f:
		json.dump(all_stats, f, indent=4)
	
	# we have 3 different stats dictionaries in our all_stats list
	# report and print mean and std deviation for each value/key pair over all 3 stats dictionaries
	# print("all_stats:", all_stats

def compute_pms_statistics(room_type, seeds):
	
	# itearte over all seeds and compute pms only for each sample in the test set
	dataset_train, dataset_val, dataset_test = load_train_val_test_datasets(room_type=room_type, use_cached_dataset=True, seed=1234)
	all_test_instrs = get_test_instrs_all(room_type)
	all_pms_scores = []
	
	for seed in seeds:

		print(f"\ncomputing PMS statistics for seed {seed}...")
		set_seeds(seed)
		all_pms_scores_seed = []

		for sample in tqdm(dataset_test):
			instr_sample = all_test_instrs.get(sample.get("pth_orig_file"))[seed]
			prompt = instr_sample.get("prompt")
			new_obj_desc = json.loads(instr_sample.get("sg_output_add")).get("desc")
			txt_pms_score = compute_pms_score(prompt, new_obj_desc)
			all_pms_scores_seed.append(txt_pms_score)
		
		all_pms_scores_seed_mean = np.mean(all_pms_scores_seed)
		all_pms_scores.append(all_pms_scores_seed_mean)

	# print mean and std deviation of pms scores over all seeds
	mean_pms = np.mean(all_pms_scores)
	std_pms = np.std(all_pms_scores)
	print(f"\nPMS scores over seeds {seeds}: mean={mean_pms:.4f}, std={std_pms:.4f}")

def compute_gt_seq_accuracy_from_cache(
	room_type: str = "all",
	seeds: list = None,
	n_per_bin: int = 20,
	max_seq_length: int = 10,
	p_add: float = 0.8,
	pth_cache_dir: str = None,
	save_dir: str = "./plots",
) -> dict:
	"""
	Compute the ground-truth upper-bound sequence accuracy for the sequential
	editing dataset and save it as:
	    {save_dir}/seq_eval_raw_{room_type}_gt.json

	Logic
	-----
	* Remove steps  : always succeed (selection is guaranteed unique by construction).
	* Add    steps  : GT object is placed into the replayed scene, then
	                  eval_scene_before_after_with_delta + is_high_quality_sample
	                  are called exactly as in evaluate_seq_step_add.

	Output JSON schema (identical to seq_eval_raw_{room_type}_bon_{n}.json):
	    List[          # one entry per seed
	        List[      # one dict per sample
	            {acc_seq, seq_length, room_type, n_add_total,
	             n_add_passed, n_remove_total, pth_orig_file}
	        ]
	    ]
	"""
	if seeds is None:
		seeds = [1234, 3456, 5678]
	if pth_cache_dir is None:
		pth_cache_dir = os.getenv("PTH_DATASET_CACHE", ".")

	pth_cache = os.path.join(
		pth_cache_dir,
		f"seq_editing_dataset_{room_type}"
		f"_nperbin{n_per_bin}_maxseq{max_seq_length}_padd{int(p_add * 100)}.pkl",
	)

	if not os.path.exists(pth_cache):
		raise FileNotFoundError(
			f"Cached sequential dataset not found at {pth_cache}. "
			"Run create_sequential_editing_dataset() first."
		)

	print(f"[gt_seq_accuracy] Loading dataset from {pth_cache}")
	with open(pth_cache, "rb") as f:
		dataset = pickle.load(f)

	os.makedirs(save_dir, exist_ok=True)
	out_path = os.path.join(save_dir, f"seq_eval_raw_{room_type}_gt.json")

	all_seq_metrics = []  # outer = seeds, inner = samples

	for seed, samples in sorted(dataset.items()):
		seed_metrics = []

		for sample in tqdm(samples, desc=f"seed={seed} GT eval", leave=False):
			steps         = sample["steps"]
			L             = sample["seq_length"]
			partial_scene = sample["partial_scene"]   # room_type, bounds_*, objects

			n_add_total    = 0
			n_add_passed   = 0
			n_remove_total = 0
			step_results   = []

			# Replay current scene state so we know exactly what scene_before
			# looks like at every step (mirroring generate_and_eval_full_sequence).
			current_objects = copy.deepcopy(partial_scene.get("objects", []))

			def _make_scene(objects):
				"""Assemble a scene dict from the partial_scene skeleton + live objects."""
				scene = copy.deepcopy(partial_scene)
				scene["objects"] = copy.deepcopy(objects)
				return scene

			for step in steps:
				op = step["op"]

				if op == "remove":
					n_remove_total += 1
					# Locate and remove the GT object from current_objects.
					# The remove step stores remove_obj_gt = {desc, jid}.
					remove_desc = (step.get("remove_obj_gt") or {}).get("desc")
					if remove_desc:
						for i, obj in enumerate(current_objects):
							if obj.get("desc") == remove_desc:
								current_objects.pop(i)
								break
					step_results.append(True)

				elif op == "add":
					n_add_total += 1
					gt_obj = step.get("add_obj_gt")

					if gt_obj is None:
						step_results.append(False)
						continue

					# Build scene_before and result_scene exactly as
					# evaluate_seq_step_add does for the live model.
					scene_before = _make_scene(current_objects)
					result_scene = _make_scene(current_objects + [copy.deepcopy(gt_obj)])

					# Inject prompt so eval_scene computes PMS against the right text.
					result_scene["objects"][-1]["prompt"] = step["add_prompt"]
					result_scene["objects"][-1]["sampled_asset_desc"] = gt_obj["desc"]

					# pdb.set_trace()

					result    = eval_scene_before_after_with_delta(scene_before, result_scene, is_debug=False, do_pms_full_scene=False)
					pms_score = result.get("txt_pms_sampled_score", 0.0)
					delta_pbl = result.get("delta_pbl_loss", float("inf"))

					#new_obj_size = result_scene["objects"][-1].get("size")
					#gt_obj_size  = gt_obj.get("size")
					size_l2      = 0.0

					new_obj_desc = result_scene["objects"][-1].get("desc", "")

					passed = bool(is_high_quality_sample(
						delta_pbl_loss=delta_pbl,
						txt_pms_sampled_score=pms_score,
						size_l2_dist=size_l2,
						txt_dss_score=0.0,
						desc=new_obj_desc,
					))

					n_add_passed += int(passed)
					step_results.append(passed)

					# Advance replay state (GT object is now in the scene).
					current_objects.append(copy.deepcopy(gt_obj))

				else:
					step_results.append(False)

			# Sequence accuracy: 1 iff every step passed
			acc_seq = sum(step_results) / L if L > 0 else 0.0

			seed_metrics.append({
				"acc_seq":        acc_seq,
				"seq_length":     L,
				"room_type":      sample.get("room_type", room_type),
				"n_add_total":    n_add_total,
				"n_add_passed":   n_add_passed,
				"n_remove_total": n_remove_total,
				"pth_orig_file":  sample.get("pth_orig_file", ""),
			})

		all_seq_metrics.append(seed_metrics)

	# -- Save -----------------------------------------------------------------
	with open(out_path, "w") as f:
		json.dump(all_seq_metrics, f, indent=2)
	print(f"[gt_seq_accuracy] Saved → {out_path}")

	# -- Summary --------------------------------------------------------------
	by_length        = defaultdict(list)
	n_add_total_all  = 0
	n_add_passed_all = 0
	n_remove_all     = 0

	for seed_metrics in all_seq_metrics:
		for m in seed_metrics:
			by_length[m["seq_length"]].append(m["acc_seq"])
			n_add_total_all  += m["n_add_total"]
			n_add_passed_all += m["n_add_passed"]
			n_remove_all     += m["n_remove_total"]

	print("\n[GT upper-bound accuracy]")
	print(f"  Add    steps : {n_add_passed_all}/{n_add_total_all} passed "
	      f"({n_add_passed_all / max(n_add_total_all, 1) * 100:.1f}%)")
	print(f"  Remove steps : {n_remove_all}/{n_remove_all} passed (100.0%)")
	print()
	print(f"  {'L':>3}  {'mean acc_seq':>12}  {'n_samples':>9}")
	for L in sorted(by_length):
		vals = by_length[L]
		print(f"  {L:>3}  {sum(vals)/len(vals):>12.3f}  {len(vals):>9}")
	overall = [v for vals in by_length.values() for v in vals]
	print(f"  {'ALL':>3}  {sum(overall)/len(overall):>12.3f}  {len(overall):>9}")
	print()

	return all_seq_metrics

if __name__ == "__main__":
	set_seeds(1234)
	
	load_dotenv(".env.stanley")
	# load_dotenv(".env.local")

	# compute_multi_seed_statistics("bedroom", seeds=[1234, 3456, 5678])
	# compute_multi_seed_statistics("livingroom", seeds=[1234, 3456, 5678])
	# compute_multi_seed_statistics("all", seeds=[1234, 3456, 5678])

	# compute_pms_statistics("all", seeds=[1234, 3456, 5678])

	compute_gt_seq_accuracy_from_cache(room_type="all", seeds=[1234, 3456, 5678], n_per_bin=20, max_seq_length=10, p_add=0.8)

	# read stats and print
	# with open("all_dataset_stats_livingroom.json", 'r') as f:
	# 	all_stats = json.load(f)
	# train_stats = all_stats[:3]
	# val_stats = all_stats[3:6]
	# test_stats = all_stats[6:]
	# for stats in [("train", train_stats), ("val", val_stats), ("test", test_stats)]:
	# 	print(f"\n=== {stats[0]} ===")
	# 	for key in stats[1][0].keys():
	# 		values = [s[key] for s in stats[1]]
	# 		mean = np.mean(values)
	# 		std = np.std(values)
	# 		print(f"{key}: {mean:.4f} (std: {std:.4f})")