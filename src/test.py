from datetime import datetime
import time
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

from src.eval import eval_scene_before_after_with_delta, compute_dss_score, compute_size_l2_dist
from src.dataset import build_full_instruction_from_prompt, load_train_val_test_datasets, create_full_scene_from_before_and_added, create_instruction_from_scene, clean_copy_of_objects, process_scene_sample
from src.sample import AssetRetrievalModule
from src.utils import set_seeds, remove_and_recreate_folder, safe_parse_scene, inherit_props_by_id, compute_fid_scores, get_system_prompt_sgllm, get_test_instrs_all, get_model
from src.viz import render_full_scene_and_export_with_gif, render_instr_scene_and_export_with_gif

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

# def get_sample_outputs_batch(batch_instrs, model, tokenizer, max_seq_length, accelerator, n_best_of_n_llm, return_logits=False, temp=None):
def get_sample_outputs_batch(batch_instrs, model, tokenizer, max_seq_length, accelerator, n_best_of_n_llm, return_logits=False, temp=None, vllm_engine=None):

	# this buddy will take care of the left padding when batching
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

	all_input_ids = torch.cat(all_input_ids, dim=0)
	all_attention_masks = torch.cat(all_attention_masks, dim=0)

	all_input_ids = all_input_ids.to(accelerator.device)
	all_attention_masks = all_attention_masks.to(accelerator.device)

	is_greedy_sampling = False

	# max_tokens = max(max_seq_length - all_input_ids.shape[-1], 1)
	# max_tokens = 150
	max_tokens = min(max_seq_length - all_input_ids.shape[-1], 150)
	
	# temp = temp if temp is not None else (0.7 if not is_greedy_sampling else None)
	temp = temp if temp is not None else (0.9 if n_best_of_n_llm > 1 else (0.7 if not is_greedy_sampling else None))

	start_time = time.time()
	readable_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
	print(f"\nnucleus sampling... (started: {readable_time}).. batch_input_ids.shape: {batch_input_ids.shape} max_new_tokens: {max_tokens}")

	if vllm_engine is not None and vllm_engine.initialized:
		print("Using vLLM for generation")
		torch.use_deterministic_algorithms(False)
		
		all_responses = vllm_engine.generate(
			all_input_ids, 
			all_attention_masks,
			max_new_tokens=max_tokens,
			temperature=temp,
			top_p=0.95,
			top_k=50,
			do_sample=not is_greedy_sampling
		)
		
		torch.use_deterministic_algorithms(True)
	else:
		# Fallback to regular generation
		print("Using regular model.generate")
		gen_kwargs = {
			"max_new_tokens": max_tokens,
			"pad_token_id": tokenizer.pad_token_id,
			"attention_mask": all_attention_masks,
			"do_sample": (False if is_greedy_sampling else True),
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
		else:
			all_responses = tokenizer.batch_decode(outputs[:, all_input_ids.shape[-1]:], skip_special_tokens=True)
			
		torch.use_deterministic_algorithms(True)

	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"finished sampling after {elapsed_time:.4f} s => {elapsed_time / len(all_responses)} s / sample")

	if return_logits:
		all_logits = torch.stack(outputs.scores, dim=1)
	
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
	else:
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

def run_test_for_addition(scene_after, scene_before_with_assets, n_best_of_n_assets, sampling_engine=None):
	best_result = init_best_result()

	if scene_after.get("objects") is None:
		scene_after = create_full_scene_from_before_and_added(scene_before_with_assets, scene_after)

	for _ in range(n_best_of_n_assets):

		# sample only last object for sample_scene and inherit other samples from scene_before
		if sampling_engine is not None:
			scene_after_with_assets = sampling_engine.sample_last_asset(scene_after, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False))
			inherit_props_by_id(scene_before_with_assets, scene_after_with_assets)
		else:
			scene_after_with_assets = scene_after

		result = eval_scene_before_after_with_delta(scene_before_with_assets, scene_after_with_assets, is_debug=False)

		if result['delta_pbl_loss'] < best_result['delta_pbl_loss']:
			best_result = result
		
		return best_result

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

def run_bon_test_for_addition(prompt, responses, scene_before_with_assets, n_best_of_n_assets, sampling_engine, pth_viz_output=None, filename=None, do_rendering_with_object_count=False, do_renderings=False):
	best_result = init_best_result()
	responses = responses if isinstance(responses, list) else [responses]
	
	# Collect all valid results
	all_results = []

	for idx, response in enumerate(responses):
		scene_after = safe_parse_scene(response)

		if scene_after:
			try:
				# add prompt to last object
				if scene_after.get("objects") is None:
					scene_after["prompt"] = prompt
				else:
					scene_after["objects"][-1]["prompt"] = prompt

				print("run test for addition")
				result = run_test_for_addition(scene_after, scene_before_with_assets, n_best_of_n_assets, sampling_engine)
				# print(result["txt_pms_sampled_score"], result["scene"]["objects"][-1]["prompt"], result["scene"]["objects"][-1]["sampled_asset_desc"])

				# Store this valid result
				all_results.append((idx, result))

				# Render intermediate results if path provided
				# For debugging full pipeline
				if pth_viz_output and do_rendering_with_object_count:
					n_objs = len(scene_before_with_assets.get("objects", [])) + 1
					process_and_render_result(result, pth_viz_output / f"samples-n-{n_objs}", filename=f"n{n_objs}-idx{idx}-dpbl{str(round(result.get('delta_pbl_loss'), 4))}", do_renderings=do_renderings, show_bboxes_also=True)
					# save scene as json too
					pth_json = pth_viz_output / f"samples-n-{n_objs}" / f"n{n_objs}-idx{idx}-dpbl{str(round(result.get('delta_pbl_loss'), 4))}.json"
					with(pth_json).open("w") as f:
						json.dump(result.get("scene"), f, indent=4)

			except Exception as exc:
				traceback.print_exc()
				print("some error during run_bon_test_for_addition. skipping current sample")
	
	# now, given all results, pick best one
	# select the ones with best pms score, then the one with lowest pbl_loss
	if all_results:
		max_pms_score = max(r[1].get('txt_pms_sampled_score', 0) for r in all_results)

		# for each result, print sampled_desc and desc
		# for r in all_results:
		# 	scene = r[1].get("scene")
		# 	print(f"sampled_asset_desc: {scene['objects'][-1]['sampled_asset_desc']}")
		# 	print(f"desc: {scene['objects'][-1]['desc']}")
		# 	print("")
		
		# Filter to only keep results with the maximum pms score
		best_pms_results = [r for r in all_results if r[1].get('txt_pms_sampled_score', 0) == max_pms_score]
		
		# Among those, find the one with minimum delta_pbl_loss
		_, best_result = min(best_pms_results, key=lambda x: x[1].get('delta_pbl_loss', float('inf')))
	
	if pth_viz_output and (filename != None):
		process_and_render_result(best_result, pth_viz_output, filename=filename, do_renderings=do_renderings)
	
	return best_result

def run_instr(prompt, scene_before_with_assets, batch_full_instrs, model, tokenizer, max_seq_length, accelerator, best_of_n, n_best_of_n_assets, sampling_engine, pth_viz_output, do_rendering_with_object_count, temp=None, vllm_engine=None):
	
	# pth_export = Path(pth_output) / "best-of-n" / str(idx)

	# get best-of-n responses for each sample
	responses = get_sample_outputs_batch(batch_full_instrs, model, tokenizer, max_seq_length, accelerator, best_of_n, return_logits=False, temp=temp, vllm_engine=vllm_engine)

	best_result = run_bon_test_for_addition(prompt, responses[0], scene_before_with_assets, n_best_of_n_assets, sampling_engine, pth_viz_output=pth_viz_output, do_rendering_with_object_count=do_rendering_with_object_count, do_renderings=True)
	
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
	

if __name__ == "__main__":
	set_seeds(1234)
	
	# load_dotenv(".env.stanley")
	load_dotenv(".env.local")

	# compute_multi_seed_statistics("bedroom", seeds=[1234, 3456, 5678])
	# compute_multi_seed_statistics("livingroom", seeds=[1234, 3456, 5678])
	# compute_multi_seed_statistics("all", seeds=[1234, 3456, 5678])

	# read stats and print
	with open("all_dataset_stats_livingroom.json", 'r') as f:
		all_stats = json.load(f)
	train_stats = all_stats[:3]
	val_stats = all_stats[3:6]
	test_stats = all_stats[6:]
	for stats in [("train", train_stats), ("val", val_stats), ("test", test_stats)]:
		print(f"\n=== {stats[0]} ===")
		for key in stats[1][0].keys():
			values = [s[key] for s in stats[1]]
			mean = np.mean(values)
			std = np.std(values)
			print(f"{key}: {mean:.4f} (std: {std:.4f})")