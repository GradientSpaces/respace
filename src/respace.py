import json
import re
import copy
import gc
import os
import math
import shutil
import time
import random
import pickle
import traceback

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib import pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from dotenv import load_dotenv
from vllm import SamplingParams

from src.utils import get_model, get_llama_vanilla_pipeline, create_floor_plan_polygon, create_category_lookup, create_vllm_engine, destroy_vllm_engine, safe_parse_scene
from src.sample import AssetRetrievalModule
from src.dataset import build_full_instruction_from_prompt, format_with_chat_template, sample_prompt, load_train_val_test_datasets, rotate_scenegraph, rotate_obj, FloorObjectSampler
from src.test import run_instr, get_sample_outputs_batch, run_bon_test_for_addition
from src.viz import render_full_scene_and_export_with_gif, create_360_video_full
from src.eval import eval_scene, evaluate_seq_step_add, evaluate_seq_step_remove, build_eval_cache_room

class ReSpace:
	def __init__(self, model_id="gradient-spaces/respace-sg-llm-1.5b", env_file=".env", dataset_room_type="all", use_gpu=True, accelerator=None, n_bon_sgllm=1, n_bon_assets=1, do_prop_sampling_for_prompt=True, do_icl_for_prompt=True, do_class_labels_for_prompt=True, use_vllm=True, do_removal_only=False, k_few_shot_samples=2, save_prompts_to=None, load_prompts_from=None, do_bon_shuffling=True, bon_shuffling=8, do_sort_add_asc=False, do_sort_add_desc=False, num_workers=1, do_bon_rotation=True, do_debug_rotation=False):

		load_dotenv(env_file)

		self.save_prompts_to = save_prompts_to
		self.load_prompts_from = load_prompts_from
		self.saved_prompts = {}

		if self.load_prompts_from and os.path.exists(self.load_prompts_from):
			with open(self.load_prompts_from, 'r') as f:
				self.saved_prompts = json.load(f)
			print(f"Loaded prompts from {self.load_prompts_from}")
		
		# prepare models
		self.model, self.tokenizer, self.max_seq_length = get_model(model_id, use_gpu, accelerator, do_not_load_hf_model=(use_vllm == True or do_removal_only == True))
		self.use_vllm = use_vllm
	
		# load SG-LLM via vLLM
		self.vllm_engine = None
		if use_vllm and not do_removal_only:
			try:
				self.vllm_engine = create_vllm_engine(model_id, self.max_seq_length, gpu_memory_utilization=0.23)
				print("SG-LLM: vLLM initialized successfully")
			except Exception as e:
				print(f"Failed to initialize vLLM: {e}. Falling back to regular generation.")
				self.use_vllm = False

		# load zero-shot LLM
		self.vanilla_vllm_engine = None
		# self.vanilla_pipeline = None
		_, self.vanilla_tokenizer, _ = get_model("meta-llama/Meta-Llama-3.1-8B-Instruct", use_gpu, accelerator=None, do_not_load_hf_model=True)
		self.vanilla_pipeline = get_llama_vanilla_pipeline()

		# sampling engine
		if not do_removal_only:
			self.sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, accelerator=accelerator, do_print=False)

		# floor stats sampler
		self.all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		
		dataset_train, _, _ = load_train_val_test_datasets(room_type=dataset_room_type, use_cached_dataset=True, do_sanity_check=False, accelerator=accelerator)
		self.dataset_train = dataset_train
		self.dataset_room_type = dataset_room_type
		
		self.accelerator = accelerator if accelerator is not None else Accelerator()
		self.n_bon_sgllm = n_bon_sgllm
		self.n_bon_assets = n_bon_assets
		self.use_gpu = use_gpu

		self.do_prop_sampling_for_prompt = do_prop_sampling_for_prompt
		self.do_icl_for_prompt = do_icl_for_prompt
		self.do_class_labels_for_prompt = do_class_labels_for_prompt
		self.k_few_shot_samples = k_few_shot_samples
		self.dataset_stats_for_prompt = None

		self.do_bon_shuffling = do_bon_shuffling
		self.bon_shuffling = bon_shuffling
		
		self.do_sort_add_asc = do_sort_add_asc
		self.do_sort_add_desc = do_sort_add_desc

		self.num_workers = num_workers
		self.do_bon_rotation = do_bon_rotation
		self.do_debug_rotation = do_debug_rotation
		print("using num_workers=", num_workers)

		self.max_n_attempts = 10
		self.pth_dataset_stats_cache = {}

	def _prepare_dataset_stats_for_object_sampler(self, gen_room_type=None):
		if gen_room_type == None:
			room_type_filter = "nofilter"
		else:
			room_type_filter = gen_room_type

		pth_dataset_stats = os.path.join(os.getenv("PTH_DATASET_CACHE"), f"merged_dataset_stats_{self.dataset_room_type}_{room_type_filter}.pkl")

		if pth_dataset_stats in self.pth_dataset_stats_cache:
			print("loading stats from in-memory cache...")
			all_stats = self.pth_dataset_stats_cache[pth_dataset_stats]
		elif os.path.exists(pth_dataset_stats):
			print("loading stats file from disk...")
			all_stats = pickle.load(open(pth_dataset_stats, "rb"))
			self.pth_dataset_stats_cache[pth_dataset_stats] = all_stats
		else:
			print("creating stats file...")
			all_assets_metadata = json.load(open(os.getenv("PTH_ASSETS_METADATA")))
			all_assets_metadata_orig = json.load(open(os.path.join(os.getenv("PTH_3DFUTURE_ASSETS"), "model_info.json")))
			desc_to_category = create_category_lookup(all_assets_metadata_orig, all_assets_metadata)

			all_stats = {
				"floor_area_n_objects": [],
				"unique_object_classes": set(),
			}

			if gen_room_type != None:
				dataset_filtered = self.dataset_train.filter(lambda x: x.get("room_type") == gen_room_type)
			else:
				dataset_filtered = self.dataset_train

			for sample in tqdm(dataset_filtered):
				# get floor area and number of objects
				floor_area = create_floor_plan_polygon(sample.get("scene").get("bounds_bottom")).area
				n_objects = len(sample.get("scene").get("objects"))
				all_stats["floor_area_n_objects"].append({
					"floor_area": floor_area, 
					"n_objects": n_objects,
					"object_prompts": [ sample_prompt(self.all_prompts, obj.get("jid")) for obj in sample.get("scene").get("objects") ]
				})

				# add unique object classes
				for obj in sample.get("scene").get("objects"):
					all_stats["unique_object_classes"].add(desc_to_category.get(obj.get("desc")))

			# remove "unknown_category" from unique object classes if present
			if "unknown_category" in all_stats["unique_object_classes"]:
				all_stats["unique_object_classes"].remove("unknown_category")

			pickle.dump(all_stats, open(pth_dataset_stats, "wb"))

		return all_stats
	
	def _generate_unique_shuffles(self, commands, k):
		if k <= 1:
			return [commands.copy()]
		
		unique_shuffles = [commands.copy()]  # Original order is first
		seen = {tuple(commands)}
		
		max_attempts = k * 10  # Prevent infinite loop
		attempts = 0
		
		# First, try to find unique shuffles
		while len(unique_shuffles) < k and attempts < max_attempts:
			shuffled = commands.copy()
			random.shuffle(shuffled)
			shuffled_tuple = tuple(shuffled)
			
			if shuffled_tuple not in seen:
				seen.add(shuffled_tuple)
				unique_shuffles.append(shuffled)
			attempts += 1
		
		# If we couldn't find enough unique shuffles, fill with duplicates
		while len(unique_shuffles) < k:
			duplicate = random.choice(unique_shuffles).copy()
			unique_shuffles.append(duplicate)
		
		return unique_shuffles
	
	def _has_token_budget(self, instr_str, min_output_tokens=150):
		"""Check if the instruction leaves enough room for output tokens within max_seq_length."""
		formatted = format_with_chat_template(self.tokenizer, instr_str)
		n_input_tokens = len(self.tokenizer.encode(formatted))
		has_budget = n_input_tokens + min_output_tokens <= self.max_seq_length
		if not has_budget:
			print(f"  Token budget exhausted: {n_input_tokens} input + {min_output_tokens} output > {self.max_seq_length} max")
		return has_budget

	def _compute_scene_metrics(self, scene, eval_cache=None):
		"""Evaluate a full scene and return total_pbl_loss + txt_pms_sampled_score."""
		result = eval_scene(scene, is_debug=False, eval_cache=eval_cache)
		return {
			"total_pbl_loss":      result.get("total_pbl_loss", float('inf')),
			"txt_pms_sampled_score": result.get("txt_pms_sampled_score", 0.0),
		}
	
	def _add_object_with_rotation_augment(self, prompt, current_scene, temp=0.7, do_rendering_with_object_count=False, pth_viz_output=None):
		"""Add object using 4 rotation augmentations (0/90/180/270 deg).

		Batches all 4 rotated prompts into ONE vLLM call, then pools ALL candidates
		(4 rotations × BoN) into a single flat selection pass. This avoids hierarchical
		selection (BoN-first then rotation-first) which compounds PMS bias.

		eval_cache and before_metrics are computed ONCE on the original (unrotated) scene.
		All candidates are inverse-rotated to the original frame before eval.
		"""
		angles = [0, math.pi/2, math.pi, 3*math.pi/2]

		# Phase 1: Create 4 rotated scene copies and prepare prompts
		rotated_scenes = []
		all_instrs = []
		for angle in angles:
			scene_copy = copy.deepcopy(current_scene)
			if angle != 0:
				rotate_scenegraph(scene_copy, angle)
			rotated_scenes.append(scene_copy)
			instr = self._prepare_input_for_addition(prompt, scene_copy)
			all_instrs.extend(instr)

		# Token budget check on longest instruction
		if not self._has_token_budget(max(all_instrs, key=len)):
			print(f"Cannot fit output for '{prompt}' with rotation augment. Returning current scene.")
			return current_scene, True

		# Phase 2: ONE batched vLLM call — 4 prompts × BoN responses
		all_responses = get_sample_outputs_batch(
			all_instrs, self.model, self.tokenizer, self.max_seq_length,
			self.accelerator, self.n_bon_sgllm, return_logits=False,
			temp=temp, vllm_engine=(self.vllm_engine if self.use_vllm else None)
		)

		# Phase 3: Flatten all responses (4 rotations × BoN) into one candidate list.
		# Inverse-rotate each candidate back to original frame for shared eval.
		flat_responses = []
		angle_per_response = []  # track which angle each flattened response came from
		for i, angle in enumerate(angles):
			responses_i = all_responses[i]
			responses_list = responses_i if isinstance(responses_i, list) else [responses_i]
			for resp in responses_list:
				scene_after = safe_parse_scene(resp)
				if scene_after is None:
					flat_responses.append(resp)
				else:
					if angle != 0:
						rotate_obj(scene_after, -angle)
					flat_responses.append(json.dumps(scene_after))
				angle_per_response.append(angle)

		# Phase 4: Single selection pass over ALL candidates (shared room cache)
		eval_cache = build_eval_cache_room(current_scene)
		before_metrics = eval_scene(current_scene, is_debug=False, eval_cache=eval_cache)

		result = run_bon_test_for_addition(
			prompt, flat_responses, current_scene,
			self.n_bon_assets, self.sampling_engine,
			pth_viz_output=pth_viz_output,
			do_rendering_with_object_count=do_rendering_with_object_count,
			num_workers=self.num_workers,
			eval_cache=eval_cache, before_metrics=before_metrics,
		)

		len_before = len(current_scene.get("objects"))
		if result.get("scene") and len(result["scene"].get("objects", [])) == len_before + 1:
			pms = result.get("txt_pms_sampled_score", 0.0)
			dpbl = result.get("delta_pbl_loss", float('inf'))

			result_scene = result["scene"]
			result_scene["objects"][-1]["prompt"] = prompt

			# Determine winning angle for logging
			# The winning object is already in the original frame; find which angle it came from
			# by checking the result index from run_bon_test_for_addition
			winning_angle = 0  # default
			best_idx = result.get("best_idx", None)
			if best_idx is not None and best_idx < len(angle_per_response):
				winning_angle = angle_per_response[best_idx]

			total_pbl = result.get("total_pbl_loss", float('inf'))
			print(f"  Added '{prompt}': PMS={pms:.4f}, PBL={total_pbl:.6f}, angle={int(math.degrees(winning_angle))}deg (BoN={self.n_bon_sgllm} + ROT, N={len(flat_responses)})")

			# # Debug: render all 4 rotations
			# if self.do_debug_rotation:
			# 	n_objs = len(result_scene.get("objects", []))
			# 	pth_debug = Path("./eval/viz/misc/rotation-debug")
			# 	# delete folder
			# 	if pth_debug.exists():
			# 		shutil.rmtree(pth_debug)
			# 	os.makedirs(pth_debug, exist_ok=True)

			# 	for r_idx, angle in enumerate(angles):
			# 		r_angle_deg = int(math.degrees(angle))
			# 		is_best = "_BEST" if angle == winning_angle else ""

			# 		resp_i = all_responses[r_idx]
			# 		raw_resp = resp_i[0] if isinstance(resp_i, list) else resp_i
			# 		raw_obj = safe_parse_scene(raw_resp)
			# 		if raw_obj is None:
			# 			print(f"  [debug-rotation] rot={r_angle_deg}deg: parse failed, skipping render")
			# 			continue

			# 		obj_to_append = raw_obj["objects"][-1] if raw_obj.get("objects") is not None else raw_obj

			# 		# (A) Rotated frame: rotated scene + raw object (what the model saw)
			# 		debug_scene_rotated = copy.deepcopy(rotated_scenes[r_idx])
			# 		debug_scene_rotated["objects"].append(copy.deepcopy(obj_to_append))
			# 		debug_scene_rotated = self.sampling_engine.sample_all_assets(debug_scene_rotated, is_greedy_sampling=True)

			# 		fname_rot = f"n{n_objs}_rot{r_angle_deg}_ROTATED-FRAME{is_best}"
			# 		print(f"  [debug-rotation] rendering {fname_rot}")
			# 		render_full_scene_and_export_with_gif(debug_scene_rotated, fname_rot, pth_debug, create_gif=False)

			# 		# (B) Original frame: original scene + raw object with inverse rotation
			# 		obj_to_append_inverted = copy.deepcopy(obj_to_append)
			# 		rotate_obj(obj_to_append_inverted, -angle)

			# 		debug_scene_original = copy.deepcopy(current_scene)
			# 		debug_scene_original["objects"].append(copy.deepcopy(obj_to_append_inverted))
			# 		debug_scene_original = self.sampling_engine.sample_all_assets(debug_scene_original, is_greedy_sampling=True)

			# 		fname_orig = f"n{n_objs}_rot{r_angle_deg}_ORIGINAL-FRAME-WITH-INVROT{is_best}"
			# 		print(f"  [debug-rotation] rendering {fname_orig}")
			# 		render_full_scene_and_export_with_gif(debug_scene_original, fname_orig, pth_debug, create_gif=False)

			# 	print(f"  [debug-rotation] saved renders to {pth_debug}")
			# 	print("\n=== DEBUG ROTATION: first object rendered, exiting early ===")
			# 	exit(0)

			return result_scene, True

		return current_scene, False

	def _run_shuffled_additions(self, add_commands, scene_after_removals, do_rendering_with_object_count=False, pth_viz_output=None):
		"""Breadth-first shuffled object addition with batched vLLM inference.

		Instead of running each shuffle sequentially (8 shuffles × M steps = 8M vLLM calls),
		this batches all shuffles' prompts at each step into a single vLLM call (M calls total).

		With rotation augmentation enabled, each shuffle gets 4 rotated variants (0/90/180/270),
		so the batch grows to N×4 prompts per step. Best rotation is picked per shuffle per step.
		"""
		shuffled_versions = self._generate_unique_shuffles(add_commands, self.bon_shuffling)
		n_shuffles = len(shuffled_versions)
		n_steps = len(add_commands)
		n_rotations = 4 if self.do_bon_rotation else 1
		angles = [0, math.pi/2, math.pi, 3*math.pi/2] if self.do_bon_rotation else [0]

		print(f"\n=== SHUFFLING MODE (breadth-first): {n_shuffles} orderings, {n_steps} steps, rot_augment={self.do_bon_rotation} ===\n")

		shuffle_scenes = [copy.deepcopy(scene_after_removals) for _ in range(n_shuffles)]
		shuffle_alive = [True] * n_shuffles
		shuffle_budget_exhausted = [False] * n_shuffles

		# Room voxel cache is independent of objects — compute once, share across all shuffles and steps
		room_eval_cache = build_eval_cache_room(scene_after_removals)

		for step_idx in range(n_steps):
			alive_indices = [i for i in range(n_shuffles) if shuffle_alive[i] and not shuffle_budget_exhausted[i]]
			if not alive_indices:
				if any(shuffle_budget_exhausted):
					print(f"All remaining shuffles have exhausted token budget at step {step_idx + 1}/{n_steps}")
				else:
					print("All shuffles dead, aborting")
				break

			print(f"\n--- Step {step_idx + 1}/{n_steps} ({len(alive_indices)} alive shuffles) ---")

			# Phase 1: Collect prompts for all alive shuffles × rotations, checking token budget
			prompts = []        # one per batch entry (shuffle that made it)
			all_instrs = []     # N_batch × n_rotations instructions
			batch_indices = []  # which shuffle indices made it into the batch

			for shuf_idx in alive_indices:
				command = shuffled_versions[shuf_idx][step_idx]
				prompt_obj = re.search(r'<add>(.*?)</add>', command).group(1).strip().lower()

				# Create rotated scenes and instructions for this shuffle
				rot_instrs = []
				for angle in angles:
					scene_copy = copy.deepcopy(shuffle_scenes[shuf_idx])
					if angle != 0:
						rotate_scenegraph(scene_copy, angle)
					instr = self._prepare_input_for_addition(prompt_obj, current_scene=scene_copy)
					rot_instrs.append(instr[0])

				# Budget check on longest instruction (rotation may change length slightly)
				if not self._has_token_budget(max(rot_instrs, key=len)):
					shuffle_budget_exhausted[shuf_idx] = True
					n_objs = len(shuffle_scenes[shuf_idx].get("objects", []))
					print(f"  Shuffle {shuf_idx + 1}: token budget exhausted at {n_objs} objects, keeping partial scene")
					continue

				prompts.append(prompt_obj)
				all_instrs.extend(rot_instrs)
				batch_indices.append(shuf_idx)

			if not batch_indices:
				print(f"  No shuffles have token budget at step {step_idx + 1}, stopping early")
				break

			# Phase 2: ONE batched vLLM call (all shuffles × rotations)
			start_time = time.time()
			all_responses = get_sample_outputs_batch(
				all_instrs, self.model, self.tokenizer, self.max_seq_length,
				self.accelerator, self.n_bon_sgllm, temp=0.7,
				vllm_engine=(self.vllm_engine if self.use_vllm else None),
			)
			n_batch = len(batch_indices)
			print(f"  Batched vLLM ({len(all_instrs)} prompts = {n_batch} shuffles × {n_rotations} rot, n={self.n_bon_sgllm}) took {time.time() - start_time:.2f}s")

			# Phase 3: Pre-compute before_metrics per shuffle (room cache is shared)
			shuffle_caches = []
			for b_idx in range(n_batch):
				shuf_idx = batch_indices[b_idx]
				bm = eval_scene(shuffle_scenes[shuf_idx], is_debug=False, eval_cache=room_eval_cache)
				shuffle_caches.append(bm)

			# Phase 4: For each shuffle, flatten all rotation×BoN candidates into one
			# selection pass (avoids hierarchical BoN-first then rotation-first bias)
			for b_idx in range(n_batch):
				shuf_idx = batch_indices[b_idx]
				prompt_obj = prompts[b_idx]
				bm = shuffle_caches[b_idx]
				len_before = len(shuffle_scenes[shuf_idx].get("objects", []))

				# Collect and inverse-rotate all candidates for this shuffle
				flat_responses = []
				for r_idx in range(n_rotations):
					resp_idx = b_idx * n_rotations + r_idx
					angle = angles[r_idx]
					responses_i = all_responses[resp_idx]
					responses_list = responses_i if isinstance(responses_i, list) else [responses_i]
					for resp in responses_list:
						scene_after = safe_parse_scene(resp)
						if scene_after is None:
							flat_responses.append(resp)
						else:
							if angle != 0:
								rotate_obj(scene_after, -angle)
							flat_responses.append(json.dumps(scene_after))

				# Single selection pass over all candidates for this shuffle
				result = run_bon_test_for_addition(
					prompt_obj, flat_responses, shuffle_scenes[shuf_idx],
					self.n_bon_assets, self.sampling_engine,
					pth_viz_output=pth_viz_output,
					do_rendering_with_object_count=do_rendering_with_object_count,
					do_renderings=True,
					num_workers=self.num_workers,
					eval_cache=room_eval_cache, before_metrics=bm,
				)

				if result.get("scene") is not None and len(result["scene"].get("objects", [])) == len_before + 1:
					new_obj = result["scene"]["objects"][-1]
					pms = result.get("txt_pms_sampled_score", 0.0)
					dpbl = result.get("delta_pbl_loss", float('inf'))

					shuffle_scenes[shuf_idx] = copy.deepcopy(shuffle_scenes[shuf_idx])
					shuffle_scenes[shuf_idx]["objects"].append(new_obj)
					shuffle_scenes[shuf_idx]["objects"][-1]["prompt"] = prompt_obj

					print(f"    Shuffle {shuf_idx + 1}: PMS={pms:.4f}, dPBL={dpbl:.4f} (N={len(flat_responses)})")
				else:
					if self.n_bon_sgllm == 1:
						print(f"  Shuffle {shuf_idx + 1}: failed, retrying '{prompt_obj}' individually...")
						scene, ok = self.add_object(
							prompt_obj, shuffle_scenes[shuf_idx],
							do_rendering_with_object_count=do_rendering_with_object_count,
							pth_viz_output=pth_viz_output,
							temp=0.7,
						)
						if ok:
							shuffle_scenes[shuf_idx] = scene
							print(f"  Shuffle {shuf_idx + 1}: retry succeeded")
						else:
							shuffle_alive[shuf_idx] = False
							print(f"  Shuffle {shuf_idx + 1}: retry failed, marking dead")
					else:
						shuffle_alive[shuf_idx] = False
						print(f"  Shuffle {shuf_idx + 1}: all BoN candidates failed, marking dead")

			# Free intermediate data from this step (responses, parsed scenes, etc.)
			gc.collect()
			torch.cuda.empty_cache()

		# Final: evaluate complete scenes in parallel, pick best
		# Include both alive and budget-exhausted shuffles (both have valid scenes)
		alive_with_objects = [
			i for i in range(n_shuffles)
			if (shuffle_alive[i] or shuffle_budget_exhausted[i]) and len(shuffle_scenes[i].get("objects", [])) > 0
		]
		for i in range(n_shuffles):
			if (shuffle_alive[i] or shuffle_budget_exhausted[i]) and i not in alive_with_objects:
				print(f"Shuffle {i + 1}: alive but no objects, skipping")

		candidates = []
		if alive_with_objects:
			n_eval_workers = min(len(alive_with_objects), self.num_workers)
			print(f"  Evaluating {len(alive_with_objects)} alive shuffles in parallel (workers={n_eval_workers})...")
			with ThreadPoolExecutor(max_workers=n_eval_workers) as pool:
				future_to_idx = {
					pool.submit(self._compute_scene_metrics, shuffle_scenes[i], room_eval_cache): i
					for i in alive_with_objects
				}
				for future in as_completed(future_to_idx):
					shuf_idx = future_to_idx[future]
					metrics = future.result()
					n_objs = len(shuffle_scenes[shuf_idx].get("objects", []))
					print(f"  Shuffle {shuf_idx + 1}: PMS={metrics['txt_pms_sampled_score']:.4f}, PBL={metrics['total_pbl_loss']:.6f}, n_objects={n_objs}")
					candidates.append((shuf_idx + 1, copy.deepcopy(shuffle_scenes[shuf_idx]), metrics))

		if candidates:
			min_pbl = min(c[2]["total_pbl_loss"] for c in candidates)
			pbl_tied = [c for c in candidates if c[2]["total_pbl_loss"] == min_pbl]
			best_shuffle_idx, best_scene, best_metrics = max(pbl_tied, key=lambda c: c[2]["txt_pms_sampled_score"])

			pool_parts = [f"BoN={self.n_bon_sgllm}", f"SHUFFL={n_shuffles}"]
			if self.do_bon_rotation:
				pool_parts.append("ROT")
			pool_desc = " + ".join(pool_parts)
			n_total = n_shuffles * n_rotations * self.n_bon_sgllm
			summary = f"PMS={best_metrics['txt_pms_sampled_score']:.4f}, PBL={best_metrics['total_pbl_loss']:.6f}, shuffle=#{best_shuffle_idx}"
			print(f"\n=== BEST SCENE: {summary} ({pool_desc}, N={n_total}) ===\n")
			return best_scene, True
		else:
			return None, False

	def _build_full_query_for_zeroshot_model(self, prompt, scenegraph):
		query = f"""<prompt>{prompt}<prompt>\n"""
		if scenegraph is not None:
			query += f"\n<scenegraph>{json.dumps(scenegraph)}</scenegraph>"
		return query
	
	def _get_system_prompt_zeroshot_handle_user_instr(self, few_shot_samples=None):
		full_prompt = f"""you are a world-class leading interior design expert. your task is to fulfill the request of the user about interior design but you have help of another world-class expert model that can only be called in an XML-style API.

# input
- <prompt> : the user request
- <scenegraph> : the current scene will be given as a JSON object. in some cases, there will be no scene graph given, which means there is no "current" scene to work with. the "bounds_top" and "bounds_bottom" keys contain the boundaries as a list of 3D vertices in metric space.

# task
- composing a list of commands to fulfill the user request via <add> and <remove> commands. ideally, you reflect the existing objects in the scenegraph, if one is given.

# adding
- if the user wants to add one or multiple objects, you create an <add> command for every object/furniture and add it to the list in "commands".
- for the description, you should refer to the subject with a maximum of five additional descriptive words. the first words should refer to the color / style / shape / etc., while the last word should always be the main subject. your description must be in 'noun phrase'.
- if the user request provides an existing scene description provided via <scenegraph>...</scenegraph> and there are existing objects in the scene, you should try to match the style of the existing objects by providing a similar style as part of the description of your commands.
- if the user provides some requirement about particular furniture that should be present in the room, you should always add these objects via <add> commands.
- your format should be: <add>description</add>
- DO NEVER use more than 5 words for each description

# removing / swapping
- if the user wants to remove one to multiple objects, you add a <remove> command for every object that should be removed.
- if the user wants to swap or replace furniture, you MUST use <remove> first and then use <add>.
- if there are similar candidates for removal you should remove the object that matches the description best.
- your format should be: <remove>description</remove>
- you can keep the description short here as well

# output
- the commands are given as a list under the "commands" key where each command follows EXACTLY the format specified above and is given as a string, i.e. "<add>...</add>" or "<remove>...</remove>".
- if there are remove commands, you always put them BEFORE add commands. 
- IMPORTANT: you NEVER use the <remove> commands unless the user EXPLICITLY asks for it via swapping or removing objects. you do not make assumptions about this.
- you NEVER remove objects to "match the style" or if there is already an object in the scene similar to the requested one. a scene can contain as many similar objects as the user wants. you ONLY remove objects if the user explicitly asks for removal or swapping.

- if you use the <remove> command, you MUST provide your reasoning under the "reasoning" key, which comes before the "commands" key in the same JSON object.
- you always output the final JSON object as a plain string and nothing else. NEVER use markdown.
"""
		if self.do_class_labels_for_prompt:
			prompt_postfix_1 = f"""\n# available object classes
- you should only pick objects for <add> based on the following high-level abstract classes
- your objects should be more specific than these classes but you should not add objects that are not part of these classes/labels
{self.dataset_stats_for_prompt.get('unique_object_classes')}
"""
			full_prompt += prompt_postfix_1
		
		if self.do_icl_for_prompt and few_shot_samples != None:
			
			full_prompt += """\n# few-shot examples for scenes that have a similar size to the requested one (your scene should be different though and stick to the user prompt):\n"""

			for sample in few_shot_samples:
				full_prompt += f"\n## example\n"
				for obj_prompt in sample:
					full_prompt += f"<add>{obj_prompt}</add>\n"

		if self.do_sort_add_asc:
			full_prompt += "\n IMPORTANT: order your <add> commands by estimated physical size of the objects in ASCENDING order (smallest objects first, largest objects last)."
		elif self.do_sort_add_desc:
			full_prompt += "\n IMPORTANT: order your <add> commands by estimated physical size of the objects in DESCENDING order (largest objects first, smallest objects last)."

		full_prompt += "\nREMINDER: each description in your <add>...</add> commands should be IN NOUN PHRASE WITH 2-3 words AND AT MAXIMUM 5 words"

		return full_prompt
	
	def _sample_random_bounds(self, dataset, room_type=None):
		if room_type != None:
			dataset_filtered = dataset.filter(lambda x: x.get("room_type") == room_type)
		else:
			dataset_filtered = dataset
		idx = np.random.choice(len(dataset_filtered))
		sample = dataset_filtered.select([idx])[0]
		scene = sample.get("scene")
		scene_bounds_only = {
			"room_type": room_type if room_type != None else sample.get("room_type"),
			"bounds_top": scene.get("bounds_top"),
			"bounds_bottom": scene.get("bounds_bottom"),
			"objects": [],
		}
		return scene_bounds_only
	
	def _prepare_input_for_addition(self, prompt, current_scene=None, sample_sg_input=None):
		if current_scene:
			# Remove asset references for forward pass
			cleaned_scene = copy.deepcopy(current_scene)
			cleaned_scene["objects"] = []
			for obj in current_scene.get("objects"):
				cleaned_obj = {k: v for k, v in obj.items() if not k.startswith('sampled_') and k != "uuid" and k != "jid"}
				cleaned_scene["objects"].append(cleaned_obj)
			sg_input = json.dumps(cleaned_scene)
		else:
			sg_input = sample_sg_input

		full_instruction = build_full_instruction_from_prompt(prompt, sg_input)
		batch_full_instrs = [full_instruction]
		return batch_full_instrs
	
	def render_scene_frame(self, scene, filename, pth_viz_output, show_bboxes=False, show_assets=True, create_gif=False, bg_color=None, camera_height=None):
		render_full_scene_and_export_with_gif(scene, filename=filename, pth_output=pth_viz_output, show_bboxes=show_bboxes, show_assets=show_assets, create_gif=False, bg_color=None, camera_height=camera_height)

	def render_scene_360video(self, scene, filename, pth_viz_output=None, resolution=(1536, 1024), video_duration=4.0, step_time=0.5, bg_color=None, camera_height=None):
		create_360_video_full(scene, filename, pth_viz_output, resolution=resolution, camera_height=camera_height, video_duration=video_duration, step_time=step_time, bg_color=bg_color)

	def resample_last_asset(self, scene, is_greedy_sampling=True):
		scene_tmp = scene.copy()
		scene_tmp["objects"][-1] = {k: v for k, v in scene_tmp["objects"][-1].items() if not k.startswith("sampled_")}
		return self.sampling_engine.sample_last_asset(scene_tmp, is_greedy_sampling=is_greedy_sampling)
	
	def resample_all_assets(self, scene, is_greedy_sampling=True):
		scene_tmp = scene.copy()
		for obj in scene_tmp.get("objects"):
			obj = {k: v for k, v in obj.items() if not k.startswith("sampled_")}
		return self.sampling_engine.sample_all_assets(scene_tmp, is_greedy_sampling=is_greedy_sampling)

	def _save_prompts(self, scene_id, prompt, commands, current_scene):
		self.saved_prompts[scene_id] = {
			"commands":      commands,
			"original_prompt": prompt,
			"room_type":     current_scene.get("room_type"),
			"n_objects":     len([c for c in commands if c.startswith("<add>")]),
		}
		os.makedirs(
			os.path.dirname(self.save_prompts_to) if os.path.dirname(self.save_prompts_to) else '.',
			exist_ok=True,
		)
		with open(self.save_prompts_to, 'w') as f:
			json.dump(self.saved_prompts, f, indent=2)
		print(f"Saved {len(commands)} commands for scene {scene_id}")
	
	def add_object(self, prompt, current_scene, do_sample_assets_for_input_scene=False, do_rendering_with_object_count=False, temp=None, do_dynamic_temp=True, pth_viz_output=None):
		print("adding object...")

		if do_sample_assets_for_input_scene:
			current_scene = self.sampling_engine.sample_all_assets(current_scene, is_greedy_sampling=(True if self.n_bon_assets == 1 else False))

		batch_full_instrs = self._prepare_input_for_addition(prompt, current_scene=current_scene)
		len_before = len(current_scene.get("objects"))

		if not self._has_token_budget(batch_full_instrs[0]):
			print(f"  Token budget exhausted for '{prompt}' (n_objs={len_before}). Keeping current scene.")
			return current_scene, True

		if self.do_bon_rotation:
			# Rotation augmentation path: 4 rotations × BoN candidates, flat selection
			scene, success = self._add_object_with_rotation_augment(
				prompt, current_scene, temp=(temp if temp else 0.7),
				do_rendering_with_object_count=do_rendering_with_object_count,
				pth_viz_output=pth_viz_output
			)
			if success:
				return scene, True
			else:
				print("  Rotation augment failed for all angles. Falling back to vanilla retry.")
				# Fall through to vanilla retry below

		# Vanilla retry loop (no rotation, or rotation fallback)
		temp = copy.copy(temp)
		remaining_attempts = copy.copy(self.max_n_attempts)

		while True:
			try:
				if do_dynamic_temp and remaining_attempts < self.max_n_attempts and temp != None:
					temp = max(temp - 0.05, 0.4)
					if temp == 0.4:
						temp = 1.2
				best_result = run_instr(prompt, current_scene, batch_full_instrs, self.model, self.tokenizer, self.max_seq_length, self.accelerator, self.n_bon_sgllm, self.n_bon_assets, self.sampling_engine, pth_viz_output, do_rendering_with_object_count=do_rendering_with_object_count, temp=temp, vllm_engine=(self.vllm_engine if self.use_vllm else None), num_workers=self.num_workers)

				if best_result.get("scene") != None and len(best_result.get("scene").get("objects")) == len_before + 1:
					pms = best_result.get("txt_pms_sampled_score", 0.0)
					total_pbl = best_result.get("total_pbl_loss", float('inf'))
					print(f"  Added '{prompt}': PMS={pms:.4f}, PBL={total_pbl:.6f} (BoN={self.n_bon_sgllm}, N={self.n_bon_sgllm})")
					current_scene = best_result.get("scene")
					current_scene["objects"][-1]["prompt"] = prompt
					return current_scene, True
				else:
					print("  No valid object added. Retrying...")

			except Exception as exc:
				print(exc)
				traceback.print_exc()
				print("  Failed to add object. Retrying...")

			if remaining_attempts > 0:
				remaining_attempts -= 1
				print(f"  Retrying add_object() ... {remaining_attempts} attempts left.")
			else:
				print("  Max attempts reached. Returning current scene without changes.")
				return current_scene, False

			gc.collect()
			torch.cuda.empty_cache()
	
	def remove_object(self, prompt, current_scene, do_rendering_with_object_count=False, do_dynamic_temp=True, pth_viz_output=None, idx=None):
		print("removing object...")

		print(f"<remove>{prompt}<remove>")

		# Build a query for the vanilla pipeline to identify which object to remove
		query = f"""<remove>{prompt}<remove>
<scenegraph>{json.dumps(current_scene)}</scenegraph>"""
		
		system_prompt = """you are a world-class leading interior design expert. your task is to remove furniture given the descriptions in the header and the current list of furniture in the body. you must respond ONLY with a valid JSON string that matches precisely the *format* of the existing JSON in the request.

if there are multiple objects that match the description precisely, you should remove all of them.

the prompt for the object to be removed will be given in the header between <remove>...</remove> tags. the current scene will be given as a JSON object in the body between <scenegraph>...</scenegraph> tags.

in the successful case, your output contains one or N fewer objects in the "objects" list and the rest of the JSON object should be EXACTLY identical to the input.

you can also remove all objects if the prompt matches those objects. in that case, you provide an empty list for the "objects" key.

you can further assume that in most cases, there will be at least one object in the scene that matches the description roughly. this object shall be removed.

only output the JSON (with the removed objects) as a plain string and nothing else."""


		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": query},
		]

		remaining_attempts = copy.copy(self.max_n_attempts)
		temp = 0.7
		while True:
			try:
				if do_dynamic_temp and remaining_attempts < self.max_n_attempts:
					# first, decrease temp to get more deterministic results
					temp = max(temp - 0.05, 0.4)
					# if not successful, increase temp to get more diverse results and see if we escape weird behavior
					if temp == 0.4:
						temp = 1.2

				# Get response from vanilla LLM
				print(f"temp: {temp}")
				if self.vanilla_vllm_engine is not None:
					formatted_prompt = self.vanilla_tokenizer.apply_chat_template(
						messages,
						tokenize=False,
						add_generation_prompt=True,
					)
					vllm_outputs = self.vanilla_vllm_engine.generate(
						[formatted_prompt],
						SamplingParams(max_tokens=16384, temperature=temp, top_p=0.95, top_k=50),
					)
					response = vllm_outputs[0].outputs[0].text
				else:
					start_time = time.time()
					outputs = self.vanilla_pipeline(
						messages,
						max_new_tokens=16384,
						pad_token_id=self.vanilla_pipeline.tokenizer.eos_token_id,
						temperature=temp
					)
					response = outputs[0]["generated_text"][-1]["content"].strip()
					print("[removal] vanilla pipeline generation took {time:.2f} seconds".format(time=time.time() - start_time))

				if response == "nothing removed":
					print("No object removed.")
					is_success = False
					return current_scene, is_success
				
				scene_after = json.loads(response)

				n_objs_scene_before = len(current_scene.get("objects"))
				n_objs_scene_after = len(scene_after.get("objects"))

				if n_objs_scene_after < n_objs_scene_before:
					print(f"SUCCESS for removal! after: {n_objs_scene_after}, before: {n_objs_scene_before}")
					is_success = True
					return scene_after, is_success
				else:
					print("ERROR: no object was removed. response: ", scene_after, "prompt:", prompt)
					
			except Exception as exc:
				traceback.print_exc()
				print(f"Failed to parse object index from response")
			
			if remaining_attempts > 0:
				remaining_attempts -= 1
				print(f"Retrying remove_object() ... {remaining_attempts} attempts left.")
			else:
				print("Max attempts reached. Returning current scene without any changes.")
				is_success = False
				return current_scene, is_success
				
			gc.collect()
			torch.cuda.empty_cache()
	
	def generate_full_scene(self, room_type=None, n_objects=None, scene_bounds_only=None, do_rendering_with_object_count=False, pth_viz_output=None, scene_id=None, do_skip_removals=False):
		
		self.dataset_stats_for_prompt = self._prepare_dataset_stats_for_object_sampler(room_type)
		self.floor_object_sampler = FloorObjectSampler(self.dataset_stats_for_prompt.get("floor_area_n_objects"))
		
		floor_area = create_floor_plan_polygon(scene_bounds_only.get("bounds_bottom")).area
			
		if n_objects == None:
			n_objects = self.floor_object_sampler.sample_obj_count_for_floor_area(floor_area, do_prop_sampling=self.do_prop_sampling_for_prompt)[0]
		
		# sample few-shot examples from training set
		few_shot_samples = None
		if self.k_few_shot_samples > 0:
			few_shot_samples = self.floor_object_sampler.sample_few_shot_samples(floor_area, n_objects, k=self.k_few_shot_samples)

		if self.floor_object_sampler == None and n_objects == None:
			print("ERROR: floor_object_sampler is None and n_objects is None. Please provide a valid number of objects or re-initialize the floor_object_sampler by providing a dataset during initialization.")
			return None
		
		prompt = f"create a {room_type if room_type != None else 'room'} with {n_objects} objects."

		if scene_bounds_only == None:
			scene_bounds_only = self._sample_random_bounds(self.dataset_train, room_type)
		
		system_prompt = self._get_system_prompt_zeroshot_handle_user_instr(few_shot_samples=few_shot_samples)

		return self.handle_prompt(prompt, scene_bounds_only, system_prompt, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, scene_id=scene_id, do_skip_removals=do_skip_removals)


	def _run_commands_sequential(self, current_scene, all_commands, gt_steps=None, skip_per_step_eval=False, do_rendering_with_object_count=False, pth_viz_output=None):
		"""Execute pre-decomposed add/remove commands in the given order with optional per-step eval."""
		step_results = [] if gt_steps is not None else None

		print("=============================================")
		print(f"existing scene with # of objects: {len(current_scene.get('objects', []))}")
		for cmd in all_commands:
			print(f"  {cmd}")
		print("=============================================")

		current_scene = self.sampling_engine.sample_all_assets(current_scene, is_greedy_sampling=True)

		for step_idx, command in enumerate(all_commands):
			scene_before = copy.deepcopy(current_scene)

			if command.startswith("<add>"):
				prompt_obj = re.search(r'<add>(.*?)</add>', command).group(1).strip().lower()
				current_scene, _ = self.add_object(
					prompt_obj, current_scene,
					do_rendering_with_object_count=do_rendering_with_object_count,
					pth_viz_output=pth_viz_output, temp=0.7,
				)
				if gt_steps is not None and not skip_per_step_eval:
					step_results.append(evaluate_seq_step_add(gt_steps[step_idx], current_scene, scene_before))

			elif command.startswith("<remove>"):
				prompt_obj = re.search(r'<remove>(.*?)</remove>', command).group(1).strip().lower()
				current_scene, _ = self.remove_object(
					prompt_obj, current_scene,
					do_rendering_with_object_count=do_rendering_with_object_count,
					pth_viz_output=pth_viz_output,
				)
				if gt_steps is not None and not skip_per_step_eval:
					step_results.append(evaluate_seq_step_remove(gt_steps[step_idx], current_scene, scene_before))

			else:
				print(f"WARNING: unrecognised command format, skipping: {command}")

		overall_success = len(current_scene.get("objects", [])) > 0
		return current_scene, overall_success, step_results

	def _decompose_prompt_to_commands(self, prompt, current_scene, system_prompt=None):
		"""Use zero-shot LLM to decompose a natural language prompt into add/remove commands."""
		query = self._build_full_query_for_zeroshot_model(prompt, scenegraph=current_scene)
		if system_prompt is None:
			system_prompt = self._get_system_prompt_zeroshot_handle_user_instr(few_shot_samples=None)
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": query},
		]

		start_time = time.time()
		if self.vanilla_vllm_engine is not None:
			formatted_prompt = self.vanilla_tokenizer.apply_chat_template(
				messages, tokenize=False, add_generation_prompt=True,
			)
			vllm_outputs = self.vanilla_vllm_engine.generate(
				[formatted_prompt],
				SamplingParams(max_tokens=4096, temperature=0.7, top_p=0.95, top_k=50),
			)
			response = vllm_outputs[0].outputs[0].text
		else:
			torch.use_deterministic_algorithms(False)
			outputs = self.vanilla_pipeline(
				messages, max_new_tokens=4096,
				pad_token_id=self.vanilla_pipeline.tokenizer.eos_token_id,
				temperature=0.7,
			)
			response = outputs[0]["generated_text"][-1]["content"]
			torch.use_deterministic_algorithms(True)

		print(f"[handle_prompt] vanilla pipeline generation took {time.time() - start_time:.2f}s")

		response_json = json.loads(response)
		if response_json.get("commands") is None:
			raise ValueError("No commands in response")

		return response_json["commands"]

	def _run_removes_then_adds(self, current_scene, remove_commands, add_commands, do_skip_removals=False, do_rendering_with_object_count=False, pth_viz_output=None):
		"""Process remove commands first, then add commands (shuffled or sequential)."""

		# Step 1: Process REMOVE commands
		if do_skip_removals and len(remove_commands) > 0:
			print(f"Skipping {len(remove_commands)} remove commands (do_skip_removals=True)")
		else:
			for command in remove_commands:
				prompt_obj = re.search(r'<remove>(.*?)</remove>', command).group(1).strip().lower()
				if not prompt_obj:
					print(f"Skipping empty remove command: {command}")
					continue
				current_scene, _ = self.remove_object(
					prompt_obj, current_scene,
					do_rendering_with_object_count=do_rendering_with_object_count,
					pth_viz_output=pth_viz_output,
				)

		scene_after_removals = copy.deepcopy(current_scene)

		# Step 2: Process ADD commands
		if len(add_commands) == 0:
			if len(current_scene.get("objects", [])) > 0 or len(remove_commands) > 0:
				return current_scene, True
			raise ValueError("No commands to process")

		if self.do_bon_shuffling and len(add_commands) > 1:
			best_scene, shuffle_success = self._run_shuffled_additions(
				add_commands, scene_after_removals,
				do_rendering_with_object_count=do_rendering_with_object_count,
				pth_viz_output=pth_viz_output,
			)
			if shuffle_success:
				return best_scene, True
			raise ValueError("All shuffles failed")

		# Standard sequential add
		for command in add_commands:
			prompt_obj = re.search(r'<add>(.*?)</add>', command).group(1).lower()
			current_scene, _ = self.add_object(
				prompt_obj, current_scene,
				do_rendering_with_object_count=do_rendering_with_object_count,
				pth_viz_output=pth_viz_output, temp=0.7,
			)

		if len(current_scene.get("objects", [])) > 0:
			return current_scene, True
		raise ValueError("No objects added")

	def handle_prompt(self, prompt, current_scene=None, system_prompt=None, room_type=None, do_rendering_with_object_count=False, pth_viz_output=None, scene_id=None, all_commands=None, gt_steps=None, skip_per_step_eval=False, do_skip_removals=False):
		if current_scene is None:
			current_scene = self._sample_random_bounds(self.dataset_train, room_type)

		if self.dataset_stats_for_prompt is None:
			self.dataset_stats_for_prompt = self._prepare_dataset_stats_for_object_sampler(current_scene.get("room_type"))

		if all_commands is not None:
			# PATH B: pre-decomposed commands → run sequentially
			return self._run_commands_sequential(
				current_scene, all_commands, gt_steps=gt_steps,
				skip_per_step_eval=skip_per_step_eval,
				do_rendering_with_object_count=do_rendering_with_object_count,
				pth_viz_output=pth_viz_output,
			)
		else:
			# PATH A: decompose prompt via zero-shot LLM, then removes-first adds
			use_loaded_prompts = self.load_prompts_from and scene_id and scene_id in self.saved_prompts

			remaining_attempts = copy.copy(self.max_n_attempts)
			while True:
				try:
					if use_loaded_prompts:
						commands = self.saved_prompts[scene_id]["commands"]
					else:
						commands = self._decompose_prompt_to_commands(prompt, current_scene, system_prompt)

					remove_commands = [c for c in commands if c.startswith("<remove>")]
					add_commands    = [c for c in commands if c.startswith("<add>")]

					print("=============================================")
					print(f"Total: {len(commands)} commands ({len(remove_commands)} removes, {len(add_commands)} adds)")
					print(f"Remove commands: {remove_commands}")
					print(f"Add commands: {add_commands}")
					print("=============================================")

					current_scene, is_success = self._run_removes_then_adds(
						current_scene, remove_commands, add_commands,
						do_skip_removals=do_skip_removals,
						do_rendering_with_object_count=do_rendering_with_object_count,
						pth_viz_output=pth_viz_output,
					)

					if is_success and self.save_prompts_to and scene_id and not use_loaded_prompts:
						self._save_prompts(scene_id, prompt, commands, current_scene)

					return current_scene, is_success, None

				except Exception as exc:
					print(f"Error: {exc}")
					traceback.print_exc()

					if use_loaded_prompts:
						print("Loaded prompts failed - not retrying")
						return current_scene, False, None

				if remaining_attempts > 0:
					remaining_attempts -= 1
					print(f"Retrying handle_prompt() ... {remaining_attempts} attempts left.")
				else:
					print("Max attempts reached. Returning current scene.")
					return current_scene, False, None

				gc.collect()
				torch.cuda.empty_cache()