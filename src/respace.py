import json
import re
import copy
import gc
import torch
import numpy as np
import traceback
from typing import Optional, Dict, List, Any, Union
import os
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from accelerate import Accelerator
import random
import pdb
from pathlib import Path
from dotenv import load_dotenv

# Import from your existing utils/modules
from src.utils import get_model, get_llama_vanilla_pipeline, create_floor_plan_polygon, create_category_lookup
from src.sample import AssetRetrievalModule
from src.dataset import build_full_instruction_from_prompt, sample_prompt, load_train_val_test_datasets
from src.test import run_instr
from src.viz import render_full_scene_and_export_with_gif, create_360_video_full
from src.vllm_inference import VLLMWrapper

class ReSpace:
	def __init__(self, model_id="gradient-spaces/respace-sg-llm-1.5b", env_file=".env", dataset_room_type="all", use_gpu=True, accelerator=None, n_bon_sgllm=8, n_bon_assets=1, do_prop_sampling_for_prompt=True, do_icl_for_prompt=True, do_class_labels_for_prompt=True, use_vllm=False, do_removal_only=False, k_few_shot_samples=2):

		load_dotenv(env_file)
		
		# prepare models
		self.model, self.tokenizer, self.max_seq_length = get_model(model_id, use_gpu, accelerator, do_not_load_hf_model=(use_vllm == True or do_removal_only == True))
		self.use_vllm = use_vllm
	
		# load SG-LLM
		self.vllm_engine = None
		if use_vllm and not do_removal_only:
			try:
				self.vllm_engine = VLLMWrapper(
					model_id=model_id,
					tokenizer=self.tokenizer,
					gpu_memory_utilization=0.2,
					max_model_len=self.max_seq_length,
				)
				print("SG-LLM: vLLM initialized successfully")
			except Exception as e:
				print(f"Failed to initialize vLLM: {e}. Falling back to regular generation.")
				self.use_vllm = False

		# load zero-shot LLM
		self.vanilla_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
		self.vanilla_vllm_engine = None
		self.vanilla_pipeline = None
		_, self.vanilla_tokenizer, _ = get_model(self.vanilla_model_id, use_gpu, accelerator=None, do_not_load_hf_model=True)
		if use_vllm and self.use_vllm:
			try:
				self.vanilla_vllm_engine = VLLMWrapper(
					model_id=self.vanilla_model_id,
					tokenizer=self.vanilla_tokenizer,
					gpu_memory_utilization=0.85,
					max_model_len=5000,
				)
				print("Vanilla LLM: vLLM initialized successfully for vanilla pipeline")
			except Exception as e:
				print(f"Failed to initialize vLLM for vanilla pipeline: {e}. Using regular pipeline.")
				self.vanilla_pipeline = get_llama_vanilla_pipeline()
		else:
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

		self.max_n_attempts = 10

	def _prepare_dataset_stats_for_object_sampler(self, gen_room_type=None):
		if gen_room_type == None:
			room_type_filter = "nofilter"
		else:
			room_type_filter = gen_room_type

		pth_dataset_stats = os.path.join(os.getenv("PTH_DATASET_CACHE"), f"merged_dataset_stats_{self.dataset_room_type}_{room_type_filter}.pkl")

		if os.path.exists(pth_dataset_stats):
			print("loading stats file...")
			all_stats = pickle.load(open(pth_dataset_stats, "rb"))
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
	
	def add_object(self, prompt, current_scene, do_sample_assets_for_input_scene=False, do_rendering_with_object_count=False, temp=None, do_dynamic_temp=True, pth_viz_output=None):
		print("adding object...")

		if do_sample_assets_for_input_scene:
			current_scene = self.sampling_engine.sample_all_assets(current_scene, is_greedy_sampling=(True if self.n_bon_assets == 1 else False))
		
		batch_full_instrs = self._prepare_input_for_addition(prompt, current_scene=current_scene)
		len_before = len(current_scene.get("objects"))

		temp = copy.copy(temp)
		remaining_attempts = copy.copy(self.max_n_attempts)

		while True:
			try:
				if do_dynamic_temp and remaining_attempts < self.max_n_attempts and temp != None:
					# first, decrease temp to get more deterministic results
					temp = max(temp - 0.05, 0.4)
					# if not successful, increase temp to get more diverse results and see if we escape weird behavior
					if temp == 0.4:
						temp = 1.2
				print(f"temp: {temp}")
				best_result = run_instr(prompt, current_scene, batch_full_instrs, self.model, self.tokenizer, self.max_seq_length, self.accelerator, self.n_bon_sgllm, self.n_bon_assets, self.sampling_engine, pth_viz_output, do_rendering_with_object_count=do_rendering_with_object_count, temp=temp, vllm_engine=(self.vllm_engine if self.use_vllm else None))

				if best_result.get("scene") != None and len(best_result.get("scene").get("objects")) == len_before + 1:
					print(f"SUCCESS! after: {len(best_result.get('scene').get('objects'))}, before: {len_before}")
					current_scene = best_result.get("scene")
					current_scene["objects"][-1]["prompt"] = prompt
					is_success = True
					return current_scene, is_success
				else:
					print("ERROR: no object was added. response: ", best_result.get("scene"))
					
			except Exception as exc:
				print(exc)
				traceback.print_exc()
				print("Failed to add object. Retrying...")
			
			if remaining_attempts > 0:
				remaining_attempts -= 1
				print(f"Retrying... {remaining_attempts} attempts left.")
			else:
				print("Max attempts reached. Returning current scene without any changes.")
				is_success = False
				return current_scene, is_success
			
			gc.collect()
			torch.cuda.empty_cache()
	
	def remove_object(self, prompt, current_scene, do_rendering_with_object_count=False, do_dynamic_temp=True, pth_viz_output=None, idx=None):
		print("removing object...")

		print(f"<remove>{prompt}<remove>")

		# Build a query for the vanilla pipeline to identify which object to remove
		query = f"""<remove>{prompt}<remove>
<scenegraph>{json.dumps(current_scene)}</scenegraph>"""
		
		# system_prompt = "You are a world-class interior design expert. Your task is to identify which object from the scene best matches the given description. Respond with ONLY the index of the object in the scene's objects list (0-based indexing). If no object matches well, respond with -1."
		system_prompt = """you are a world-class leading interior design expert. your task is to remove furniture given the descriptions in the header and the current list of furniture in the body. you must respond ONLY with a valid JSON string that matches precisely the *format* of the existing JSON in the request.

if there are multiple objects that match the description precisely, you should remove all of them.

the prompt for the object to be removed will be given in the header between <remove>...</remove> tags. the current scene will be given as a JSON object in the body between <scenegraph>...</scenegraph> tags.

in the successful case, your output contains one or N fewer objects in the "objects" list and the rest of the JSON object should be EXACTLY identical to the input.

you can also remove all objects if the prompt matches those objects. in that case, you provide an empty list for the "objects" key.

you can further assume that in most cases, there will be at least one object in the scene that matches the description roughly. this object shall be removed.

only output the JSON (with the removed objects) as a plain string and nothing else."""

		# if no object matches the description, you should respond with 'nothing removed'.

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
				torch.use_deterministic_algorithms(False)
				if self.vanilla_vllm_engine is not None:
					vllm_prompt = f"<s>[INST] {system_prompt} [/INST]\n\n{query}</s>"
					inputs = self.vanilla_tokenizer(vllm_prompt, return_tensors="pt")
					input_ids = inputs["input_ids"]
					attention_mask = inputs["attention_mask"]
					response = self.vanilla_vllm_engine.generate(
						input_ids, 
						attention_mask,
						max_new_tokens=16384,
						temperature=temp,
						top_p=0.95,
						top_k=50,
					)
					# TODO: format correctly for output
				else:
					outputs = self.vanilla_pipeline(
						messages,
						max_new_tokens=16384,
						pad_token_id=self.vanilla_pipeline.tokenizer.eos_token_id,
						temperature=temp
					)
					response = outputs[0]["generated_text"][-1]["content"].strip()
				torch.use_deterministic_algorithms(True)
				
				response = outputs[0]["generated_text"][-1]["content"].strip()

				if response == "nothing removed":
					print("No object removed.")
					is_success = False
					return current_scene, is_success
				
				scene_after = json.loads(response)

				n_objs_scene_before = len(current_scene.get("objects"))
				n_objs_scene_after = len(scene_after.get("objects"))

				if n_objs_scene_after < n_objs_scene_before:
					print(f"SUCCESS! after: {n_objs_scene_after}, before: {n_objs_scene_before}")
					is_success = True
					return scene_after, is_success
				else:
					print("ERROR: no object was removed. response: ", scene_after, "prompt:", prompt)
					
			except Exception as exc:
				traceback.print_exc()
				print(f"Failed to parse object index from response")
			
			if remaining_attempts > 0:
				remaining_attempts -= 1
				print(f"Retrying... {remaining_attempts} attempts left.")
			else:
				print("Max attempts reached. Returning current scene without any changes.")
				is_success = False
				return current_scene, is_success
				
			gc.collect()
			torch.cuda.empty_cache()
	
	def generate_full_scene(self, room_type=None, n_objects=None, scene_bounds_only=None, do_rendering_with_object_count=False, pth_viz_output=None):
		
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

		return self.handle_prompt(prompt, scene_bounds_only, system_prompt, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)
		
		
	def handle_prompt(self, prompt, current_scene=None, room_type=None, do_rendering_with_object_count=False, pth_viz_output=None):

		if current_scene == None:
			current_scene = self._sample_random_bounds(self.dataset_train, room_type)

		# skip few shot samples here for the moment (we would need to inject n_objects as a prior, would probably randomly sample this number if not provided ?)
		# floor_area = create_floor_plan_polygon(current_scene.get("bounds_bottom")).area
		# few_shot_samples = None
		# if self.k_few_shot_samples > 0:
		# 	few_shot_samples = self.floor_object_sampler.sample_few_shot_samples(floor_area, n_objects, k=self.k_few_shot_samples)

		if self.dataset_stats_for_prompt == None:
			self.dataset_stats_for_prompt = self._prepare_dataset_stats_for_object_sampler(current_scene.get("room_type"))

		query = self._build_full_query_for_zeroshot_model(prompt, scenegraph=current_scene)

		system_prompt = self._get_system_prompt_zeroshot_handle_user_instr(few_shot_samples=None)

		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": query},
		]
		
		remaining_attempts = copy.copy(self.max_n_attempts)
		while True:
			try:
				# get list of objects from vanilla pipeline
				torch.use_deterministic_algorithms(False)
				outputs = self.vanilla_pipeline(messages, max_new_tokens=4096, pad_token_id=self.vanilla_pipeline.tokenizer.eos_token_id, temperature=0.7)
				torch.use_deterministic_algorithms(True)
				response = outputs[0]["generated_text"][-1]["content"]

				# Parse response
				response_json = json.loads(response)
				if response_json.get("commands") is None:
					print("ERROR: no commands found in response.")
				else:
					# sort commands by remove first, then add
					response_json["commands"].sort(key=lambda x: (not x.startswith("<remove>"), x))

					print("=============================================")
					print(len(response_json.get("commands")), response_json)
					print("=============================================")
					
					# Process commands one by one
					print("processing commands...")
					for command in response_json.get("commands"):
						if command.startswith("<add>"):
							prompt = re.search(r'<add>(.*?)</add>', command).group(1).lower()
							temp = 0.7
							current_scene, is_success = self.add_object(prompt, current_scene, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, temp=temp)
						elif command.startswith("<remove>"):
							prompt = re.search(r'<remove>(.*?)</remove>', command).group(1).lower()
							current_scene, is_success = self.remove_object(prompt, current_scene, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)
						else:
							print(f"UNKNOWN COMMAND {command}")
				
				if len(current_scene.get("objects")) > 0:
					print("SUCCESS! after: ", len(current_scene.get("objects")))
					is_success = True
					return current_scene, is_success
				else:
					print("ERROR: no object was added")
					
			except Exception as exc:
				print(f"Error: {exc}")
				print(f"Response: {response}")
				traceback.print_exc()
			
			if remaining_attempts > 0:
				remaining_attempts -= 1
				print(f"Retrying... {remaining_attempts} attempts left.")
			else:
				print("Max attempts reached. Returning empty scene.")
				is_success = False
				return current_scene, is_success

			gc.collect()
			torch.cuda.empty_cache()
		

class FloorObjectSampler:
	def __init__(self, dataset_stats, num_bins_floor=25):
		self.floor_areas = np.array([item["floor_area"] for item in dataset_stats])
		self.object_counts = np.array([item["n_objects"] for item in dataset_stats])
		
		self.floor_min = np.min(self.floor_areas)
		self.floor_max = np.max(self.floor_areas)
		self.floor_bins = np.linspace(self.floor_min, self.floor_max, num_bins_floor + 1)
		
		self.obj_min = np.min(self.object_counts)
		self.obj_max = np.max(self.object_counts)
		self.obj_bins = np.linspace(self.obj_min - 0.5, self.obj_max + 0.5, self.obj_max - self.obj_min + 2)
		
		self.hist, _, _ = np.histogram2d(self.floor_areas, self.object_counts, bins=[self.floor_bins, self.obj_bins])
		
		epsilon = 1e-10
		row_sums = np.sum(self.hist, axis=1)
		row_sums = np.where(row_sums == 0, epsilon, row_sums)
		# rows are floor area bins, columns are object count bins, normalize by row so each floor area bin sums to 1
		self.conditional_probs = self.hist / row_sums[:, np.newaxis]

		self.objects_lookup = defaultdict(list)

		for item in dataset_stats:
			floor_area = item["floor_area"]
			obj_count = item["n_objects"]
			objects_list = item["object_prompts"]
			
			floor_bin = np.digitize(floor_area, self.floor_bins) - 1
			floor_bin = max(0, min(floor_bin, len(self.floor_bins) - 2))
			
			obj_bin = obj_count - self.obj_min
			obj_bin = max(0, min(obj_bin, len(self.conditional_probs[0]) - 1))
			
			key = (floor_bin, obj_bin)
			self.objects_lookup[key].append(objects_list)
	
	def sample_obj_count_for_floor_area(self, floor_area, do_prop_sampling=True, n=1):
		floor_area = np.clip(floor_area, self.floor_min, self.floor_max)
		floor_bin_idx = np.digitize(floor_area, self.floor_bins) - 1
		floor_bin_idx = max(0, min(floor_bin_idx, len(self.floor_bins) - 2))

		if do_prop_sampling:
			# sample from discrete distribution that is conditioned on floor area bin
			probs = self.conditional_probs[floor_bin_idx]
			if np.all(probs == 0):
				probs = np.ones_like(probs) / len(probs)
			obj_bin_idx = np.random.choice(len(probs), p=probs, size=n)

			obj_cnts = []
			for idx in obj_bin_idx:
				obj_cnts.append(self.obj_min + idx)
		else:
			# sample uniformly within given floor area bin, given obj_min and obj_max for that bin
			obj_cnts = []
			valid_obj_bins = np.where(self.hist[floor_bin_idx] > 0)[0]

			if len(valid_obj_bins) == 0:
				obj_bin_indices = np.random.randint(0, self.obj_max - self.obj_min + 1, size=n)
				for idx in obj_bin_indices:
					obj_cnts.append(self.obj_min + idx)
			else:
				# Get the min and max object counts in this floor bin
				min_obj_bin = valid_obj_bins.min()
				max_obj_bin = valid_obj_bins.max()
				min_obj_count = self.obj_min + min_obj_bin
				max_obj_count = self.obj_min + max_obj_bin
				
				# Sample uniformly from the range of valid object counts
				for _ in range(n):
					obj_count = np.random.randint(min_obj_count, max_obj_count + 1)
					obj_cnts.append(obj_count)

		return obj_cnts
	
	def sample_few_shot_samples(self, floor_area, n_objects, k=5):
		floor_area = np.clip(floor_area, self.floor_min, self.floor_max)
		floor_bin_idx = np.digitize(floor_area, self.floor_bins) - 1
		floor_bin_idx = max(0, min(floor_bin_idx, len(self.floor_bins) - 2))

		obj_bin_idx = n_objects - self.obj_min
		obj_bin_idx = max(0, min(obj_bin_idx, len(self.conditional_probs[0]) - 1))
		
		key = (floor_bin_idx, obj_bin_idx)
		obj_prompt_lists = []
		
		# Step 1: Try to get samples for the exact floor+object bin combination
		if key in self.objects_lookup and self.objects_lookup[key]:
			available = self.objects_lookup[key].copy()
			random.shuffle(available)
			obj_prompt_lists.extend(available[:min(k, len(available))])
		
		# Step 2: If we need more samples, collect all valid bins in the current floor area
		if len(obj_prompt_lists) < k:
			floor_bin_samples = []
			for obj_bin in range(len(self.conditional_probs[0])):
				test_key = (floor_bin_idx, obj_bin)
				if test_key in self.objects_lookup and self.objects_lookup[test_key]:
					floor_bin_samples.extend(self.objects_lookup[test_key])
			
			# If we have other samples from this floor bin, use them without duplicating
			if floor_bin_samples:
				# Filter out samples we've already taken
				available_samples = [s for s in floor_bin_samples if s not in obj_prompt_lists]
				random.shuffle(available_samples)
				to_take = min(k - len(obj_prompt_lists), len(available_samples))
				obj_prompt_lists.extend(available_samples[:to_take])
		
		# Step 3: If we still need more samples, search in adjacent floor bins
		if len(obj_prompt_lists) < k:
			# Create a list of all floor bins ordered by distance from current bin
			floor_bins_by_distance = sorted(range(len(self.floor_bins)-1), key=lambda x: abs(x - floor_bin_idx))
			
			for floor_bin in floor_bins_by_distance:
				if floor_bin == floor_bin_idx:  # Skip the current bin, already processed
					continue
					
				bin_samples = []
				for obj_bin in range(len(self.conditional_probs[0])): # for each bin in all object bins
					test_key = (floor_bin, obj_bin)
					if test_key in self.objects_lookup and self.objects_lookup[test_key]:
						bin_samples.extend(self.objects_lookup[test_key])
				
				if bin_samples:
					# Filter out samples we've already taken
					available_samples = [s for s in bin_samples if s not in obj_prompt_lists]
					random.shuffle(available_samples)
					to_take = min(k - len(obj_prompt_lists), len(available_samples))
					obj_prompt_lists.extend(available_samples[:to_take])
				
				# Stop if we've reached our target
				if len(obj_prompt_lists) >= k:
					break
		
		# Step 4: Last resort - if somehow we still don't have enough samples,
		# collect all samples from the entire histogram and sample randomly
		if len(obj_prompt_lists) < k:
			all_samples = []
			for f_bin in range(len(self.floor_bins)-1):
				for o_bin in range(len(self.conditional_probs[0])):
					test_key = (f_bin, o_bin)
					if test_key in self.objects_lookup and self.objects_lookup[test_key]:
						all_samples.extend(self.objects_lookup[test_key])
			
			if all_samples:
				# Filter out samples we've already taken
				available_samples = [s for s in all_samples if s not in obj_prompt_lists]
				
				# If we've somehow used all samples already, allow reuse
				if not available_samples and all_samples:
					available_samples = all_samples

				random.shuffle(available_samples)
				to_take = min(k - len(obj_prompt_lists), len(available_samples))
				obj_prompt_lists.extend(available_samples[:to_take])
		
		# if we still don't have k samples, we need to reuse some
		while len(obj_prompt_lists) < k and obj_prompt_lists:
			obj_prompt_lists.append(random.choice(obj_prompt_lists))

		random.shuffle(obj_prompt_lists)
		
		return obj_prompt_lists[:k]

	def visualize(self) -> None:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
		im = ax1.imshow(
			self.hist.T,
			origin='lower', 
			aspect='auto',
			extent=[self.floor_min, self.floor_max, self.obj_min, self.obj_max],
			cmap='viridis'
		)
		ax1.set_xlabel('Floor Area')
		ax1.set_ylabel('Number of Objects')
		ax1.set_title('2D Histogram of Floor Area vs. Object Count')
		plt.colorbar(im, ax=ax1, label='Count')
		im2 = ax2.imshow(
			self.conditional_probs.T, 
			origin='lower', 
			aspect='auto',
			extent=[self.floor_min, self.floor_max, self.obj_min, self.obj_max],
			cmap='plasma'
		)
		ax2.set_xlabel('Floor Area')
		ax2.set_ylabel('Number of Objects')
		ax2.set_title('P(Objects | Floor Area)')
		plt.colorbar(im2, ax=ax2, label='Probability')
		plt.tight_layout()
		plt.savefig("respace_full_floor_area_vs_object_count.png")