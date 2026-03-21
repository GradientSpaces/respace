import numpy as np
import pickle
import pandas as pd
import glob
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset
import copy
import os
from scipy.spatial.transform import Rotation as R
import json
from tqdm import tqdm
import pdb
from pathlib import Path
from shapely.geometry import Polygon, box
import random
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import math
from collections import Counter
from collections import defaultdict

from src.utils import create_floor_plan_polygon, get_pths_dataset_split, get_system_prompt_sgllm, is_high_quality_sample


def rotate_around_y(point, angle_radians):
	rotation_matrix = np.array([
		[np.cos(angle_radians), 0, np.sin(angle_radians)],
		[0, 1, 0],
		[-np.sin(angle_radians), 0, np.cos(angle_radians)]
	])
	rot_point = np.dot(rotation_matrix, point).tolist()
	rot_point = [ round(elem, 2) for elem in rot_point ]

	return rot_point

def combine_quaternion_with_y_rot_for_global_rot(original_quat, angle_radians):
	y_axis_rotation = R.from_euler('y', angle_radians).as_quat()
	original_rotation = R.from_quat(original_quat)
	combined_rotation = R.from_quat(y_axis_rotation) * original_rotation
	return [round(elem, 5) for elem in combined_rotation.as_quat().tolist()]

def rotate_obj(obj, angle_radians):
	obj["pos"] = rotate_around_y(obj["pos"], angle_radians)
	obj["rot"] = combine_quaternion_with_y_rot_for_global_rot(obj["rot"], angle_radians)

def rotate_scenegraph(sample_scene, angle_radians):
	if sample_scene.get("bounds_top"):
		for key in ["bounds_top", "bounds_bottom"]:
			sample_scene[key] = [rotate_around_y(point, angle_radians) for point in sample_scene[key]]
	if sample_scene.get("objects"):
		for obj in sample_scene.get("objects"):
			rotate_obj(obj, angle_radians)
	if sample_scene.get("pos"):
		rotate_obj(sample_scene, angle_radians)

def offset_bounds(sample_scene, shift_amount):
	if sample_scene.get("bounds_top") and shift_amount != 0:
		sample_scene["bounds_top"] = sample_scene["bounds_top"][shift_amount:] + sample_scene["bounds_top"][:shift_amount]
		sample_scene["bounds_bottom"] = sample_scene["bounds_bottom"][shift_amount:] + sample_scene["bounds_bottom"][:shift_amount]

def get_2d_bbox(pos, size):
	x, _, z = pos
	width, _, depth = size
	half_width, half_depth = width/2, depth/2
	return box(x - half_width, z - half_depth, x + half_width, z + half_depth)

def perturb_value_with_bounds(value, bounds, min_delta=-0.02, max_delta=0.02):
	min_val, max_val = bounds
	delta = np.random.uniform(min_delta, max_delta)
	new_val = round(value + delta, 2)
	# return np.clip(new_val, min_val, max_val)
	return new_val

def get_safe_perturbation(pos, size, floor_polygon, desc, max_attempts=10):	
	# Get bounds from the floor polygon
	minx, minz, maxx, maxz = floor_polygon.bounds

	# if not floor_polygon.contains(get_2d_bbox(pos, size)):
	# 	print("object not contained in floor polygon already !")
	# 	print(f"\tdesc: {desc}")
	# 	print(f"\tpos: {pos}")
	# 	print(f"\tsize: {size}")
	# 	print(f"\tbounds: ", minx, minz, maxx, maxz)
	
	for i in range(max_attempts):
		# print(f"trying to find perturbation... ({i}/{max_attempts})")
		pos_perturbed = pos.copy()
		pos_perturbed[0] = perturb_value_with_bounds(pos[0], bounds=(minx, maxx))
		pos_perturbed[2] = perturb_value_with_bounds(pos[2], bounds=(minz, maxz))
		
		size_perturbed = size.copy()
		size_perturbed[0] = perturb_value_with_bounds(size[0], bounds=(minx, maxx))
		size_perturbed[2] = perturb_value_with_bounds(size[2], bounds=(minx, maxx))
		
		new_bbox = get_2d_bbox(pos_perturbed, size_perturbed)
		if floor_polygon.contains(new_bbox):
			# print("valid perturbation found at iter ", i)
			# print(f"\tdesc: {desc}")
			# print(f"\tbounds: ", minx, minz, maxx, maxz)
			# print(f"\tpos: {pos} -> {pos_perturbed}")
			# print(f"\tsize: {size} -> {size_perturbed}")
			return pos_perturbed, size_perturbed
			
	# if no valid perturbation found, return original values
	# print("no valid perturbation found, returning original values")
	# print(f"\tdesc: {desc}")
	return pos, size

def perturb_scene(sample_scene, floor_polygon):
	if sample_scene.get("objects"):
		for obj in sample_scene["objects"]:
			new_pos, new_size = get_safe_perturbation(obj["pos"], obj["size"], floor_polygon, obj["desc"])
			obj["pos"] = new_pos
			obj["size"] = new_size
	if sample_scene.get("pos"):
		new_pos, new_size = get_safe_perturbation(sample_scene["pos"], sample_scene["size"], floor_polygon, sample_scene["desc"])
		sample_scene["pos"] = new_pos
		sample_scene["size"] = new_size

def do_random_augm_on_sgs(sample, augm_prob=0.85):

	sg_input = sample.get("sg_input")
	sg_output_add = sample.get("sg_output_add")

	# (1) with 15% prob, we don't perform any augmentation (return original data)
	# TODO: could be hyperparam itself actually...
	if np.random.rand() > augm_prob:
		return sg_input, sg_output_add

	sg_input_augm = copy.deepcopy(json.loads(sg_input))
	sg_output_add_augm = copy.deepcopy(json.loads(sg_output_add))

	# (2) do random rotation
	angle_radians = np.radians(np.random.choice([0, 90, 180, 270]))
	rotate_scenegraph(sg_input_augm, angle_radians)
	rotate_scenegraph(sg_output_add_augm, angle_radians)

	# (3) circular shift of room boundaries
	shift_amount = np.random.randint(0, len(sg_input_augm.get("bounds_bottom")))
	offset_bounds(sg_input_augm, shift_amount)
	offset_bounds(sg_output_add_augm, shift_amount)

	# (4) scale entire "polygon" of bounds equally by +[0, 5] cm
	# TODO
	# scale_factor = np.random.uniform(0, 0.05)
	# ...

	# (5) Perturb object sizes and positions
	floor_polygon = create_floor_plan_polygon(sg_input_augm.get("bounds_bottom"))
	perturb_scene(sg_input_augm, floor_polygon)
	perturb_scene(sg_output_add_augm, floor_polygon)

	return json.dumps(sg_input_augm), json.dumps(sg_output_add_augm)
	
def create_dataset_from_files(pth_output, room_type, dataset_split):
	data = {
		"room_type": [],
		"n_objects": [],
		"pth_orig_file": [],
		"split": [],
		"scene": [],
	}

	pth_root = os.getenv("PTH_STAGE_2_DEDUP")

	all_pths = get_pths_dataset_split(room_type, dataset_split)
	
	for pth_scene in tqdm(all_pths, desc=f"Loading {room_type or 'all'} ({dataset_split} split)"):
		with open(os.path.join(pth_root, pth_scene), 'r') as f:
			scene = json.load(f)

		data["room_type"].append(scene.get("room_type"))
		data["n_objects"].append(len(scene.get("objects")))
		data["split"].append(dataset_split)
		data["scene"].append(scene)
		data["pth_orig_file"].append(pth_scene)

		# data["sg_input"].append(sample.get("sg_input"))
		# data["sg_output_add"].append(sample.get("sg_output_add"))
		#data["prompt_var"].append(sample.get("prompt_var"))
		#data["n_objects_query"].append(sample.get("n_objects_query"))
		#data["n_objects_full"].append(sample.get("n_objects_full"))
		#data["is_complete"].append(sample.get("is_complete"))

	dataset = Dataset.from_dict(data)

	with open(pth_output, 'wb') as fp:
		pickle.dump(dataset, fp)

	return dataset

def simplify_descs_for_ablation(sg_raw, all_assets_metadata_simple_descs):
	sg_simplified = copy.deepcopy(json.loads(sg_raw))
	if sg_simplified.get("objects"):
		for obj in sg_simplified["objects"]:
			obj["desc"] = all_assets_metadata_simple_descs.get(obj["desc"])
	if sg_simplified.get("desc"):
		sg_simplified["desc"] = all_assets_metadata_simple_descs.get(sg_simplified["desc"])
	return json.dumps(sg_simplified)

def simplify_sample(sample, all_assets_metadata_simple_descs):
	sample["sg_input"] = simplify_descs_for_ablation(sample["sg_input"], all_assets_metadata_simple_descs)
	sample["sg_output_add"] = simplify_descs_for_ablation(sample["sg_output_add"], all_assets_metadata_simple_descs)
	return sample

def create_full_scene_from_before_and_added(scene_before, obj_add):
	scene_after = copy.deepcopy(scene_before)
	scene_after["objects"].append(obj_add)
	return scene_after

def ensure_order_of_keys_for_sg_input_dict(sg_input, do_keep_jids=False):
	sg_input_ordered = {}

	sg_input_ordered["room_type"] = sg_input.get("room_type")
	sg_input_ordered["bounds_top"] = sg_input.get("bounds_top")
	sg_input_ordered["bounds_bottom"] = sg_input.get("bounds_bottom")
	
	# for each object in the scene, ensure fixed order such that we always have "desc", "size", "pos", "rot":
	objects_ordered = []
	for obj in sg_input.get("objects"):
		obj_ordered = {}
		obj_ordered["desc"] = obj.get("desc")
		obj_ordered["size"] = obj.get("size")
		obj_ordered["pos"] = obj.get("pos")
		obj_ordered["rot"] = obj.get("rot")
		if do_keep_jids:
			obj_ordered["jid"] = obj.get("jid")
		objects_ordered.append(obj_ordered)
	sg_input_ordered["objects"] = objects_ordered

	return sg_input_ordered

def create_instruction_dict(prompt, scene_query, obj_add, room_type, n_objects_full, is_complete=False, do_keep_jids=False):
	# remove id from sg_input
	scene_query.pop("room_id")

	# ensure fixed order of keys for sg_input
	scene_query = ensure_order_of_keys_for_sg_input_dict(scene_query, do_keep_jids=do_keep_jids)

	return {
		"instr_type": "add",
		"prompt": prompt,
		"room_type": room_type,
		"sg_input": json.dumps(scene_query),
		"sg_output_add": json.dumps(obj_add),
		"n_objects_query": len(scene_query["objects"]),
		"n_objects_full": n_objects_full,
		"is_complete": is_complete
	}

def clean_copy_of_objects(objects, do_keep_jids=False):
	cleaned = deepcopy(objects)
	
	if do_keep_jids:
		return cleaned
	
	if isinstance(cleaned, list):
		for obj in cleaned:
			obj.pop("jid", None)
	else:
		cleaned.pop("jid", None)
	return cleaned

def get_exposure_factor(n_objects, lambda_instr_exp=None):
	if lambda_instr_exp is None:
		return 1
	
	# lambda_instr_exp should be in range 0.0 - 0.9 for reasonably scales ?
	
	# base = 4 * np.log(n_objects + 1) # +1 to handle n_objects=1 case
	# scaled = base * (lambda_instr_exp ** 2)
	# exposure = max(1, int(scaled + 1))
	# print(f"n_objects: {n_objects}, n_instructions: {n_instructions}")
	exposure = np.exp(lambda_instr_exp * n_objects)

	return exposure
	
def plot_scaling_curves():
	n_objects = range(2, 50)
	for param in [0.1, 0.5, 1.0, 1.5, 2.0]:
		instructions = [get_exposure_factor(n, param) for n in n_objects]
		plt.plot(n_objects, instructions, label=f'param={param}')
	
	plt.xlabel('Number of Objects')
	plt.ylabel('Number of Instructions')
	plt.title('Instruction Exposure Scaling')
	plt.legend()
	plt.show()

def get_sampling_weights(dataset, lambda_instr_exp):
	weights = [get_exposure_factor(sample["n_objects"], lambda_instr_exp) for sample in dataset]
	return np.array(weights) / sum(weights)

class WeightedRandomSampler(torch.utils.data.Sampler):
	def __init__(self, weights, num_samples, replacement=True):
		self.weights = weights
		self.num_samples = num_samples
		self.replacement = replacement
	
	def __iter__(self):
		return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
	
	def __len__(self):
		return self.num_samples

def load_train_val_test_datasets(lambda_instr_exp=None, use_cached_dataset=True, room_type="all", do_sanity_check=False, seed=1234, accelerator=None):
	pth_train = f"{os.getenv('PTH_DATASET_CACHE')}/dataset_{room_type}_train.pkl"
	pth_val = f"{os.getenv('PTH_DATASET_CACHE')}/dataset_{room_type}_val.pkl"
	pth_test = f"{os.getenv('PTH_DATASET_CACHE')}/dataset_{room_type}_test.pkl"

	if not use_cached_dataset:
		dataset_train = create_dataset_from_files(pth_train, room_type, "train")
		dataset_val = create_dataset_from_files(pth_val, room_type, "val")
		dataset_test = create_dataset_from_files(pth_test, room_type, "test")
	else:
		print(f"loading cached dataset from {os.getenv('PTH_DATASET_CACHE')}")
		dataset_train = pd.read_pickle(pth_train)
		dataset_val = pd.read_pickle(pth_val)
		dataset_test = pd.read_pickle(pth_test)

	# ONLY FOR TRAINING SPLIT: get normalized sampling weights
	if lambda_instr_exp is not None:
		train_weights = get_sampling_weights(dataset_train, lambda_instr_exp)
		dataset_train = dataset_train.add_column("sampling_weight", train_weights)

	# plots weights vs number of objects and save as image
	# x axis: number of objects, y axis: sampling weight
	# plt.scatter([sample["n_objects"] for sample in dataset_train], train_weights)
	# plt.xlabel('Number of Objects')
	# plt.ylabel('Sampling Weight')
	# plt.title('Sampling Weights vs Number of Objects')
	# plt.savefig(f"sampling_weights_{room_type}.png")

	gen = np.random.default_rng(seed)
	dataset_train = dataset_train.shuffle(generator=gen)
	dataset_val = dataset_val.shuffle(generator=gen)
	dataset_test = dataset_test.shuffle(generator=gen)

	if do_sanity_check:
		n_max = 32
		dataset_train = dataset_train.select(range(n_max))
		dataset_val = dataset_val.select(range(n_max))
		dataset_test = dataset_test.select(range(n_max))

	if accelerator:
		dataset_train = accelerator.prepare(dataset_train)
		dataset_val = accelerator.prepare(dataset_val)
		dataset_test = accelerator.prepare(dataset_test)

	print(f"len of train dataset: {len(dataset_train)}")
	print(f"len of val dataset: {len(dataset_val)}")
	print(f"len of test dataset: {len(dataset_test)}")

	return dataset_train, dataset_val, dataset_test

def build_full_instruction_from_prompt(prompt, sg_input):
	sg_input_str = json.dumps(ensure_order_of_keys_for_sg_input_dict(json.loads(sg_input)))
	return f"<instruction>\n\t<add>{prompt}</add>\n</instruction>\n<scenegraph>\n\t{sg_input_str}\n</scenegraph>"

def sample_prompt(all_prompts, jid):
	if "-(" in jid:
		jid_clean = jid.split("-(")[0]
	else:
		jid_clean = jid
	return random.choice(all_prompts[jid_clean])
	
def create_instruction_from_scene(sample, all_prompts, all_assets_metadata_simple_descs=None, do_simple_descs=False, do_keep_jids=False, do_augm=False):
	scene = sample["scene"]
	n_objects = sample["n_objects"]
	room_type = sample["room_type"]
	
	# get weight for instruction generation
	# n_zero_start = min(1 // n_objects, 0.1)
	# n_full_scene = min(1 // n_objects, 0.1)

	# problem !!
	# during training, we want to set the seeds for sanity check repro ("overfit on same scene combos")
	# during eval, we set do_augm=false because we don't want augm on val/test splits, so this will run infinite loop
	# if not do_augm:
	# 	np.random.seed(4567)
	# 	random.seed(4567)
	# 	torch.manual_seed(4567)

	n_zero_start = 0.1
	n_full_scene = 0.1

	n_random = 1.0 - n_zero_start - n_full_scene

	complete_scene = deepcopy(scene)
	n_objects_full = len(complete_scene["objects"])

	# sample instruction style from probs
	instr_style = np.random.choice(["zero_start", "full_scene", "random"], p=[n_zero_start, n_full_scene, n_random])

	# print(instr_style)

	if instr_style == "zero_start":
		# Start with empty scene but keep everything else
		scene_query = deepcopy(scene)
		scene_query["objects"] = []

		# select random object to add
		obj_add = random.choice(scene["objects"])
		prompt = sample_prompt(all_prompts, obj_add.get("jid"))
		obj_add = clean_copy_of_objects(obj_add, do_keep_jids)

		instr = create_instruction_dict(prompt, scene_query, obj_add, room_type, n_objects_full, do_keep_jids=do_keep_jids)

	elif instr_style == "full_scene":
		# shuffle scene then pick first one and remove from list
		scene_query = deepcopy(scene)

		obj_add = random.choice(scene["objects"])
		prompt = sample_prompt(all_prompts, obj_add.get("jid"))
		obj_add = clean_copy_of_objects(obj_add, do_keep_jids)

		remaining_objects = clean_copy_of_objects(scene["objects"], do_keep_jids)
		random.shuffle(remaining_objects)
		scene_query["objects"] = [obj for obj in remaining_objects if obj != obj_add]

		instr = create_instruction_dict(prompt, scene_query, obj_add, room_type, n_objects_full, is_complete=True, do_keep_jids=do_keep_jids)

	else:
		scene_query = deepcopy(scene)
		random.shuffle(scene_query["objects"])
		
		# drop between 0 and N-1 objects
		n_total = len(scene_query["objects"])
		m_drop = np.random.choice(np.arange(0, n_total-1))
		n_total_new = n_total - m_drop
		scene_query["objects"] = scene_query["objects"][:n_total_new]

		obj_add = scene_query["objects"][-1]
		prompt = sample_prompt(all_prompts, obj_add.get("jid"))
		obj_add = clean_copy_of_objects(obj_add, do_keep_jids)

		scene_query["objects"].pop()
		scene_query["objects"] = clean_copy_of_objects(scene_query["objects"], do_keep_jids)

		instr = create_instruction_dict(prompt, scene_query, obj_add, room_type, n_objects_full, is_complete=(True if n_total_new == n_objects_full else False), do_keep_jids=do_keep_jids)

	# simplify descriptions if needed
	if do_simple_descs:
		instr = simplify_sample(instr, all_assets_metadata_simple_descs)

	return instr

def format_and_tokenize(tokenizer, full_sample_instr, sample_sg_output_full, max_seq_length, padding_free, truncate=True):
	
	formatted_text = format_with_chat_template(tokenizer, full_sample_instr, sample_sg_output_full)

	if truncate:
		tokenized_inputs = tokenizer(
			formatted_text, 
			truncation=True, 
			max_length=max_seq_length, 
			padding="max_length" if not padding_free else False, 
			return_tensors="pt",
			return_length=True
		)
	else:
		tokenized_inputs = tokenizer(
			formatted_text, 
			truncation=False, 
			return_tensors="pt",
			return_length=True
		)

	length = tokenized_inputs.get("length", None)

	return tokenized_inputs, length

def strip_jids_from_json(json_str):
	"""Remove jid keys from a JSON string (single object or full scene with objects list)."""
	parsed = json.loads(json_str)
	if isinstance(parsed, dict):
		parsed.pop("jid", None)
		if parsed.get("objects"):
			for obj in parsed["objects"]:
				obj.pop("jid", None)
	return json.dumps(parsed)

def process_scene_sample(orig_sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, do_augm=False, do_full_sg_outputs=False, do_keep_jids=False):
	while True:
		# Create instruction from scene
		sample = create_instruction_from_scene(orig_sample, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, do_keep_jids=True, do_augm=do_augm)

		# print(sample.get("split"))
		# print(orig_sample.get("split"))
	
		# Apply data augmentation if enabled
		# if sample.get("split") == "train" and do_augm:
		if orig_sample.get("split") == "train" and do_augm:
			# print("doing augm on sample with room type ", sample.get("room_type"), " and n_objects ", sample.get("n_objects"))
			sample_sg_input, sample_sg_output_add = do_random_augm_on_sgs(sample)
		else:
			# print("NO augm")
			sample_sg_input, sample_sg_output_add = sample["sg_input"], sample["sg_output_add"]

		# Store jid versions (post-augmentation) for scoring with GT assets
		sample["sg_input_with_jids"] = sample_sg_input
		sample["sg_output_add_with_jids"] = sample_sg_output_add

		# Strip jids for LLM-facing strings (unless caller explicitly wants jids)
		if not do_keep_jids:
			sg_input_for_llm = strip_jids_from_json(sample_sg_input)
			sg_output_add_for_llm = strip_jids_from_json(sample_sg_output_add)
		else:
			sg_input_for_llm = sample_sg_input
			sg_output_add_for_llm = sample_sg_output_add
		
		# Prepare the scene output/completion
		if do_full_sg_outputs:
			scene = json.loads(sg_input_for_llm)
			scene["objects"].append(json.loads(sg_output_add_for_llm))
			completion = json.dumps(scene)
		else:
			completion = sg_output_add_for_llm
		
		# Build the full instruction
		full_sample_instr = build_full_instruction_from_prompt(sample["prompt"], sg_input_for_llm)

		# check tok length and if it exceeds max length, retry
		_, tok_length = format_and_tokenize(tokenizer, full_sample_instr, completion, max_seq_length, padding_free=True, truncate=False)
		# subtract 150 tokens for the next object as the latter needs to fit into the context as well
		if tok_length <= (max_seq_length - 150):
			break
		else:
			print(f"sample exceeded max length ({tok_length} > {max_seq_length}-150), # of objects: {len(json.loads(sg_input_for_llm).get('objects'))}, retrying...")
	
	return full_sample_instr, completion, sample["prompt"], sample
		
def format_with_chat_template(tokenizer, prompt, completion=None):
	messages = [
		{"role": "system", "content": get_system_prompt_sgllm()},
		{"role": "user", "content": prompt}
	]
	if completion is not None:
		messages.append({"role": "assistant", "content": completion})
		
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(True if completion is None else False))

class RejSFTSceneDataCollator(DataCollatorForCompletionOnlyLM):
	"""Data collator for rejection sampling SFT. Takes pre-generated instruction/completion
	pairs (from the pickle file) and tokenizes them, using the same DataCollatorForCompletionOnlyLM
	base class as vanilla SFT for label masking.
	
	Also handles raw scene datasets for evaluation (same format as SFTSceneDataCollator)."""

	def __init__(self, response_template, tokenizer, max_seq_length, do_simple_descs=False, do_augm=False, do_full_sg_outputs=False, **kwargs):
		super().__init__(response_template=response_template, tokenizer=tokenizer, padding_free=True, mlm=False, **kwargs)
		self.tokenizer = tokenizer
		self.max_seq_length = max_seq_length
		self.do_simple_descs = do_simple_descs
		self.do_augm = do_augm
		self.do_full_sg_outputs = do_full_sg_outputs
		
		# Load metadata for processing raw scene samples (for eval)
		self.all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		self.all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

	def __call__(self, samples):
		batch_input_ids = []
		batch_attention_masks = []

		for sample in samples:
			# Check if this is a filtered rejection sample or raw scene sample
			if "instruction" in sample and "completion" in sample:
				# Filtered rejection sampling format
				instruction = sample["instruction"]
				completion = sample["completion"]
			else:
				# Raw scene format (for evaluation) - process like SFTSceneDataCollator
				instruction, completion, _, _ = process_scene_sample(
					sample, self.tokenizer, self.max_seq_length, 
					self.all_prompts, self.all_assets_metadata_simple_descs, 
					self.do_simple_descs, self.do_augm, self.do_full_sg_outputs
				)

			tok_inputs, tok_length = format_and_tokenize(
				self.tokenizer, instruction, completion, self.max_seq_length, padding_free=True
			)

			batch_input_ids.append(tok_inputs["input_ids"].squeeze(0))
			batch_attention_masks.append(tok_inputs["attention_mask"].squeeze(0))

		batch = [{"input_ids": input_ids, "attention_mask": attention_mask} for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks)]

		return super().__call__(batch)


class SFTSceneDataCollator(DataCollatorForCompletionOnlyLM):
	def __init__(self, do_augm, response_template, tokenizer, padding_free, max_seq_length, do_simple_descs, do_full_sg_outputs, **kwargs):
		super().__init__(response_template=response_template, tokenizer=tokenizer, padding_free=padding_free, mlm=False, **kwargs)
		
		self.tokenizer = tokenizer
		self.max_seq_length = max_seq_length
		self.padding_free = padding_free
		self.do_augm = do_augm
		self.do_simple_descs = do_simple_descs
		self.do_full_sg_outputs = do_full_sg_outputs
		self.all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		self.all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

	def __call__(self, samples):
		batch_input_ids = []
		batch_attention_masks = []

		# print room type and number of object for each sample
		# for sample in samples:
			# print(f"room_type: {sample['room_type']}, n_objects: {sample['n_objects']}")

		# make histogram with n_objects and count for each bin
		# plt.clf()
		# n_objects = [sample["n_objects"] for sample in samples]
		# plt.hist(n_objects, bins=range(0, max(n_objects), 1))
		# plt.xlabel('Number of Objects')
		# plt.ylabel('Count')
		# plt.title('Distribution of Number of Objects in Scenes')
		# plt.savefig("batch_n_objects_histogram_0.1.png")
		# exit()
		
		for idx, sample in enumerate(samples):

			full_sample_instr, sample_sg_output_full, _, _ = process_scene_sample(sample, self.tokenizer, self.max_seq_length, self.all_prompts, self.all_assets_metadata_simple_descs, self.do_simple_descs, self.do_augm, self.do_full_sg_outputs)

			tok_inputs, tok_length = format_and_tokenize(self.tokenizer, full_sample_instr, sample_sg_output_full, self.max_seq_length, self.padding_free)

			if tok_length is not None and tok_length > self.max_seq_length:
				print(f"Input was truncated. Original length: {tok_length}, Max length: {self.max_seq_length}")

			batch_input_ids.append(tok_inputs["input_ids"].squeeze(0))
			batch_attention_masks.append(tok_inputs["attention_mask"].squeeze(0))

		batch = [{"input_ids": input_ids, "attention_mask": attention_mask} for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks)]

		return super().__call__(batch)
	
def count_samples_exceeding_max_length(dataset, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False, do_full_sg_outputs=False):
	for attempt in range(10):
		count_exceeding = 0

		for sample in tqdm(dataset, desc="Counting samples exceeding max length"):

			full_sample_instr, sample_sg_output_full, _, _ = process_scene_sample(sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, False, do_full_sg_outputs)

		print("\n\n")
		
			# tok_inputs, tok_length = format_and_tokenize(tokenizer, full_sample_instr, sample_sg_output_full, max_seq_length, False, truncate=False)
			
			# Format with BOTH input and output included
			# formatted_text = format_with_chat_template(tokenizer, full_sample_instr, sample_sg_output_full)
			
			# Tokenize the COMPLETE text (input + output)
			# tokenized = tokenizer(formatted_text, truncation=False)
			
		# 	if tok_length > max_seq_length:
		# 		print("exceeding", tok_length)
		# 		print("number of corners: ", len(sample["scene"].get("bounds_top")))
		# 		print("n_objects: ", sample["n_objects"])
		# 		print("")
		# 		count_exceeding += 1

		# print("total exceeding samples: ", count_exceeding)
		# print("")

def count_samples_testset_seeds_exceeding_max_length(dataset, tokenizer, max_seq_length, all_test_instrs, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False, do_full_sg_outputs=False):
	for sample in tqdm(dataset, desc="Counting samples exceeding max length"):

		# full_sample_instr, sample_sg_output_full, _, _ = process_scene_sample(sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, False, do_full_sg_outputs)
		
		for seed in [1234, 3456, 5678]:
			instr_sample = all_test_instrs.get(sample.get("pth_orig_file"))[seed]

			full_instruction = build_full_instruction_from_prompt(instr_sample.get("prompt"), instr_sample.get("sg_input"))

			tok_inputs, tok_length = format_and_tokenize(tokenizer, full_instruction, instr_sample.get("sg_output_add"), max_seq_length, False, truncate=False)
			
			if tok_length > max_seq_length:
				print("exceeding", tok_length)
				print("number of corners: ", len(sample["scene"].get("bounds_top")))
				print("n_objects: ", sample["n_objects"])
				print("")

def get_random_sample(dataset, idx=None):
	if idx is None:
		idx = np.random.choice(len(dataset))
		print("choosing random sample with idx:", idx)
	return dataset.select([idx])[0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_obj_for_llm(obj: dict) -> dict:
	"""Strip jid, uuid, sampled_*, prompt keys for LLM-facing representation."""
	return {
		k: v for k, v in obj.items()
		if k not in ("jid", "uuid")
		and not k.startswith("sampled_")
		and not k.startswith("prompt")
	}

def _unique_desc_jids(objects_by_jid: dict) -> list:
	"""
	Return jids whose desc appears exactly once in the current object set.
	Only these are safe removal targets (guarantees remove_expected_delta == 1).
	"""
	desc_counts = Counter(o["desc"] for o in objects_by_jid.values())
	return [jid for jid, o in objects_by_jid.items() if desc_counts[o["desc"]] == 1]


def _sample_prompt_for_jid(jid: str, all_prompts: dict, rng_random: random.Random) -> str:
	"""Sample one prompt string from the prompt bank for a given jid."""
	jid_clean = jid.split("-(")[0]   # strip scale suffix e.g. "abc-(0.5)-(1.0)"
	return rng_random.choice(all_prompts[jid_clean])


def _check_scene_capacity(full_objects: list, partial_objects: list,
						   L: int, p_add: float) -> bool:
	"""
	Verify that (full, partial) can support a sequence of length L.

	Constraints:
	  L_add = ceil(L * p_add)
	  L_rem = L - L_add
	  (a) len(full) - len(partial) >= L_add
	  (b) n_unique_desc_objects(partial) >= L_rem
	"""
	L_add = math.ceil(L * p_add)
	L_rem = L - L_add

	if len(full_objects) - len(partial_objects) < L_add:
		return False

	if L_rem > 0:
		by_jid = {o["jid"]: o for o in partial_objects}
		if len(_unique_desc_jids(by_jid)) < L_rem:
			return False

	return True


def _build_op_sequence(
	partial_objects: list,
	full_objects: list,
	L: int,
	p_add: float,
	all_prompts: dict,
	rng: np.random.Generator,
	rng_random: random.Random,
) -> list:
	"""
	Build exactly L steps.  Returns list of step dicts with sg_input=None
	(filled by caller via replay).

	Removal uniqueness rule enforced at every step.
	"""
	full_by_jid    = {o["jid"]: o for o in full_objects}
	current_by_jid = {o["jid"]: deepcopy(o) for o in partial_objects}

	steps = []
	for step_idx in range(L):
		addable_jids   = [jid for jid in full_by_jid if jid not in current_by_jid]
		removable_jids = _unique_desc_jids(current_by_jid)

		can_add    = len(addable_jids) > 0
		can_remove = len(removable_jids) > 0

		if not can_add and not can_remove:
			break  # degenerate, capacity check should have prevented this

		if can_add and can_remove:
			op = "add" if rng.random() < p_add else "remove"
		elif can_add:
			op = "add"
		else:
			op = "remove"

		step = {
			"step_idx":              step_idx,
			"op":                    op,
			"add_prompt":            None,
			"add_obj_gt":            None,
			"remove_prompt":         None,
			"remove_obj_gt":         None,
			"remove_expected_delta": None,
		}

		if op == "add":
			chosen_jid = rng_random.choice(addable_jids)
			target_obj = deepcopy(full_by_jid[chosen_jid])
			step["add_prompt"]  = _sample_prompt_for_jid(chosen_jid, all_prompts, rng_random)
			step["add_obj_gt"]  = target_obj
			current_by_jid[chosen_jid] = target_obj

		else:
			chosen_jid = rng_random.choice(removable_jids)
			target_obj = current_by_jid[chosen_jid]
			step["remove_prompt"]         = _sample_prompt_for_jid(chosen_jid, all_prompts, rng_random)
			step["remove_obj_gt"]         = {"desc": target_obj["desc"], "jid": chosen_jid}
			step["remove_expected_delta"] = 1
			del current_by_jid[chosen_jid]

		steps.append(step)

	return steps

def create_sequential_editing_dataset(
	room_type: str = "all",
	seeds: list = None,
	n_per_bin: int = 20,
	max_seq_length: int = 10,
	p_add: float = 0.8,
	use_cached_dataset: bool = True,
	pth_cache_dir: str = None,
) -> dict:
	"""
	Generate (or load from cache) the sequential editing dataset.

	Args:
		all_prompts:        jid → list[str], loaded from PTH_ASSETS_METADATA_PROMPTS
		room_type:          "all" | "bedroom" | "livingroom"
		seeds:              random seeds (default [1234, 3456, 5678])
		n_per_bin:          samples per sequence-length bin (default 20)
							total per seed = n_per_bin * max_seq_length
		max_seq_length:     max instructions per sequence (default 10)
		p_add:              P(add | both ops feasible) (default 0.8)
		use_cached_dataset: if True and cache file exists, load and return it
		pth_cache_dir:      cache directory (falls back to PTH_DATASET_CACHE env)

	Returns:
		dict  {seed (int): [sample_dict, ...]}
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

	if use_cached_dataset and os.path.exists(pth_cache):
		print(f"[seq_dataset] Loading cached dataset from {pth_cache}")
		with open(pth_cache, "rb") as f:
			return pickle.load(f)

	print(
		f"[seq_dataset] Generating  room_type={room_type}  "
		f"n_per_bin={n_per_bin}  max_seq={max_seq_length}  p_add={p_add}"
	)

	# --- Load test scenes ---
	pth_root      = os.getenv("PTH_STAGE_2_DEDUP")
	all_test_pths = get_pths_dataset_split(room_type, "test")

	print(f"[seq_dataset] Pre-loading {len(all_test_pths)} test scenes …")
	all_scenes = {}
	for pth in tqdm(all_test_pths, desc="Loading scenes"):
		with open(os.path.join(pth_root, pth), "r", encoding="utf-8") as f:
			all_scenes[pth] = json.load(f)

	all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))

	# --- Index eligible scenes per bin ---
	# For bin L:
	#   L_add = ceil(L * p_add),  L_rem = L - L_add,  n_keep = max(1, L_rem)
	# A scene is eligible if it has enough distinct descs to build a valid partial
	# and enough remaining objects to cover L_add additions.
	print("[seq_dataset] Indexing scene capacity per bin …")
	bin_eligible = {L: [] for L in range(1, max_seq_length + 1)}

	for pth, scene in all_scenes.items():
		full_objects = scene.get("objects", [])
		n_full       = len(full_objects)
		all_descs    = [o["desc"] for o in full_objects]
		n_distinct   = len(set(all_descs))

		for L in range(1, max_seq_length + 1):
			L_add  = math.ceil(L * p_add)
			L_rem  = L - L_add
			n_keep = max(1, L_rem)

			if n_full - n_keep < L_add:
				continue
			if n_distinct < n_keep:
				# Can't form a partial with n_keep unique-desc objects
				continue
			bin_eligible[L].append(pth)

	for L in range(1, max_seq_length + 1):
		print(f"  bin L={L:2d}: {len(bin_eligible[L]):4d} eligible scenes")

	# --- Generate per seed ---
	dataset = {}

	for seed in seeds:
		print(f"\n[seq_dataset] seed={seed} …")
		rng        = np.random.default_rng(seed)
		rng_random = random.Random(seed)

		samples    = []
		bin_counts = {L: 0 for L in range(1, max_seq_length + 1)}

		# Shuffle bin processing order to avoid systematic bias in scene reuse
		bin_order = list(range(1, max_seq_length + 1))
		rng.shuffle(bin_order)

		for L in bin_order:
			eligible = list(bin_eligible[L])
			if not eligible:
				print(f"  WARNING: bin L={L} has no eligible scenes, skipping")
				continue

			rng.shuffle(eligible)
			pool_idx    = 0
			attempts    = 0
			max_attempts = n_per_bin * 50

			with tqdm(total=n_per_bin, desc=f"seed={seed} L={L}", leave=False) as pbar:
				while bin_counts[L] < n_per_bin and attempts < max_attempts:
					attempts += 1

					if pool_idx >= len(eligible):
						pool_idx = 0
						rng.shuffle(eligible)

					pth          = eligible[pool_idx]
					pool_idx    += 1
					scene        = all_scenes[pth]
					full_objects = scene.get("objects", [])

					L_add  = math.ceil(L * p_add)
					L_rem  = L - L_add
					n_keep = max(1, L_rem)

					# Build partial: pick n_keep objects with all unique descs
					indices = list(range(len(full_objects)))
					rng.shuffle(indices)

					seen_descs   = set()
					kept_indices = []
					for i in indices:
						if len(kept_indices) == n_keep:
							break
						desc = full_objects[i]["desc"]
						if desc not in seen_descs:
							kept_indices.append(i)
							seen_descs.add(desc)

					if len(kept_indices) < n_keep:
						continue  # not enough unique-desc objects in this scene

					partial_objects = [deepcopy(full_objects[i]) for i in kept_indices]

					if not _check_scene_capacity(full_objects, partial_objects, L, p_add):
						continue

					# Build step sequence
					steps = _build_op_sequence(
						partial_objects=partial_objects,
						full_objects=full_objects,
						L=L,
						p_add=p_add,
						all_prompts=all_prompts,
						rng=rng,
						rng_random=rng_random,
					)

					if len(steps) < L:
						continue  # sequence was truncated

					# Fill frozen sg_inputs via replay
					# _fill_sg_inputs(steps, scene, partial_objects, full_objects)

					sample = {
						"pth_orig_file":     pth,
						"seed":              seed,
						"room_type":         scene["room_type"],
						"seq_length":        L,
						"n_objects_full":    len(full_objects),
						"n_objects_partial": len(partial_objects),
						"partial_scene": {
							"room_type":     scene["room_type"],
							"bounds_top":    scene["bounds_top"],
							"bounds_bottom": scene["bounds_bottom"],
							"objects":       [deepcopy(o) for o in partial_objects],
						},
						"steps": steps,
					}

					samples.append(sample)
					bin_counts[L] += 1
					pbar.update(1)

			if bin_counts[L] < n_per_bin:
				print(f"  WARNING: bin L={L} only filled {bin_counts[L]}/{n_per_bin}")

		# sort by increasing seq_length
		# samples.sort(key=lambda s: s["seq_length"])

		# randomly shuffle by seq_length to avoid any bias from processing order
		rng.shuffle(samples)

		dataset[seed] = samples

		# Summary
		n_adds    = sum(sum(1 for st in s["steps"] if st["op"] == "add")    for s in samples)
		n_removes = sum(sum(1 for st in s["steps"] if st["op"] == "remove") for s in samples)
		total_ops = n_adds + n_removes
		print(
			f"  seed={seed}: {len(samples)} samples  "
			f"adds={n_adds}  removes={n_removes}  "
			f"add_ratio={n_adds/total_ops:.2f}  "
			f"bins={ {L: bin_counts[L] for L in range(1, max_seq_length+1)} }"
		)

	# Save
	os.makedirs(pth_cache_dir, exist_ok=True)
	with open(pth_cache, "wb") as f:
		pickle.dump(dataset, f)
	print(f"\n[seq_dataset] Saved to {pth_cache}")

	return dataset


if __name__ == "__main__":
	from dotenv import load_dotenv
	load_dotenv(".env.stanley")

	dataset = create_sequential_editing_dataset(room_type="all", seeds=[1234, 3456, 5678], n_per_bin=20, max_seq_length=10, p_add=0.8, use_cached_dataset=False)

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