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

from src.utils import create_floor_plan_polygon, inherit_props_by_id, get_pths_dataset_split, remove_and_recreate_folder, get_system_prompt_sgllm

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
	
def create_instruction_from_scene(sample, all_prompts, all_assets_metadata_simple_descs=None, do_simple_descs=False, do_keep_jids=False):
	scene = sample["scene"]
	n_objects = sample["n_objects"]
	room_type = sample["room_type"]
	
	# get weight for instruction generation
	# n_zero_start = min(1 // n_objects, 0.1)
	# n_full_scene = min(1 // n_objects, 0.1)

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

def process_scene_sample(orig_sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, do_augm=False, do_full_sg_outputs=False, do_keep_jids=False):
	while True:
		# Create instruction from scene
		sample = create_instruction_from_scene(orig_sample, all_prompts, all_assets_metadata_simple_descs, do_simple_descs, do_keep_jids=do_keep_jids)
	
		# Apply data augmentation if enabled
		if sample.get("split") == "train" and do_augm:
			sample_sg_input, sample_sg_output_add = do_random_augm_on_sgs(sample)
		else:
			sample_sg_input, sample_sg_output_add = sample["sg_input"], sample["sg_output_add"]
		
		# Prepare the scene output/completion
		if do_full_sg_outputs:
			scene = json.loads(sample_sg_input)
			scene["objects"].append(json.loads(sample_sg_output_add))
			completion = json.dumps(scene)
		else:
			completion = sample_sg_output_add
		
		# Build the full instruction
		full_sample_instr = build_full_instruction_from_prompt(sample["prompt"], sample_sg_input)

		# check tok length and if it exceeds max length, retry
		_, tok_length = format_and_tokenize(tokenizer, full_sample_instr, completion, max_seq_length, padding_free=True, truncate=False)
		# subtract 150 tokens for the next object as the latter needs to fit into the context as well
		if tok_length <= (max_seq_length - 150):
			break
		else:
			print(f"sample exceeded max length ({tok_length} > {max_seq_length}-150), # of objects: {len(json.loads(sample_sg_input).get('objects'))}, retrying...")
	
	return full_sample_instr, completion, sample["prompt"], sample
		
def format_with_chat_template(tokenizer, prompt, completion=None):
	messages = [
		{"role": "system", "content": get_system_prompt_sgllm()},
		{"role": "user", "content": prompt}
	]
	if completion is not None:
		messages.append({"role": "assistant", "content": completion})
		
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(True if completion is None else False))

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