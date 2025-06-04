# custom_generate_results.py for mi-diff

import pdb

import argparse
import os
import sys
import json
import uuid
import numpy as np
import torch
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
import trimesh
import copy

from scripts.utils import PROJ_DIR, load_config, update_data_file_paths
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from midiffusion.datasets.threed_front_encoding import get_dataset_raw_and_encoded
from midiffusion.networks import build_network
from midiffusion.evaluation.utils import generate_layouts, get_feature_mask
from threed_front.datasets.threed_future_dataset import ThreedFutureDataset

pth_base = "../../../.."
sys.path.append(pth_base)
from src.sample import AssetRetrievalModule
from src.utils import set_seeds, create_category_lookup, get_test_instrs_all, get_pths_dataset_split

def create_object_entry(class_label, pos, size, angle, furniture, all_assets_metadata=None):
	# Convert angle to quaternion (y-axis rotation)
	rot = [0.0, np.sin(angle/2), 0.0, np.cos(angle/2)]
	
	try:
		raw_mesh = trimesh.load(furniture.raw_model_path, force="mesh", ignore_materials=True, process=False)
	except Exception as e:
		raw_mesh = trimesh.load(furniture.raw_model_path)

	raw_mesh.vertices *= furniture.scale
	real_bbox = raw_mesh.bounds
	real_size = (real_bbox[1] - real_bbox[0]).tolist()
	
	# Adjust position based on real size
	pos_adjusted = pos.copy() if isinstance(pos, list) else pos.tolist()
	pos_adjusted[1] = pos_adjusted[1] - (real_size[1]/2)
	
	# Create object entry
	obj_entry = {
		"desc": class_label,
		"size": real_size,
		"query_size": size,
		"pos": pos_adjusted,
		"rot": rot,
		"scale": furniture.scale,
		"sampled_asset_jid": furniture.model_jid,
		"sampled_asset_desc": all_assets_metadata.get(furniture.model_jid).get("summary"),
		"sampled_asset_size": real_size,
		"uuid": str(uuid.uuid4())
	}
	
	return obj_entry

def get_scene_both_formats(pth_base, pth_scene, test_dataset, raw_test_dataset, feature_extractor_name, dvc):
	
	with open(os.path.join(pth_base, os.getenv("PTH_STAGE_2_DEDUP"), pth_scene), "r", encoding='utf-8') as f:
		scene_ours = json.loads(f.read())
		room_id_full = scene_ours.get("room_id", "") + "-" + "-".join(pth_scene.split(".")[0].split("-")[:5])

	# 'LivingRoom-157488-ca96647a-15ef-4d0c-9b27-02aa414bcaef'

	scene_midiff_raw = None
	for i, sample_raw_test_dataset in enumerate(raw_test_dataset):
		if sample_raw_test_dataset.scene_id == room_id_full:
			scene_midiff_raw = sample_raw_test_dataset
			break

	scene_midiff = None
	for i, sample_test_dataset in enumerate(test_dataset):
		if sample_test_dataset["scene_id"] == room_id_full:
			scene_midiff = sample_test_dataset
			break

	if scene_midiff is None or scene_midiff_raw is None:
		print("Scene not found in test dataset:", room_id_full)
		idx = np.random.randint(0, len(test_dataset))
		scene_midiff = test_dataset[idx]
		scene_midiff_raw = raw_test_dataset[idx]

	if feature_extractor_name == "resnet18":
		room_feature = torch.from_numpy(scene_midiff["room_layout"]).to(dvc).unsqueeze(0)
	elif feature_extractor_name == "pointnet_simple":
		room_feature = torch.from_numpy(scene_midiff["fpbpn"]).to(dvc).unsqueeze(0)

	return scene_midiff_raw, scene_ours, room_feature

def generate_full_scene(bon_per_sample, room_type, bounds_top, bounds_bottom, pth_scene, test_dataset, raw_test_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, rand_seed):
	# print("gen full scene")

	feature_extractor_name = args.config["feature_extractor"]["name"]
	_, _, room_feature = get_scene_both_formats(pth_base, pth_scene, test_dataset, raw_test_dataset, feature_extractor_name, dvc)
	
	# Get feature mask for the specified experiment
	feature_mask = get_feature_mask(network, args.experiment, args.n_known_objects, dvc)
	
	with torch.no_grad():
		bbox_params_dict = network.generate_layout(
			room_feature=room_feature,
			batch_size=1,
			input_boxes=None,
			feature_mask=feature_mask,
			device=dvc,
		)[0]
	
	# Post-process to get the layout
	boxes = test_dataset.post_process(bbox_params_dict)
	
	# Convert to our JSON format
	objects = convert_full_layout_to_json_format(boxes, test_dataset.class_labels, objects_dataset, all_assets_metadata)
	
	# Create the final scene
	final_scene = {
		"room_type": room_type,
		"bounds_top": bounds_top,
		"bounds_bottom": bounds_bottom,
		"objects": objects
	}
	
	return final_scene

def convert_full_layout_to_json_format(layout, class_labels, objects_dataset, all_assets_metadata):
	objects = []
	
	# Process the bbox_params from the layout
	class_labels_array = layout["class_labels"].squeeze(0)
	translations = layout["translations"].squeeze(0)
	sizes = layout["sizes"].squeeze(0)
	angles = layout["angles"].squeeze(0)
	
	# Process each object
	for obj_idx in range(len(class_labels_array)):
		class_idx = np.argmax(class_labels_array[obj_idx])
		if class_idx >= len(class_labels) - 1:  # Skip the "end" token if present
			print("Skipping object with class index:", class_idx)
			continue
			
		class_label = class_labels[class_idx]
		pos = translations[obj_idx].tolist()
		size = sizes[obj_idx].tolist()
		angle = angles[obj_idx, 0].item()
		
		# Find matching furniture from dataset for additional properties
		furniture = objects_dataset.get_closest_furniture_to_box(class_label, size)
		
		if furniture:
			# Create object entry using helper function
			obj_entry = create_object_entry(class_label, pos, size, angle, furniture, all_assets_metadata)
			objects.append(obj_entry)
	
	return objects

def convert_layout_to_single_json_object(layout, class_labels, objects_dataset, all_assets_metadata, object_idx):
	# Get the class label
	class_label_array = layout["class_labels"].squeeze(0)
	class_idx = np.argmax(class_label_array[object_idx])
	
	if class_idx >= len(class_labels) - 1:  # Skip the "end" token
		print("Skipping object with class index:", class_idx)
		return None
		
	class_label = class_labels[class_idx]
	
	# Get position, size, and angle
	pos = layout["translations"].squeeze(0)[object_idx].tolist()
	size = layout["sizes"].squeeze(0)[object_idx].tolist()
	
	# For MiDiffusion, the angle might be represented as 2 values (sin and cos)
	# Check if angles has shape [N, 2] or [N, 1]
	# if layout["angles"].shape[-1] == 2:
	# 	sin_val = layout["angles"].squeeze(0)[object_idx, 0].item()
	# 	cos_val = layout["angles"].squeeze(0)[object_idx, 1].item()
	# 	angle = np.arctan2(sin_val, cos_val)
	# else:
	angle = layout["angles"].squeeze(0)[object_idx, 0].item()
	
	# Find matching furniture from dataset
	furniture = objects_dataset.get_closest_furniture_to_box(class_label, size)
	
	# Create object entry using helper function
	return create_object_entry(class_label, pos, size, angle, furniture, all_assets_metadata)

def convert_our_objects_to_midiff_format(objects, desc_to_category, test_dataset, network, num_existing_objects, class_idx, dvc):
	existing_objects = []
	for obj in objects:
		obj_category = desc_to_category[obj["desc"]]
		obj_class_idx = None
		for idx, cls in enumerate(test_dataset.class_labels):
			if cls.lower() == obj_category:
				obj_class_idx = idx
				break
		
		if obj_class_idx is not None:
			# Create one-hot encoding for the class
			class_label = np.zeros(len(test_dataset.class_labels))
			class_label[obj_class_idx] = 1
			
			# Get position, size and angle
			translation = obj["pos"]
			size = obj["size"]
			
			# Extract angle from quaternion
			quat = obj["rot"]
			angle = 2 * np.arctan2(quat[1], quat[3])  # Extract from y and w components
			
			existing_objects.append({
				"class_labels": class_label,
				"translations": translation,
				"sizes": size,
				"angles": [angle]  # Assuming angle_dim = 1
			})
		
	# Create input_boxes tensor
	max_num_points = network.sample_num_points
	point_dim = network.point_dim + network.class_dim
	input_boxes = torch.zeros((1, max_num_points, point_dim), device=dvc)
	
	# Fill in existing objects
	for i, obj in enumerate(existing_objects):
		# Fill class one-hot
		class_start = network.bbox_dim
		class_end = class_start + network.class_dim
		input_boxes[0, i, class_start:class_end] = torch.tensor(obj["class_labels"], device=dvc)
		
		# Fill translation
		input_boxes[0, i, 0:network.translation_dim] = torch.tensor(obj["translations"], device=dvc)
		
		# Fill size
		size_start = network.translation_dim
		size_end = size_start + network.size_dim
		input_boxes[0, i, size_start:size_end] = torch.tensor(obj["sizes"], device=dvc)
		
		# Fill angle
		angle_start = size_end
		angle_end = angle_start + network.angle_dim
		
		if network.angle_dim == 1:
			input_boxes[0, i, angle_start:angle_end] = torch.tensor([obj["angles"][0]], device=dvc)
		else:  # network.angle_dim == 2
			# Convert angle to sin/cos representation
			angle = obj["angles"][0]
			input_boxes[0, i, angle_start:angle_end] = torch.tensor([np.sin(angle), np.cos(angle)], device=dvc)
	
	# Create feature mask - True means fixed, False means to be generated
	feature_mask = torch.zeros((max_num_points, point_dim), dtype=torch.bool, device=dvc)
	
	# Fix all existing objects
	feature_mask[:num_existing_objects, :] = True
	
	# For the new object we want to add, fix only the class
	feature_mask[num_existing_objects, class_start:class_end] = True

	# remaining rows we disable as well
	feature_mask[num_existing_objects+1:, :] = True
	for i in range(num_existing_objects+1, max_num_points):
		empty_class_idx = network.class_dim - 1  # Assuming last class is "empty"
		empty_one_hot = torch.zeros(network.class_dim, device=dvc)
		empty_one_hot[empty_class_idx] = 1
		input_boxes[0, i, class_start:class_end] = empty_one_hot
	
	# Create one-hot for the class we want to add
	class_one_hot = torch.zeros(network.class_dim, device=dvc)
	class_one_hot[class_idx] = 1
	
	# Set the class for the new object
	input_boxes[0, num_existing_objects, class_start:class_end] = class_one_hot

	return feature_mask, input_boxes

def generate_instr_scene(bon_per_sample, rand_seed, pth_scene, all_test_instrs, test_dataset, raw_test_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, pth_base, all_prompts, desc_to_category, sampling_engine):
	# print("gen instr scene")

	feature_extractor_name = args.config["feature_extractor"]["name"]
	scene_midiff_raw, scene_ours, room_feature = get_scene_both_formats(pth_base, pth_scene, test_dataset, raw_test_dataset, feature_extractor_name, dvc)

	instr_sample = all_test_instrs.get(pth_scene)[rand_seed]
	
	scene_query = json.loads(instr_sample["sg_input"])
	obj_to_add = json.loads(instr_sample["sg_output_add"])

	final_scene = copy.deepcopy(scene_query)
	final_scene = sampling_engine.sample_all_assets(final_scene, is_greedy_sampling=True)
	
	# Get the category of the object to add
	category_to_add = desc_to_category[obj_to_add["desc"]]
	class_idx = None
	for idx, cls in enumerate(test_dataset.class_labels):
		if cls.lower() == category_to_add:
			class_idx = idx
			break
	if class_idx is None:
		print("couldn't find class_idx for category_to_add!", category_to_add, obj_to_add["desc"])

	num_existing_objects = len(scene_query["objects"])
		
	feature_mask = None
	input_boxes = None
	if num_existing_objects > 0:
		feature_mask, input_boxes = convert_our_objects_to_midiff_format(scene_query["objects"], desc_to_category, test_dataset, network, num_existing_objects, class_idx, dvc)
	else:
		max_num_points = network.sample_num_points
		point_dim = network.point_dim + network.class_dim
		input_boxes = torch.zeros((1, max_num_points, point_dim), device=dvc)
		
		feature_mask = torch.zeros((max_num_points, point_dim), dtype=torch.bool, device=dvc)

		# Set the first object (index 0) class to the one we want to add
		class_start = network.bbox_dim
		class_end = class_start + network.class_dim
		class_one_hot = torch.zeros(network.class_dim, device=dvc)
		class_one_hot[class_idx] = 1
		input_boxes[0, 0, class_start:class_end] = class_one_hot

		# Fix only the class for the first object, let the network generate position/size/angle
		feature_mask[0, class_start:class_end] = True

		# Set remaining objects to "empty" class
		for i in range(1, max_num_points):
			empty_class_idx = network.class_dim - 1  # Assuming last class is "empty"
			empty_one_hot = torch.zeros(network.class_dim, device=dvc)
			empty_one_hot[empty_class_idx] = 1
			input_boxes[0, i, class_start:class_end] = empty_one_hot

		# Disable all other objects (index > 0)
		feature_mask[1:, :] = True

	# Generate layout 
	with torch.no_grad():
		bbox_params_dict = network.generate_layout(
			room_feature=room_feature,
			batch_size=1,
			input_boxes=input_boxes,
			feature_mask=feature_mask,
			device=dvc,
		)[0]

	boxes = test_dataset.post_process(bbox_params_dict)
		
	# Convert to our JSON format - we only want the newly added object
	if num_existing_objects < len(boxes["class_labels"][0]):
		added_object = convert_layout_to_single_json_object(
			boxes, 
			test_dataset.class_labels, 
			objects_dataset, 
			all_assets_metadata,
			num_existing_objects
		)
		added_object["prompt"] = instr_sample["prompt"]
		final_scene["objects"].append(added_object)
	else:
		print(f"Warning: No new object was added to the scene")

	# Add other scene information
	bounds_top = scene_ours.get("bounds_top")
	bounds_bottom = scene_ours.get("bounds_bottom")
	room_type = scene_ours.get("room_type")
	
	final_scene["room_type"] = room_type
	final_scene["bounds_top"] = bounds_top
	final_scene["bounds_bottom"] = bounds_bottom
	
	return final_scene
		

def main(argv):
	parser = argparse.ArgumentParser(
		description="Generate scenes using a previously trained MiDiffusion model and save in JSON format"
	)

	parser.add_argument(
		"weight_file",
		help="Path to a pretrained model"
	)
	parser.add_argument(
		"--config_file",
		default=None,
		help="Path to the file that contains the experiment configuration"
		"(default: config.yaml in the model directory)"
	)
	parser.add_argument(
		"--output_directory",
		default=PROJ_DIR+"/output/predicted_results/",
		help="Path to the output directory"
	)
	parser.add_argument(
		"--n_known_objects",
		default=0,
		type=int,
		help="Number of existing objects for scene completion task"
	)
	parser.add_argument(
		"--experiment",
		default="synthesis",
		choices=[
			"synthesis",
			"scene_completion",
			"furniture_arrangement",
			"object_conditioned",
			"scene_completion_conditioned"
		],
		help="Experiment name"
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Seed for the sampling floor plan"
	)
	parser.add_argument(
		"--n_syn_scenes",
		default=1,
		type=int,
		help="Number of scenes to be synthesized"
	)
	parser.add_argument(
		"--batch_size",
		default=8,
		type=int,
		help="Number of synthesized scene in each batch"
	)
	parser.add_argument(
		"--result_tag",
		default=None,
		help="Result sub-directory name"
	)
	parser.add_argument(
		"--path_to_pickled_3d_future_models",
		help="Path to the 3D-FUTURE model meshes",
		required=True
	)
	parser.add_argument(
		"--gpu",
		type=int,
		default=0,
		help="GPU ID"
	)
	parser.add_argument(
		"--n-test-scenes",
		type=int,
		default=10,
		help="Number of test scenes to process"
	)
	parser.add_argument(
		"--bon-per-sample",
		type=int,
		default=1,
		help="Number of BON per sample"
	)
	parser.add_argument(
		"--room-type", 
		type=str, 
		default="bedroom",
		help="Room type for generation"
	)
	parser.add_argument(
		"--do-full-scenes",
		action="store_true",
		help="Generate full scenes (instead of scene completion)"
	)

	args = parser.parse_args(argv)
	dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	rand_seed=1234
	set_seeds(rand_seed)
	load_dotenv(f"{pth_base}/.env.stanley")

	output_dir = os.path.join(pth_base, args.output_directory)

	all_prompts = json.load(open(os.path.join(pth_base, os.getenv("PTH_ASSETS_METADATA_PROMPTS"))))
	all_assets_metadata = json.load(open(os.path.join(pth_base, os.getenv("PTH_ASSETS_METADATA"))))
	all_assets_metadata_orig = json.load(open(os.path.join(pth_base, os.path.join(os.getenv("PTH_3DFUTURE_ASSETS"), "model_info.json"))))
	desc_to_category = create_category_lookup(all_assets_metadata_orig, all_assets_metadata)
	sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=rand_seed, accelerator=None, dvc=dvc, do_print=False)

	# Parse the config file
	if args.config_file is None:
		args.config_file = os.path.join(os.path.dirname(args.weight_file), "config.yaml")
	config = load_config(args.config_file)
	if "_eval" not in config["data"]["encoding_type"] and args.experiment == "synthesis":
		config["data"]["encoding_type"] += "_eval"
	args.config = config

	# Raw training data (for record keeping)
	raw_train_dataset = get_raw_dataset(
		update_data_file_paths(config["data"]), 
		split=config["training"].get("splits", ["train", "val"]),
		include_room_mask=config["network"].get("room_mask_condition", True)
	) 

	# Get Scaled dataset encoding (without data augmentation)
	raw_test_dataset, test_dataset = get_dataset_raw_and_encoded(
		update_data_file_paths(config["data"]),
		split=config["validation"].get("splits", ["test"]),
		max_length=config["network"]["sample_num_points"],
		include_room_mask=config["network"].get("room_mask_condition", True)
	)
	
	# Load 3D-FUTURE furniture models
	objects_dataset = ThreedFutureDataset.from_pickled_dataset(args.path_to_pickled_3d_future_models)

	# Build network with saved weights
	network, _, _ = build_network(test_dataset.n_object_types, config, args.weight_file, device=dvc)
	network.eval()

	# test set split
	all_pths = get_pths_dataset_split(args.room_type, "test", prefix=pth_base)
	
	all_pths = all_pths[:args.n_test_scenes]
	
	all_test_instrs = get_test_instrs_all(args.room_type)
	rand_seeds = [1234, 3456, 5678]
	
	# generate scenes
	for rand_seed in rand_seeds:
		set_seeds(rand_seed)

		for idx, pth_scene in tqdm(enumerate(all_pths)):
		
			# get bounds from each floor plan for json later
			with open(os.path.join(pth_base, os.getenv("PTH_STAGE_2_DEDUP"), pth_scene), "r", encoding='utf-8') as f:
				scene = json.loads(f.read())
			bounds_top = scene.get("bounds_top")
			bounds_bottom = scene.get("bounds_bottom")
			room_type = scene.get("room_type")
			
			if args.do_full_scenes:
				final_scene = generate_full_scene(args.bon_per_sample, room_type, bounds_top, bounds_bottom, pth_scene, test_dataset, raw_test_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, rand_seed)
			else:
				final_scene = generate_instr_scene(args.bon_per_sample, rand_seed, pth_scene, all_test_instrs, test_dataset, raw_test_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, pth_base, all_prompts, desc_to_category, sampling_engine)

			# Save JSON file
			output_file = os.path.join(output_dir, str(rand_seed), f"{idx}_{rand_seed}.json")
			with open(output_file, 'w') as f:
				json.dump(final_scene, f, indent=4)

if __name__ == "__main__":
	main(sys.argv[1:])