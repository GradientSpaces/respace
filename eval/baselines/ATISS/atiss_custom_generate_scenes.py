# custom_generate_scenes.py for ATISS

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime

import numpy as np
import torch
import pdb
import trimesh
from tqdm import tqdm
from dotenv import load_dotenv
import pickle
import traceback
from matplotlib import pyplot as plt
from pyrr import Matrix44
import random
import copy

# Import their utilities
from scripts.training_utils import load_config
from scripts.utils import floor_plan_from_scene, floor_plan_renderable, render
from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects

from simple_3dviz import Scene
from simple_3dviz.renderables import Mesh
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render as render_3dviz

pth_base = "../../../.."
sys.path.append(pth_base)
from src.sample import AssetRetrievalModule
from src.utils import set_seeds, create_category_lookup, get_test_instrs_all, get_pths_dataset_split

def render_vanilla(floor_plan, tr_floor, bbox_params_t, objects_dataset, classes, current_scene, scene_idx, args, output_dir):

	scene = Scene(size=args.window_size)
	scene.up_vector = args.up_vector
	scene.camera_target = args.camera_target
	scene.camera_position = args.camera_position
	scene.light = args.camera_position

	renderables, trimesh_meshes = get_textured_objects(
		bbox_params_t, objects_dataset, classes
	)
	renderables += floor_plan
	trimesh_meshes += tr_floor

	# Do the rendering
	path_to_image = "{}/{}_{}_{:03d}".format(
		output_dir,
		current_scene.scene_id,
		scene_idx,
		0,
	)
	behaviours = [
		LightToCamera(),
		SaveFrames(path_to_image+".png", 1)
	]
	
	behaviours += [
		CameraTrajectory(
			Circle(
				[0, args.camera_position[1], 0],
				args.camera_position,
				args.up_vector
			),
			speed=1/360
		),
		SaveGif(path_to_image+".gif", 1)
	]

	render_3dviz(
		renderables,
		behaviours=behaviours,
		size=args.window_size,
		camera_position=args.camera_position,
		camera_target=args.camera_target,
		up_vector=args.up_vector,
		background=args.background,
		n_frames=args.n_frames,
		scene=scene
	)

def convert_atiss_output_to_our_format(boxes, indices, classes, objects_dataset, all_assets_metadata):

	if isinstance(indices, int):
		indices = [indices]
		single_output = True
	else:
		single_output = False
	
	# Convert each requested object
	objects = []
	for i in indices:
		obj = convert_atiss_object_to_our_format(
			boxes["class_labels"][0, i],
			boxes["translations"][0, i],
			boxes["sizes"][0, i],
			boxes["angles"][0, i],
			classes,
			objects_dataset,
			all_assets_metadata
		)
		objects.append(obj)
	
	# Return either a list or a single object
	return objects[0] if single_output else objects

def get_scene_both_formats(pth_base, pth_scene, raw_dataset):
	with open(os.path.join(pth_base, os.getenv("PTH_STAGE_2_DEDUP"), pth_scene), "r", encoding='utf-8') as f:
		scene_ours = json.loads(f.read())
		room_id_full = scene_ours.get("room_id") + "-" + "-".join(pth_scene.split(".")[0].split("-")[:5])

	scene_atiss = None
	for sample in raw_dataset:
		if sample.scene_id == room_id_full:
			scene_atiss = sample
			break

	if scene_atiss is None:
		print("Scene not found in test dataset:", room_id_full)
		idx = np.random.randint(0, len(raw_dataset))
		scene_atiss = raw_dataset[idx]

	return scene_atiss, scene_ours

def convert_atiss_object_to_our_format(class_label_tensor, translation_tensor, size_tensor, angle_tensor, classes, objects_dataset, all_assets_metadata):

	# Get the class label
	class_label_idx = class_label_tensor.argmax(-1).item()
	class_label = classes[class_label_idx]
	
	# Get position, size, and angle
	translation = translation_tensor
	size = size_tensor
	angle = angle_tensor[0]
	
	# Get the matching furniture from the objects dataset
	furniture = objects_dataset.get_closest_furniture_to_box(class_label, size)

	# raw_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
	try:
		raw_mesh = trimesh.load(furniture.raw_model_path, force="mesh", ignore_materials=True, process=False)
	except Exception as e:
		print(e)
		print(f"error loading mesh: {furniture.raw_model_path}. loading with fallback...")
		try:
			print(e)
			raw_mesh = trimesh.load(furniture.raw_model_path)
		except Exception as e:
			print("loading glb file !!")
			raw_mesh = trimesh.load(furniture.raw_model_path.replace('.obj', '.glb'))

	# Check if it's a Scene object and extract the mesh if needed
	if isinstance(raw_mesh, trimesh.Scene):
		# Extract all meshes from the scene and combine them
		mesh_list = []
		for geometry_name, geometry in raw_mesh.geometry.items():
			if hasattr(geometry, 'vertices'):
				mesh_list.append(geometry)
		
		if mesh_list:
			# Combine all meshes if multiple exist
			if len(mesh_list) > 1:
				raw_mesh = trimesh.util.concatenate(mesh_list)
			else:
				raw_mesh = mesh_list[0]
		else:
			raise ValueError("No valid meshes found in the scene")

	raw_mesh.vertices *= furniture.scale
	real_bbox = raw_mesh.bounds
	real_size = (real_bbox[1] - real_bbox[0]).tolist()
	translation[1] = translation[1] - (real_size[1]/2)
	
	# Create the rotation quaternion from the angle
	rot = [0.0, np.sin(angle/2), 0.0, np.cos(angle/2)]
	
	# Construct the object in our format
	obj = {
		"desc": class_label,
		"size": real_size,
		"query_size": size.tolist(),
		"pos": translation.tolist(),
		"rot": rot,
		"scale": furniture.scale,
		"sampled_asset_jid": furniture.model_jid,
		"sampled_asset_desc": all_assets_metadata.get(furniture.model_jid).get("summary"),
		"sampled_asset_size": real_size,
		"uuid": str(uuid.uuid4())
	}
	
	return obj

def convert_our_objects_to_atiss_format(objects, classes, desc_to_category, dvc):
	n_objects = len(objects)
	n_classes = len(classes)
	
	# Initialize tensors
	class_labels = torch.zeros(1, n_objects, n_classes, device=dvc)
	translations = torch.zeros(1, n_objects, 3, device=dvc)
	sizes = torch.zeros(1, n_objects, 3, device=dvc)
	angles = torch.zeros(1, n_objects, 1, device=dvc)
	
	# Fill tensors with object data
	for i, obj in enumerate(objects):

		class_idx = match_class_label_to_idx(obj.get("desc"), classes, desc_to_category)
			
		class_labels[0, i, class_idx] = 1
		
		# Set position
		translations[0, i] = torch.tensor(obj.get("pos"), device=dvc)
		
		# Set size
		sizes[0, i] = torch.tensor(obj.get("size"), device=dvc)
		
		# Set angle (convert from quaternion to single angle)
		rot_quat = obj.get("rot")
		y_angle = quaternion_to_y_angle(rot_quat)
		angles[0, i, 0] = y_angle
	
	return {
		"class_labels": class_labels,
		"translations": translations,
		"sizes": sizes,
		"angles": angles
	}

def quaternion_to_y_angle(quaternion):
	"""Convert quaternion [x, y, z, w] to rotation angle around Y axis"""
	x, y, z, w = quaternion
	
	# Extract angle around Y axis (simplified for Y-axis rotation)
	# This is simplified for Y-axis rotations, assuming that's what we're dealing with
	siny_cosp = 2.0 * (w * y + x * z)
	cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
	
	return np.arctan2(siny_cosp, cosy_cosp)

def match_class_label_to_idx(desc, classes, desc_to_category):
	category = desc_to_category[desc]
	for i, cls in enumerate(classes):
		if cls.lower() == category:
			# print(f"Matched class label {cls} to idx {i} for category {category}")
			return i
		
	# print(f"--{category}--")
	# print([ f"--{cls}--" for cls in classes ])

	print("could not match class label to idx")

def generate_instr_scene(bon_per_sample, room_type, rand_seed, pth_scene, all_test_instrs, test_dataset, raw_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, pth_base, all_prompts, desc_to_category, sampling_engine):
	# sprint("gen instr scene")

	scene_atiss, scene_ours = get_scene_both_formats(pth_base, pth_scene, raw_dataset)

	floor_plan, tr_floor, room_mask = floor_plan_from_scene(scene_atiss, args.path_to_floor_plan_textures)
	classes = np.array(test_dataset.class_labels)

	instr_sample = all_test_instrs.get(pth_scene)[rand_seed]

	scene_query = json.loads(instr_sample["sg_input"])
	obj_to_add = json.loads(instr_sample["sg_output_add"])

	final_scene = copy.deepcopy(scene_query)
	final_scene = sampling_engine.sample_all_assets(final_scene, is_greedy_sampling=True)

	if scene_query["objects"]:
		atiss_boxes = convert_our_objects_to_atiss_format(scene_query["objects"], classes, desc_to_category, dvc)

	# get one-hot vector for class label
	class_idx = match_class_label_to_idx(obj_to_add["desc"], classes, desc_to_category)
	one_hot = torch.zeros(len(classes))
	one_hot[class_idx] = 1
	class_label = one_hot.unsqueeze(0).unsqueeze(0)

	# make forward pass
	bbox_params = network.add_object(
		room_mask=room_mask.to(dvc),
		class_label=class_label.to(dvc),
		boxes=atiss_boxes if scene_query["objects"] else None,
		device=dvc
	)

	# postprocess
	bbox_params = {k: v.cpu().numpy() for k, v in bbox_params.items()}
	boxes = test_dataset.post_process(bbox_params)
	
	# Add the object to our scene
	added_object = convert_atiss_output_to_our_format(boxes, -2, classes, objects_dataset, all_assets_metadata)
	added_object["prompt"] = instr_sample["prompt"]
	final_scene["objects"].append(added_object)

	return final_scene

def generate_full_scene(bon_per_sample, room_type, bounds_top, bounds_bottom, pth_scene, test_dataset, raw_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, pth_base):
	# print("gen full scene")

	# get random floor plan from their preprocessed dataset and extract bounds form it
	# scene_idx = np.random.choice(len(dataset))
	# current_scene = raw_dataset[scene_idx]
	# floor_plan, tr_floor, room_mask = floor_plan_from_scene(current_scene, args.path_to_floor_plan_textures)
	# vertices, faces = current_scene.floor_plan
	# vertices = vertices - current_scene.floor_plan_centroid
	# floor_vertices = extract_corners(vertices[:, [0, 2]]).tolist()
	# bounds_bottom = [[x, 0.0, z] for x, z in floor_vertices]
	# bounds_top = [[x, 2.6, z] for x, z in floor_vertices]

	# custom
	# vertices = np.array(bounds_bottom)
	# vertices = bounds_bottom
	# span = 6.5
	# vertices = [[-span, 0.0, span], [span, 0.0, span], [span, 0.0, -span], [-span, 0.0, -span]]
	# num_vertices = len(vertices)
	# faces = []
	# Generate triangles
	# for i in range(1, num_vertices - 1):
	# 	face = [0, i, i + 1]  # Triangle using first vertex and two adjacent vertices
	# 	faces.append(face)
	# faces = np.array(faces, dtype=np.int32)

	scene_atiss, _ = get_scene_both_formats(pth_base, pth_scene, raw_dataset)
	
	floor_plan, tr_floor, room_mask = floor_plan_from_scene(scene_atiss, args.path_to_floor_plan_textures)
	classes = np.array(test_dataset.class_labels)
	
	# make forward pass
	bbox_params = network.generate_boxes(room_mask=room_mask.to(dvc), max_boxes=52, device=dvc)
	# bbox_params = network.generate_boxes(room_mask=room_mask.to(dvc), device=dvc)
	
	# postprocess
	bbox_params = {k: v.cpu().numpy() for k, v in bbox_params.items()}
	boxes = test_dataset.post_process(bbox_params)

	objects = convert_atiss_output_to_our_format(boxes, range(1, bbox_params["class_labels"].shape[1] - 1), classes, objects_dataset, all_assets_metadata)

	final_scene = {
		"room_type": room_type,
		"bounds_top": bounds_top,
		"bounds_bottom": bounds_bottom,
		"objects": objects
	}

	# print("num objects: ", len(objects))

	return final_scene

def main(argv):
	parser = argparse.ArgumentParser(
		description="Generate scenes using ATISS and save in custom JSON format"
	)
	
	parser.add_argument(
		"config_file",
		help="Path to the file that contains the experiment configuration"
	)
	parser.add_argument(
		"path_to_pickled_3d_futute_models",
		help="Path to the 3D-FUTURE model meshes"
	)
	parser.add_argument(
		"path_to_floor_plan_textures",
		help="Path to floor texture images"
	)
	parser.add_argument(
		"--output_directory",
		help="Path to the output directory"
	)
	parser.add_argument(
		"--weight_file",
		default=None,
		help="Path to a pretrained model"
	)
	parser.add_argument(
		"--n-test-scenes",
		default=10,
		type=int,
		help="The number of sequences to be generated"
	)
	parser.add_argument(
		"--n-samples-per-scene",
		default=16,
		type=int,
		help="The number of samples per scene"
	)
	parser.add_argument(
		"--background",
		type=lambda x: list(map(float, x.split(","))),
		default="1,1,1,1",
		help="Set the background of the scene"
	)
	parser.add_argument(
		"--up_vector",
		type=lambda x: tuple(map(float, x.split(","))),
		default="0,1,0",
		help="Up vector of the scene"
	)
	parser.add_argument(
		"--camera_position",
		type=lambda x: tuple(map(float, x.split(","))),
		default="-0.10923499,1.9325259,-7.19009",
		help="Camer position in the scene"
	)
	parser.add_argument(
		"--camera_target",
		type=lambda x: tuple(map(float, x.split(","))),
		default="0,0,0",
		help="Set the target for the camera"
	)
	parser.add_argument(
		"--window_size",
		type=lambda x: tuple(map(int, x.split(","))),
		default="512,512",
		help="Define the size of the scene and the window"
	)
	parser.add_argument(
		"--with_rotating_camera",
		action="store_true",
		help="Use a camera rotating around the object"
	)
	parser.add_argument(
		"--save_frames",
		help="Path to save the visualization frames to"
	)
	parser.add_argument(
		"--n_frames",
		type=int,
		default=360,
		help="Number of frames to be rendered"
	)
	parser.add_argument(
		"--without_screen",
		action="store_true",
		help="Perform no screen rendering"
	)
	parser.add_argument(
		"--scene_id",
		default=None,
		help="The scene id to be used for conditioning"
	)
	parser.add_argument(
		"--room-type", 
		type=str, 
		default=None
	)
	parser.add_argument(
		"--do-full-scenes",
		action="store_true",
	)

	parser.add_argument(
		"--bon-per-sample",
		type=int,
		default=1,
	)
	
	# **********************************************************************************************************

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
	
	# Load config and dataset
	config = load_config(args.config_file)
	objects_dataset = ThreedFutureDataset.from_pickled_dataset(args.path_to_pickled_3d_futute_models)
	raw_test_dataset, test_dataset = get_dataset_raw_and_encoded(
		config["data"],
		filter_fn=filter_function(
			config["data"],
			split=config["validation"].get("splits", ["test"])
		),
		split=config["validation"].get("splits", ["test"])
	)
	
	# Build network
	network, _, _ = build_network(test_dataset.feature_size, test_dataset.n_classes, config, args.weight_file, device=dvc)
	network.eval()

	# test set split
	all_pths = get_pths_dataset_split(args.room_type, "test", prefix=pth_base)
	all_pths = all_pths[:args.n_test_scenes]

	all_test_instrs = get_test_instrs_all(args.room_type)
	
	rand_seeds = [1234, 3456, 5678]
	
	# for idx in tqdm(range(args.n_sequences)):
	# for idx, pth_scene in tqdm(enumerate(all_pths)):
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
				final_scene = generate_full_scene(args.bon_per_sample, room_type, bounds_top, bounds_bottom, pth_scene, test_dataset, raw_test_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, pth_base)
			else:
				final_scene = generate_instr_scene(args.bon_per_sample, room_type, rand_seed, pth_scene, all_test_instrs, test_dataset, raw_test_dataset, network, objects_dataset, args, output_dir, dvc, all_assets_metadata, pth_base, all_prompts, desc_to_category, sampling_engine)

			# save json file
			output_file = os.path.join(output_dir, str(rand_seed), f"{idx}_{rand_seed}.json")
			with open(output_file, 'w') as f:
				json.dump(final_scene, f, indent=4)

if __name__ == "__main__":
	main(sys.argv[1:])