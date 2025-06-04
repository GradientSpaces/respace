import numpy as np
import pandas as pd
import torch
import json
from shapely.geometry import box, Polygon
from matplotlib import pyplot as plt
import pdb
import os
from tqdm import tqdm
import trimesh
import trimesh.transformations as tf
from scipy.spatial.transform import Rotation as R
import pickle
from dotenv import load_dotenv
import argparse
from pathlib import Path
from trimesh.voxel.encoding import DenseEncoding
from trimesh.transformations import quaternion_matrix
from scipy.spatial.transform import Rotation as R
import copy
from transformers import AutoProcessor, AutoModelForVision2Seq, PaliGemmaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
import gc

from src.utils import get_pth_mesh, create_floor_plan_polygon, compute_fid_scores, get_scene_hash, get_vlm_prompt, compute_diversity_score
from src.viz import render_full_scene_and_export_with_gif, render_instr_scene_and_export_with_gif
from src.dataset import create_full_scene_from_before_and_added

def get_xz_bbox_from_obj(obj):

	bbox_position = obj.get("pos")
	bbox_size = obj.get("size")

	rotation_xyzw = np.array(obj.get("rot"))
	asset_rot_angle_euler, asset_rot_angle_radians = get_y_angle_from_xyzw_quaternion(rotation_xyzw)

	half_size_x = bbox_size[0] / 2
	half_size_z = bbox_size[2] / 2
	corners_2d_floor = np.array([
		[half_size_x, half_size_z],
		[-half_size_x, half_size_z],
		[-half_size_x, -half_size_z],
		[half_size_x, -half_size_z]
	])

	cos_theta = np.cos(asset_rot_angle_radians)
	sin_theta = np.sin(asset_rot_angle_radians)
	rotation_matrix = np.array([
		[cos_theta, -sin_theta],
		[sin_theta, cos_theta]
	])

	rotated_corners_2d_floor = np.dot(corners_2d_floor, rotation_matrix.T)
	translated_corners_2d_floor = rotated_corners_2d_floor + np.array([bbox_position[0], bbox_position[2]])
	
	polygon_coords_2d_floor = [(corner[0], corner[1]) for corner in translated_corners_2d_floor]
	bbox_2d_obj = Polygon(polygon_coords_2d_floor)

	# get height information of 3D bbox
	obj_height = bbox_size[1]
	obj_y_start = bbox_position[1]
	obj_y_end = bbox_position[1] + obj_height

	return bbox_2d_obj, obj_height, obj_y_start, obj_y_end

def create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon):
	num_verts = len(bounds_bottom)
	all_vertices = np.array(bounds_bottom + bounds_top)

	vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_plan_polygon, engine="triangle")
	idxs = []
	for i, row in enumerate(floor_faces):
		if np.any(row == num_verts):
			idxs.append(i)
	floor_faces = np.delete(floor_faces, idxs, axis=0)

	floor_mesh = trimesh.Trimesh(vertices=vtx, faces=floor_faces)

	ceiling_faces = floor_faces + num_verts

	side_faces = []
	for i in range(num_verts):
		next_i = (i + 1) % num_verts
		side_faces.append([i, next_i, i + num_verts])
		side_faces.append([next_i, next_i + num_verts, i + num_verts])
	side_faces = np.array(side_faces)

	all_faces = np.concatenate((floor_faces, ceiling_faces, side_faces), axis=0)
	
	room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

	trimesh.repair.fix_normals(room_mesh)

	# fig = plt.figure()
	# ax = fig.add_subplot(projection='3d')
	# ax.plot_trisurf(room_mesh.vertices[:, 0], room_mesh.vertices[:,2], room_mesh.vertices[:,1], triangles=room_mesh.faces);
	# plt.show()
	
	return room_mesh

def get_intersection_area(obj_x, obj_y, epsilon=1e-7):
	intersection = obj_x.intersection(obj_y)
	if intersection.is_empty:
		return 0.0
	area = intersection.area
	if area < epsilon:
		return 0.0
	return area

def compute_oob(obj, floor_plan_polygon, bounds_bottom, bounds_top, epsilon=1e-7, is_debug=False):

	bbox_obj, obj_height, obj_y_start, obj_y_end = get_xz_bbox_from_obj(obj)

	intersection_area = get_intersection_area(floor_plan_polygon, bbox_obj)

	room_bottom = bounds_bottom[0][1]
	room_top = bounds_top[0][1]

	if (obj_y_start < room_bottom and obj_y_end < room_bottom) or (obj_y_start > room_top and obj_y_end > room_top):
		obj_intersection_height = 0
	else:
		obj_intersection_height = abs(np.clip(obj_y_end, room_bottom, room_top) - np.clip(obj_y_start, room_bottom, room_top))

	bbox_vol_total = (bbox_obj.area)*obj_height
	bbox_vol_inside = (intersection_area*obj_intersection_height)

	oob = bbox_vol_total - bbox_vol_inside

	# if is_debug:
	# 	# visualize intersection with matplotlib
	# 	print(f"desc: {obj.get('desc')}")
	# 	print(f"oob: {oob}")
	# 	fig, ax = plt.subplots()
	# 	x, y = floor_plan_polygon.exterior.xy
	# 	ax.plot(x, y, color='b')
	# 	x, y = bbox_obj.exterior.xy
	# 	ax.plot(x, y, color='r')
	# 	ax.invert_yaxis()
	# 	plt.xticks(rotation=90)
	# 	plt.gca().set_aspect('equal')
	# 	plt.show()
	
	if oob < epsilon:
		return 0.0
	
	return oob

def compute_bbl(obj_x, obj_y, epsilon=1e-7, is_debug=False):

	bbox_obj_x, height_x, y_start_x, y_end_x = get_xz_bbox_from_obj(obj_x)
	bbox_obj_y, height_y, y_start_y, y_end_y = get_xz_bbox_from_obj(obj_y)

	intersection_area = get_intersection_area(bbox_obj_x, bbox_obj_y)

	# if is_debug:
	# 	# visualize intersection with matplotlib
	# 	print(f"desc: {obj_x.get('desc')} and {obj_y.get('desc')}")
	# 	print(f"bbl: {intersection_area}")
	# 	fig, ax = plt.subplots()
	# 	x, y = bbox_obj_x.exterior.xy
	# 	ax.plot(x, y, color='b')
	# 	x, y = bbox_obj_y.exterior.xy
	# 	ax.plot(x, y, color='r')
	# 	ax.invert_yaxis()
	# 	plt.xticks(rotation=90)
	# 	plt.gca().set_aspect('equal')
	# 	plt.show()
	
	if intersection_area == 0.0:
		return 0.0

	y_start_intersection = max(y_start_x, y_start_y)
	y_end_intersection = min(y_end_x, y_end_y)
	overlap_height = max(0, y_end_intersection - y_start_intersection)

	bbl = intersection_area * overlap_height

	if bbl < epsilon:
		return 0.0
	
	return bbl

# def visualize_voxels_matplotlib(voxel_matrix, voxel_size):
	
# 	# Create a figure and 3D axes
# 	fig = plt.figure()
# 	ax = fig.add_subplot(projection='3d')
	
# 	# Get the dimensions of the voxel matrix
# 	x_dim, y_dim, z_dim = voxel_matrix.shape
	
# 	# Create a color array the same shape as the voxel matrix
# 	colors = np.empty(voxel_matrix.shape, dtype=object)
# 	colors[voxel_matrix] = 'red'  # Set all filled voxels to red
	
# 	# Plot the voxels
# 	ax.voxels(voxel_matrix, 
# 			  facecolors=colors,
# 			  edgecolor='k',  # Black edges
# 			  alpha=0.5)      # Slight transparency to better see structure
	
# 	# Scale the axes to reflect voxel_size
# 	ax.set_xlim(0, x_dim)
# 	ax.set_ylim(0, y_dim)
# 	ax.set_zlim(0, z_dim)
	
# 	# Set labels
# 	ax.set_xlabel('X')
# 	ax.set_ylabel('Y')
# 	ax.set_zlabel('Z')
	
# 	# Optional: make the plot more visually appealing
# 	ax.grid(True)
	
# 	plt.show()

def show_colored_voxels_as_trimesh_scene(voxel_matrix, colors_matrix=None, pitch=0.05, origin=[0, 0, 0]):

	import pyglet.app

	if colors_matrix is None:
		colors_matrix = np.zeros((*voxel_matrix.shape, 3))
		colors_matrix[voxel_matrix] = [0, 0, 1]
	
	# Create transform matrix for origin offset
	transform = np.eye(4)
	transform[:3, 3] = origin
	transform[:3, :3] *= pitch  # Apply pitch scaling
	
	# Get coordinates of filled voxels
	filled_voxels = np.argwhere(voxel_matrix)
	
	if len(filled_voxels) == 0:
		return
	
	# Create a scene
	scene = trimesh.Scene()
	
	# Create vertices and faces for all boxes at once
	unit_box = trimesh.creation.box(extents=[1, 1, 1])
	vertices = np.tile(unit_box.vertices, (len(filled_voxels), 1))
	faces = np.tile(unit_box.faces, (len(filled_voxels), 1))
	
	# Adjust face indices
	for i in range(len(filled_voxels)):
		faces[i*12:(i+1)*12] += i * 8  # 8 vertices per box
	
	# Transform vertices for each box
	for i, (x, y, z) in enumerate(filled_voxels):
		start_idx = i * 8
		end_idx = start_idx + 8
		vertices[start_idx:end_idx] = (vertices[start_idx:end_idx] * pitch) + ([x * pitch, y * pitch, z * pitch])
	
	# Create colors array for all faces
	face_colors = np.zeros((len(faces), 4))
	for i, (x, y, z) in enumerate(filled_voxels):
		color = colors_matrix[x, y, z]
		if len(color) == 3:
			color = np.append(color, 0.8)
		face_colors[i*12:(i+1)*12] = color
	
	# Create single mesh for all boxes
	mesh = trimesh.Trimesh(
		vertices=vertices,
		faces=faces,
		face_colors=face_colors
	)
	
	scene.add_geometry(mesh)
	
	# Add coordinate axes
	# axis_length = max(voxel_matrix.shape) * pitch
	# axis = trimesh.creation.axis(origin_size=pitch, axis_length=axis_length)
	# scene.add_geometry(axis)

	# Handle pyglet event loop
	event_loop_constructor = pyglet.app.EventLoop
	event_loop_instance = pyglet.app.event_loop
	pyglet.app.EventLoop = pyglet.app.base.EventLoop
	pyglet.app.event_loop = pyglet.app.EventLoop()
	
	background_color = [240, 240, 240, 255]  # [R, G, B, A]
	scene.show(smooth=False, background=background_color)
	# scene.show()
	
	pyglet.app.EventLoop = event_loop_constructor
	pyglet.app.event_loop = event_loop_instance

def voxelize_mesh_and_get_matrix(asset_mesh, voxel_size):
	asset_voxels = asset_mesh.voxelized(pitch=voxel_size).fill()
	asset_voxel_matrix = asset_voxels.matrix
	return asset_voxel_matrix

def voxelize_raw_asset(pth_voxelized_mesh, obj, voxel_size, rotation_matrix=None):
	# print(f"voxelizing asset... for {pth_voxelized_mesh}")

	asset_jid = obj.get("sampled_asset_jid") if obj.get("sampled_asset_jid") is not None else obj.get("jid")

	pth_mesh = get_pth_mesh(asset_jid)
	asset_scene = trimesh.load(pth_mesh)

	if isinstance(asset_scene, trimesh.Scene):
		# asset_mesh = asset_scene.dump(concatenate=True)
		asset_mesh = asset_scene.to_geometry()
	else:
		asset_mesh = asset_scene
	
	if rotation_matrix is not None:
		#transform_matrix = np.eye(4)
		#transform_matrix[:3, :3] = rotation_matrix
		asset_mesh.apply_transform(rotation_matrix)
	
		asset_voxel_matrix = voxelize_mesh_and_get_matrix(asset_mesh, voxel_size)

		with open(pth_voxelized_mesh, 'wb') as fp:
			pickle.dump(asset_voxel_matrix, fp)
	else:
		asset_voxel_matrix = voxelize_mesh_and_get_matrix(asset_mesh, voxel_size)

	return asset_voxel_matrix

def get_y_angle_from_xyzw_quaternion(quaternion_xyzw):
	x, y, z, w = quaternion_xyzw

	angle_yaw_radians = np.arctan2(2 * (w * y + x * z), 1 - 2 * (y**2 + z**2))
	angle_yaw_degrees = np.degrees(angle_yaw_radians)
	angle_yaw_degrees = np.round(angle_yaw_degrees, 1)

	return angle_yaw_degrees, angle_yaw_radians

def prepare_asset(obj, voxel_size, metric_type, is_debug=False):

	rotation_xyzw = np.array(obj.get("rot"))
	asset_rot_y_euler_angle, _ = get_y_angle_from_xyzw_quaternion(rotation_xyzw)

	# print(obj.get("sampled_asset_jid"), obj)
	asset_jid = obj.get("sampled_asset_jid") if obj.get("sampled_asset_jid") is not None else obj.get("jid")

	if is_debug: 
		print(f"[{metric_type}] prepare asset with rot {asset_rot_y_euler_angle} and asset_jid {asset_jid}")

	# read from cache or create new voxelization
	# print(os.getenv("PTH_3DFUTURE_ASSETS"), asset_jid, f"rot-{str(asset_rot_y_euler_angle)}-scale-{str(voxel_size)}")
	pth_voxelized_mesh = os.path.join(os.getenv("PTH_3DFUTURE_ASSETS"), asset_jid, f"rot-{str(asset_rot_y_euler_angle)}-scale-{str(voxel_size)}.pkl")

	if os.path.isfile(pth_voxelized_mesh):
		with open(pth_voxelized_mesh, 'rb') as fp: 
			asset_voxel_matrix = pickle.load(fp)
	else:
		# trimesh expects wxyz instead of xyzw so we need to convert
		# we assume that rotation is roughly precise although we cache by single digit precision only
		quat_wxyz = [rotation_xyzw[3], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2]]
		rotation_matrix = quaternion_matrix(quat_wxyz)
		asset_voxel_matrix = voxelize_raw_asset(pth_voxelized_mesh, obj, voxel_size, rotation_matrix)

	# if obj.get("desc") == "A modern minimalist dark gray wardrobe with sliding mirror doors, shelves, and a hanging rod.":
	#if obj.get("desc") == "Modern minimalist king-size bed with dark brown fabric upholstery, low-profile wooden frame, and sleek design.":
		#raw_asset_matrix = voxelize_raw_asset(pth_voxelized_mesh, obj, voxel_size, None)  # Get raw
		#rotated_asset_matrix = voxelize_raw_asset(pth_voxelized_mesh, obj, voxel_size, rotation_matrix)  # Get rotated
		#show_colored_voxels_as_trimesh_scene(raw_asset_matrix, pitch=voxel_size)
		#show_colored_voxels_as_trimesh_scene(rotated_asset_matrix, pitch=voxel_size)
	# visualize_raw_and_rotated_asset(raw_asset_matrix, rotated_asset_matrix)

	asset_pos = np.array(obj.get("pos"))
	asset_pos_voxels = np.floor(asset_pos / voxel_size)

	asset_start_voxels = np.array([asset_voxel_matrix.shape[0] // 2, 0, asset_voxel_matrix.shape[2] // 2])
	asset_shift_from_origin = asset_pos_voxels - asset_start_voxels

	# print("asset_pos", asset_pos)
	# print("asset_pos_voxels", asset_pos_voxels)
	# print("asset_size", asset_voxel_matrix.shape)
	# print("asset_start_voxels", asset_start_voxels)
	# print("asset_shift_from_origin", asset_shift_from_origin)

	return asset_voxel_matrix, asset_shift_from_origin

def occupancy_overlap(voxel_matrix_a, voxel_matrix_b, offset_b):
	# overlap_matrix = voxel_matrix_a.copy().astype(int)
	overlap_matrix = copy.deepcopy(voxel_matrix_a).astype(int)
	for i in range(voxel_matrix_b.shape[0]):
		for j in range(voxel_matrix_b.shape[1]):
			for k in range(voxel_matrix_b.shape[2]):
				if voxel_matrix_b[i, j, k]:
					shifted_pos = (i + offset_b[0], j + offset_b[1], k + offset_b[2])
					if 0 <= shifted_pos[0] < overlap_matrix.shape[0] and 0 <= shifted_pos[1] < overlap_matrix.shape[1] and 0 <= shifted_pos[2] < overlap_matrix.shape[2]:
						# print(shifted_pos)
						overlap_matrix[shifted_pos[0], shifted_pos[1], shifted_pos[2]] += 1
	# visualize_voxels_mayavi(overlap_matrix == 2, voxel_size)
	return (overlap_matrix == 2)

def compute_mesh_oob(obj, voxel_size, room_origin_shift, room_voxel_matrix, voxel_volume, is_debug=False):

	asset_voxel_matrix, asset_shift_from_origin = prepare_asset(obj, voxel_size, "oob", is_debug)
	asset_offset = np.floor(room_origin_shift + asset_shift_from_origin).astype(int)

	inside_voxels = occupancy_overlap(room_voxel_matrix, asset_voxel_matrix, asset_offset)
	num_asset_voxels = np.sum(asset_voxel_matrix)
	asset_volume = num_asset_voxels * voxel_volume

	num_inside_voxels = np.sum(inside_voxels)
	inside_volume = num_inside_voxels * voxel_volume

	num_outside_voxels = num_asset_voxels - num_inside_voxels
	outside_volume = num_outside_voxels * voxel_volume

	if is_debug:
		print(f"desc: {obj.get('desc')}")
		print(f"total: {num_asset_voxels}")
		print(f"total volume asset: {asset_volume}")
		print(f"inside: {num_inside_voxels}")
		print(f"outside: {num_outside_voxels} ({round(num_outside_voxels/num_asset_voxels * 100, 2)}%)", )
		print(f"outside volume: {outside_volume}")
		print("")

	mesh_oob = asset_volume - inside_volume

	if mesh_oob > 0.0 and is_debug:
		colors = np.zeros((*asset_voxel_matrix.shape, 3))
		positions = np.argwhere(asset_voxel_matrix)
		room_space_positions = positions + asset_offset
		# Create mask for valid positions
		valid_mask = (
			(room_space_positions[:, 0] >= 0) & 
			(room_space_positions[:, 0] < inside_voxels.shape[0]) &
			(room_space_positions[:, 1] >= 0) & 
			(room_space_positions[:, 1] < inside_voxels.shape[1]) &
			(room_space_positions[:, 2] >= 0) & 
			(room_space_positions[:, 2] < inside_voxels.shape[2])
		)
		# Set all asset voxels to red first
		colors[asset_voxel_matrix] = [1, 0, 0]
		# Set green for valid inside voxels
		valid_positions = positions[valid_mask]
		room_positions = room_space_positions[valid_mask]
		inside_mask = inside_voxels[room_positions[:, 0], room_positions[:, 1], room_positions[:, 2]]
		colors[valid_positions[inside_mask][:, 0], valid_positions[inside_mask][:, 1], valid_positions[inside_mask][:, 2]] = [0.9, 0.9, 0.9]
		show_colored_voxels_as_trimesh_scene(asset_voxel_matrix, colors, pitch=voxel_size)

	return mesh_oob

def compute_mesh_bbl(obj_x, obj_y, voxel_size, voxel_volume, is_debug=False):

	asset_voxel_matrix_x, asset_shift_from_origin_x = prepare_asset(obj_x, voxel_size, "bbl", is_debug)
	asset_voxel_matrix_y, asset_shift_from_origin_y = prepare_asset(obj_y, voxel_size, "bbl", is_debug)

	inside_voxels = occupancy_overlap(asset_voxel_matrix_x, asset_voxel_matrix_y, np.floor(asset_shift_from_origin_y - asset_shift_from_origin_x).astype(int))
	
	num_inside_voxels = np.sum(inside_voxels)
	intersection_volume = num_inside_voxels * voxel_volume

	num_asset_voxels_x = np.sum(asset_voxel_matrix_x)
	# asset_volume_x = num_asset_voxels_x * voxel_volume
	num_asset_voxels_y = np.sum(asset_voxel_matrix_y)
	# asset_volume_y = num_asset_voxels_y * voxel_volume
	# asset_volume_union = asset_volume_x + asset_volume_y - intersection_volume

	mesh_bbl = intersection_volume

	if mesh_bbl > 0.0 and is_debug:
		# visualize in 3D
		colors = np.zeros((*asset_voxel_matrix_x.shape, 3))
		colors[asset_voxel_matrix_x] = [0.9, 0.9, 0.9]
		colors[inside_voxels] = [1, 0, 1]
		show_colored_voxels_as_trimesh_scene(asset_voxel_matrix_x, colors, pitch=0.05)

		# some stats
		print(f"obj_x: {obj_x.get('desc')}")
		print(f"obj_y: {obj_y.get('desc')}")
		print(f"num_asset_voxels_x: {num_asset_voxels_x}")
		print(f"num_asset_voxels_y: {num_asset_voxels_y}")
		print(f"intersection (inside x):", num_inside_voxels)
		print(f"intersection volume:", intersection_volume)
		print("")

	return mesh_bbl

def compute_pms_score(prompt, new_obj_desc):
	if prompt == None:
		return float("inf")

	prompt_words = prompt.split(" ")
	correct_words = 0
	for word in prompt_words:
		if word in new_obj_desc.lower():
			correct_words += 1

	# for pms, compute recall: how many words from the prompt are in the generated desc
	score = correct_words / len(prompt_words)
	# print(prompt_words, new_obj_desc, score)

	return score

def compute_dss_score(new_obj_desc, gt_obj_desc, sampling_engine):
	txt_dss_score = sampling_engine.compute_text_similarity(new_obj_desc, gt_obj_desc)
	return txt_dss_score

def compute_size_l2_dist(new_obj_size, gt_obj_size):
	w_pred, h_pred, d_pred = new_obj_size
	w_gt, h_gt, d_gt = gt_obj_size
	epsilon = 1e-6
	l2_dist_norm = np.sqrt(((w_pred - w_gt)/(w_gt + epsilon))**2 + ((h_pred - h_gt)/(h_gt + epsilon))**2 + ((d_pred - d_gt)/(d_gt + epsilon))**2)
	return l2_dist_norm

def eval_bounds(scene):
	floor_plan_polygon = create_floor_plan_polygon(scene.get("bounds_bottom"))
	if floor_plan_polygon.area > 0 and np.array(scene.get("bounds_bottom")).shape == np.array(scene.get("bounds_top")).shape:
		return True
	else:
		return False

def eval_scene(scene, is_debug=True, voxel_size=0.05, total_loss_threshold=0.1, idx=None, do_pms_full_scene=False):

	bounds_top = scene.get("bounds_top")
	bounds_bottom = scene.get("bounds_bottom")
	floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)
	objs = scene.get("objects")
	voxel_volume = voxel_size ** 3

	# voxelize room mesh
	room_mesh = create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon)
	room_voxels = room_mesh.voxelized(pitch=voxel_size).fill()
	room_voxel_matrix = room_voxels.matrix
	room_size_voxels = np.ceil(abs(room_mesh.bounds[0] - room_mesh.bounds[1]) / voxel_size)
	room_origin_shift = np.array([room_size_voxels[0] / 2.0, 0, room_size_voxels[2] / 2.0])

	mesh_oobs, mesh_bbls = [], []
	
	idx_highest_pbl_loss = None
	highest_pbl_loss = float("-inf")

	if objs is not None:
		for i, obj_x in enumerate(objs):
			obj_pbl = 0.0

			# oob = out of bounds loss
			oob = compute_oob(obj_x, floor_plan_polygon, bounds_bottom, bounds_top, is_debug=is_debug)
			if oob > 0.0:
				if is_debug:
					print("oob is not zero!", oob, "computing voxelized mesh loss...")
				try:
					mesh_oob = compute_mesh_oob(obj_x, voxel_size, room_origin_shift, room_voxel_matrix, voxel_volume, is_debug=is_debug)
				except Exception as e:
					print(f"Error computing mesh oob for {obj_x.get('desc')}: {e}")
					mesh_oob = 0.0
				obj_pbl += mesh_oob
			else:
				mesh_oob = 0.0
			mesh_oobs.append(mesh_oob)

			# mbl = mesh based loss
			for obj_y in objs[i + 1:]:
				bbl = compute_bbl(obj_x, obj_y, is_debug=is_debug)
				if bbl > 0.0:
					if is_debug:
						print("bbl is not zero!", bbl, "computing voxelized mesh loss...")
					try:
						mesh_bbl = compute_mesh_bbl(obj_x, obj_y, voxel_size, voxel_volume, is_debug=is_debug)
					except Exception as e:
						print(f"Error computing mesh bbl for {obj_x.get('desc')} and {obj_y.get('desc')}: {e}")
						mesh_bbl = 0.0
					obj_pbl += mesh_bbl
				else:
					mesh_bbl = 0.0
				mesh_bbls.append(mesh_bbl)
			
			if obj_pbl > highest_pbl_loss:
				idx_highest_pbl_loss = i
				highest_pbl_loss = obj_pbl

	metrics = {
		'total_oob_loss': np.sum(mesh_oobs).item() if len(mesh_oobs) > 0 else 0.0,
		'total_mbl_loss': np.sum(mesh_bbls).item() if len(mesh_bbls) > 0 else 0.0,
		'obj_with_highest_pbl_loss': {
			'idx': idx_highest_pbl_loss,
			'pbl': highest_pbl_loss,
		}
	}

	metrics["total_pbl_loss"] = metrics['total_oob_loss'] + metrics['total_mbl_loss']
	metrics['is_valid_scene_pbl'] = bool(metrics['total_pbl_loss'] <= total_loss_threshold)

	# metrics["txt_pms_score"] = float('inf')
	# metrics["txt_pms_sampled_score"] = float('inf')
	metrics["txt_pms_score"] = 0.0
	metrics["txt_pms_sampled_score"] = 0.0

	if objs is not None and len(objs) > 0:
		all_txt_pms_scores = []
		all_txt_pms_sampled_scores = []
		objs_pms = objs if do_pms_full_scene else [ objs[-1] ]
		for obj in objs_pms:
			if obj.get("prompt") != None:
				new_obj_desc = obj.get("desc")
				
				txt_pms_score = compute_pms_score(obj.get("prompt"), new_obj_desc)
				all_txt_pms_scores.append(txt_pms_score)

				txt_pms_score_sampled = compute_pms_score(obj.get("prompt"), obj.get("sampled_asset_desc"))
				# print(f"prompt: {obj.get('prompt')}, new_obj_desc: {new_obj_desc}, txt_pms_score: {txt_pms_score}, txt_pms_score_sampled: {txt_pms_score_sampled}")
				all_txt_pms_sampled_scores.append(txt_pms_score_sampled)

		if len(all_txt_pms_scores) > 0:
			metrics["txt_pms_score"] = np.mean(all_txt_pms_scores)
		
		if len(all_txt_pms_sampled_scores) > 0:
			metrics["txt_pms_sampled_score"] = np.mean(all_txt_pms_sampled_scores)
			
	if is_debug:
		print(f">> ✅ valid scene according to metrics ({metrics['total_pbl_loss']})" if metrics['is_valid_scene_pbl'] else f">> ⛔️ INVALID scene according to metrics ({metrics['total_pbl_loss']})")

	return metrics

def eval_scene_before_after_with_delta(scene_before, scene_after, is_debug=False):
	before_metrics = eval_scene(scene_before, is_debug=False)
	
	if is_debug:
		print(f"before metrics: {before_metrics}")

	after_metrics = eval_scene(scene_after, is_debug=is_debug)
	
	return {
		'is_valid_scene_pbl': after_metrics['is_valid_scene_pbl'],
		'scene': scene_after,
		'total_oob_loss': after_metrics['total_oob_loss'],
		'total_mbl_loss': after_metrics['total_mbl_loss'],
		'total_pbl_loss': after_metrics['total_pbl_loss'],
		'delta_oob_loss': after_metrics['total_oob_loss'] - before_metrics['total_oob_loss'],
		'delta_mbl_loss': after_metrics['total_mbl_loss'] - before_metrics['total_mbl_loss'],
		'delta_pbl_loss': after_metrics['total_pbl_loss'] - before_metrics['total_pbl_loss'],
		'txt_pms_score': after_metrics['txt_pms_score'],
		'txt_pms_sampled_score': after_metrics['txt_pms_sampled_score'],
	}

def compute_mean_metrics_for_seed(room_type, is_full_scene, metrics_list, pth_output, n_test_scenes):

	mean_metrics = {
		'total_oob_loss': np.mean([m['total_oob_loss'] for m in metrics_list]),
		'total_mbl_loss': np.mean([m['total_mbl_loss'] for m in metrics_list]),
		'total_pbl_loss': np.mean([m['total_pbl_loss'] for m in metrics_list]),		

		'valid_scene_ratio_pbl': np.mean([m['is_valid_scene_pbl'] for m in metrics_list]),
		'valid_scene_ratio_json': len([ 1 for m in metrics_list if m.get('total_pbl_loss') is not None ]) / n_test_scenes,

		# 'novel_scene_ratio': np.mean([m['is_novel_scene'] for m in metrics_list]),
		# 'unique_scene_ratio': np.mean([m['is_unique_scene'] for m in metrics_list]),
		
		'txt_pms_score': np.mean([m['txt_pms_score'] for m in metrics_list]),
		'txt_pms_sampled_score': np.mean([m['txt_pms_sampled_score'] for m in metrics_list]),
	}
	
	if metrics_list[0].get('delta_oob_loss') != None:
		mean_metrics['delta_oob_loss'] = np.mean([m['delta_oob_loss'] for m in metrics_list])
		mean_metrics['delta_mbl_loss'] = np.mean([m['delta_mbl_loss'] for m in metrics_list])
		mean_metrics['delta_pbl_loss'] = np.mean([m['delta_pbl_loss'] for m in metrics_list])

	compute_fid_scores("diag", fid_score_name=f"3d-front-train-{'full' if is_full_scene else 'instr'}-scenes-{room_type}-diag", pth_src=f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/3d-front-train-{'full' if is_full_scene else 'instr'}-scenes-{room_type}/diag", pth_gen=f"{pth_output}/diag", aggregated_metrics=mean_metrics, do_renderings=True, dataset_res=1024)
	compute_fid_scores("top", fid_score_name=f"3d-front-train-{'full' if is_full_scene else 'instr'}-scenes-{room_type}-top", pth_src=f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/3d-front-train-{'full' if is_full_scene else 'instr'}-scenes-{room_type}/top", pth_gen=f"{pth_output}/top", aggregated_metrics=mean_metrics, do_renderings=True, dataset_res=1024)

	compute_diversity_score("top", pth_gen=f"{pth_output}/diag", do_renderings=True, dvc="cuda", aggregated_metrics=mean_metrics)

	return mean_metrics

def get_all_train_scene_hashes_for_room_type(room_type):
	# precompute hashes of all full scenes for given room_type in training set (if not done already)
	hash_file = os.getenv("PTH_DATASET_CACHE") + f"/scene_hashes_train_{room_type}.pkl"
	if os.path.isfile(hash_file):
		print("loading train scene hashes...")
		with open(hash_file, 'rb') as fp:
			hashes = pickle.load(fp)
	else:
		print("precomputing train scene hashes...")
		hashes = set()
		pth_root = os.getenv("PTH_STAGE_2_DEDUP")
		all_pths_train = [f for f in os.listdir(pth_root) if f.endswith('.json') and not f.startswith(".")]
		for pth in tqdm(all_pths_train):
			scene = json.load(open(os.path.join(pth_root, pth)))
			if scene.get("room_type") != "all" and scene.get("room_type") != room_type:
				continue
			scene_hash = get_scene_hash(scene)
			hashes.add(scene_hash)
		with open(hash_file, 'wb') as fp:
			pickle.dump(hashes, fp)
		
	return hashes

def get_simplified_scene_for_novelty_and_uniqueness(scene, all_assets_metadata_simple_descs):
	scene_simplified = copy.deepcopy(scene)
	for obj in scene_simplified.get("objects"):
		obj["desc"] = all_assets_metadata_simple_descs.get(obj["desc"])

def compute_mean_and_std_from_list(metrics_list, all_n_samples_actual, n_test_scenes):
	# Initialize dictionaries to store means and standard deviations
	mean_metrics = {}
	std_metrics = {}
	
	# Get all keys from the first dictionary
	all_keys = metrics_list[0].keys()

	# scale oob / mbl / pbl loss and their delta losses by 1e-3
	for key in all_keys:
		if key in ['total_oob_loss', 'total_mbl_loss', 'total_pbl_loss', 'delta_oob_loss', 'delta_mbl_loss', 'delta_pbl_loss']:
			for metrics in metrics_list:
				metrics[key] *= 1e3
	
	# Calculate mean and standard deviation for each key
	for key in all_keys:
		values = [metrics[key] for metrics in metrics_list if key in metrics]
		if values:
			mean_metrics[key] = np.mean(values)
			std_metrics[key] = np.std(values)

	print(f"\n============== eval ({','.join([str(n) for n in all_n_samples_actual])} / {n_test_scenes}) ==============\n")

	# Create a dictionary for formatted metrics with 3 decimal places
	print_metrics = {}

	# Helper function to format metrics with mean and std
	def format_metric(key, suffix=""):
		formatted = f"{mean_metrics[key]:.2f} (+/- {std_metrics[key]:.2f}){suffix}"
		print(f"{key}: {formatted}")
		return formatted

	# Format and print all metrics
	print_metrics['fid_score_top'] = format_metric('fid_score_top')
	print_metrics['fid_clip_score_top'] = format_metric('fid_clip_score_top')
	print_metrics['kid_score_top'] = format_metric('kid_score_top', " (x 0.001)")
	print("")

	print_metrics['total_oob_loss'] = format_metric('total_oob_loss', " (x 0.001)")
	print_metrics['total_mbl_loss'] = format_metric('total_mbl_loss', " (x 0.001)")
	print_metrics['total_pbl_loss'] = format_metric('total_pbl_loss', " (x 0.001)")
	print("")

	if metrics_list[0].get('delta_oob_loss') is not None:
		print_metrics['delta_oob_loss'] = format_metric('delta_oob_loss', " (x 0.001)")
		print_metrics['delta_mbl_loss'] = format_metric('delta_mbl_loss', " (x 0.001)")
		print_metrics['delta_pbl_loss'] = format_metric('delta_pbl_loss', " (x 0.001)")
		print("")

	print_metrics['valid_scene_ratio_pbl'] = format_metric('valid_scene_ratio_pbl')
	print_metrics['valid_scene_ratio_json'] = format_metric('valid_scene_ratio_json')
	print("")

	print_metrics['txt_pms_score'] = format_metric('txt_pms_score')
	print_metrics['txt_pms_sampled_score'] = format_metric('txt_pms_sampled_score')
	print("")

	print_metrics['diversity_score_top'] = format_metric('diversity_score_top')
	print("")

	# print_metrics['novel_scene_ratio'] = format_metric('novel_scene_ratio')
	# print_metrics['unique_scene_ratio'] = format_metric('unique_scene_ratio')
	# print("")

	print("==================================\n")

	# save mean and std to file
	final_metrics = {
		"mean_metrics": mean_metrics,
		"std_metrics": std_metrics,
		"print_metrics": print_metrics,
		"all_n_samples_actual": all_n_samples_actual,
	}

	return final_metrics

def run_eval(args):

	print("running eval for pth_output:", args.pth_output)

	env_file = f".env.{args.env}"
	load_dotenv(env_file)

	# train_scene_hashes = get_all_train_scene_hashes_for_room_type(args.room_type)
	# gen_scene_hashes = set()

	all_metrics_mean_seed = []
	all_metrics_raw_seed = []
	all_n_samples_actual = []

	rand_seeds = [1234, 3456, 5678]
	# rand_seeds = [ 5678 ]

	# all_metrics_raw_seed = json.load(open("/home/martinbucher/git/stan-24-sgllm/eval/metrics-raw/eval_samples_respace_instr_bedroom_qwen1.5B_raw_V2.json"))

	for idx_seed, rand_seed in enumerate(rand_seeds):
		print(f"evaluating samples for seed {rand_seed}...")

		metrics_list = []

		pth_input = Path(args.pth_input) / str(rand_seed)
		pth_viz_output = Path(args.pth_output) / str(rand_seed)

		n_samples_actual = len([f for f in os.listdir(Path(args.pth_input) / str(rand_seed)) if f.endswith('.json') and not f.startswith(".")])
		if n_samples_actual == 0:
			print("no scenes found... skipping eval for rand seed", rand_seed)
			return
		
		n_samples_actual = min(n_samples_actual, args.n_test_scenes)
		all_n_samples_actual.append(n_samples_actual)
		
		all_pths_scenes = [f for f in os.listdir(Path(args.pth_input) / str(rand_seed)) if f.endswith('.json') and not f.startswith(".")]
		all_pths_scenes = sorted(all_pths_scenes, key=lambda x: int(x.split("_")[0]))
		
		all_pths_scenes = all_pths_scenes[:args.n_test_scenes]
		# all_pths_scenes = all_pths_scenes[413:414]
		
		for pth in tqdm(all_pths_scenes):
			# print(f"evaluating scene {pth}...")
			scene = json.load(open(pth_input / pth))
			idx = int(pth.split("_")[0])
			
			if args.is_full_scene:
				render_full_scene_and_export_with_gif(scene, idx, pth_output=pth_viz_output, create_gif=args.create_gifs)
				metrics = eval_scene(scene, is_debug=False)
				metrics["scene"] = scene
			else:
				render_instr_scene_and_export_with_gif(scene, idx, pth_output=pth_viz_output, create_gif=args.create_gifs)
				scene_before = copy.deepcopy(scene)
				scene_before["objects"] = scene_before["objects"][:-1]
				metrics = eval_scene_before_after_with_delta(scene_before, scene_after=scene, is_debug=False)
				# render_instr_scene_and_export_with_gif(scene_before, f"{idx}-before", pth_output=pth_viz_output, create_gif=args.create_gifs)

			# replace raw file:
			# metrics_raw = json.load(open("/home/martinbucher/git/stan-24-sgllm/eval/metrics-raw/eval_samples_respace_instr_bedroom_qwen1.5B_raw.json"))
			# metrics_raw[2][413] = metrics
			# with open("/home/martinbucher/git/stan-24-sgllm/eval/metrics-raw/eval_samples_respace_instr_bedroom_qwen1.5B_raw_V2.json", 'w') as f:
			# 	json.dump(metrics_raw, f, indent=4)
			# print(metrics)
			# exit()

			metrics_list.append(metrics)

		if args.do_metrics:
			# save raw list of metrics for each scene
			all_metrics_raw_seed.append(metrics_list)

			# from cache
			# metrics_list = all_metrics_raw_seed[idx_seed]

			# compute mean metrics for this seed across all test scenes
			metrics_mean_seed = compute_mean_metrics_for_seed(args.room_type, args.is_full_scene, metrics_list, os.path.join(args.pth_output, str(rand_seed)), args.n_test_scenes)
			all_metrics_mean_seed.append(metrics_mean_seed)
	
	if args.do_metrics:
		# construct filename from props
		filename = args.pth_output.split("/")[:-1]
		filename = filename[1:]
		filename = "_".join(filename)
		if args.metrics_file_postfix is not None:
			filename += "_" + args.metrics_file_postfix
	
		final_metrics = compute_mean_and_std_from_list(all_metrics_mean_seed, all_n_samples_actual, args.n_test_scenes)
		with open(f"./eval/metrics/{filename}.json", 'w') as f:
			json.dump(final_metrics, f, indent=4)

		# save metrics to file
		with open(f"./eval/metrics-raw/{filename}_raw.json", 'w') as f:
			json.dump(all_metrics_raw_seed, f, indent=4)

	print("EVALUATION FINISHED!")

def eval_full_scenes_autogressively():
	# for midiff and atiss, load each scene from folder, then eval scene with increasing number objects from list in the same order
	# save each list of metrics to json file

	os.makedirs("./eval/metrics-full-objs", exist_ok=True)

	for room_type in ["bedroom", "livingroom", "all"]:
		for baseline in ["midiff", "atiss"]:
			for seed in [1234, 3456, 5678]:
				all_metrics = {}
				pth_root = f"./eval/samples/baseline-{baseline}/full/{room_type}/json/{seed}"
				for idx in range(500):
					pth = os.path.join(pth_root, f"{idx}_{seed}.json")
					if os.path.isfile(pth):
						metrics_for_scene = {}
						scene = json.load(open(pth))
						n_objects = len(scene.get("objects"))
						for i in range(1, n_objects):
							print("doing eval: ", i, "/", n_objects, "for scene", pth, seed, baseline, room_type)
							scene_cp = copy.deepcopy(scene)
							scene_cp["objects"] = scene_cp["objects"][:i + 1]
							metrics = eval_scene(scene_cp, is_debug=False)
							metrics_for_scene[i] = metrics
						all_metrics[idx] = metrics_for_scene
					else:
						print(f"scene {pth} not found...")

				# save metrics to file
				filename = f"eval_samples_{baseline}_{room_type}_{seed}"
				with open(f"./eval/metrics-full-objs/{filename}.json", 'w') as f:
					json.dump(all_metrics, f, indent=4)


if __name__ == "__main__":

	load_dotenv(".env.stanley")
	# load_dotenv(".env.local")

	parser = argparse.ArgumentParser(description='Author: Martin Juan José Bucher')

	parser.add_argument('--env', dest='env', type=str, choices=["sherlock", "local", "stanley"], default="local")
	parser.add_argument('--pth-input', type=str)
	parser.add_argument('--pth-output', type=str)
	parser.add_argument('--do-metrics', action='store_true', default=False)
	parser.add_argument('--room-type', type=str, choices=["bedroom", "diningroom", "livingroom", "all"])
	parser.add_argument('--is-full-scene', action='store_true', default=False)
	parser.add_argument('--n-test-scenes', type=int, default=500)
	parser.add_argument('--create-gifs', action='store_true', default=False)

	parser.add_argument('--metrics-file-postfix', type=str, default=None)

	run_eval(parser.parse_args())

	# scene = json.loads('{"room_type": "bedroom", "bounds_top": [[-1.45, 2.6, 2.45], [0.45, 2.6, 2.45], [0.45, 2.6, 1.45], [1.45, 2.6, 1.45], [1.45, 2.6, -2.45], [-1.45, 2.6, -2.45]], "bounds_bottom": [[-1.45, 0.0, 2.45], [0.45, 0.0, 2.45], [0.45, 0.0, 1.45], [1.45, 0.0, 1.45], [1.45, 0.0, -2.45], [-1.45, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "size": [0.57, 1.21, 0.63], "pos": [1.25, 0.0, 1.25], "rot": [0, 0, 0, 1], "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6-(0.68)-(0.68)-(0.68)"}, {"desc": "Elegant wooden wardrobe with three geometric-patterned glass doors, two drawers, and modern metal handles.", "size": [1.45, 2.28, 0.62], "pos": [0.87, 0.0, -2.1], "rot": [0, 0, 0, 0], "sampled_asset_jid": "a0b67c64-15a4-4969-91a6-89e365d87d12"}, {"desc": "Modern contemporary pendant lamp featuring white fabric conical shades on a geometric gold metal frame with multiple light sources.", "size": [1.06, 1.03, 0.47], "pos": [0.02, 2.08, -0.44], "rot": [0, -0.71254, 0, 0.70164], "sampled_asset_jid": "5a72093d-b9e5-4823-906b-331ced5e08d7"}, {"desc": "Modern beige upholstered king-size bed with minimalist design and neatly tailored edges.", "size": [1.9, 1.11, 2.23], "pos": [-0.29, 0.0, -0.3], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "6c7bf8e0-37a2-4661-a554-3af2b1e242d6"}, {"desc": "A modern-traditional nightstand in dark brown wood with a gold geometric patterned front, featuring two drawers and sleek elevated legs.", "size": [0.58, 0.59, 0.46], "pos": [-1.31, 0.0, -1.31], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "8b8cdbde-57e3-432a-a46a-89a77f8e6294"}, {"desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "pos": [-1.1, 0.0, 1.38], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [1.1, 1.36, 0.81], "prompt": "modern dark wooden desk", "sampled_asset_jid": "ec9190d1-cc42-4a85-bb1e-730ed7642f51", "sampled_asset_desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "sampled_asset_size": [1.1008340120315552, 1.3596680217888206, 0.8073000013828278], "uuid": "51b03ac6-941c-4beb-a8c1-84d69f8a41c1"}, {"desc": "A modern, ergonomic office chair with a mesh back, leather seat, metal frame, 360-degree swivel base, and rolling casters.", "pos": [-0.64, 0.0, 1.56], "rot": [0.0, -0.80486, 0.0, 0.59347], "size": [0.66, 0.95, 0.65], "prompt": "office chair", "sampled_asset_jid": "284277da-b2ed-4dea-bc97-498596443294", "sampled_asset_desc": "A modern, ergonomic office chair with a mesh back, leather seat, metal frame, 360-degree swivel base, and rolling casters.", "sampled_asset_size": [0.663752019405365, 0.9482090100936098, 0.6519539952278137], "uuid": "f2259272-7d9d-4015-8353-d8a5d46f1b33"}]}')
	# eval_scene(scene, is_debug=True, voxel_size=0.05, total_loss_threshold=0.1, idx=None, do_pms_full_scene=False)

	# eval_full_scenes_autogressively()
	
	# scene = json.loads('{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "Modern mid-century dark brown leather three-seat sofa with tufted backrest, padded arms, and silver decorative pillows.", "pos": [1.47, 0.0, 0.45], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.21, 0.92, 0.98], "prompt": "modern mid century brown couch", "sampled_asset_jid": "2d8e7040-14d8-4aba-84ee-356a1eae11e8", "sampled_asset_desc": "Modern three-seat sofa with classic tufting, brown leather upholstery, and contrasting cushions.", "sampled_asset_size": [2.214682102203369, 0.9059539784238559, 0.9729260504245758], "uuid": "13873eeb-191d-483a-aa61-254c446b0d7a"}]}')
	# eval_scene(scene, is_debug=True, voxel_size=0.05, total_loss_threshold=0.1, idx=None, do_pms_full_scene=False)
