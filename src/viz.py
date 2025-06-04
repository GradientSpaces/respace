import numpy as np
import trimesh
import pyrender
from PIL import Image
from pathlib import Path
import os
import json
from trimesh.transformations import quaternion_matrix, translation_matrix
from dotenv import load_dotenv
import pdb
from tqdm import tqdm
import time
import re
import traceback
from collections import defaultdict
from src.sample import AssetRetrievalModule
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import copy
import seaborn as sns
import glob
from scipy import stats
import pandas as pd
import cv2
import imageio.v2 as imageio
import math

from src.utils import get_pth_mesh, create_floor_plan_polygon, remove_and_recreate_folder, precompute_fid_scores_for_caching, get_pths_dataset_split, get_model, get_test_instrs_all
from src.dataset import load_train_val_test_datasets, create_full_scene_from_before_and_added, create_instruction_from_scene, process_scene_sample

# Add this before your rendering code
import ctypes
from OpenGL.GL import glGenTextures
from OpenGL.GL import GLuint

# Monkey patch the problematic function
def patched_glGenTextures(count, textures):
	textures_array = (GLuint * count)()
	glGenTextures(count, textures_array)
	return textures_array[0]

# Replace the original function
import OpenGL.GL
OpenGL.GL.glGenTextures = patched_glGenTextures

def fix_textures(mesh, mesh_path):
	# print("fixing mesh...")
	if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
		try:
			# Convert material image to PIL Image
			img = Image.fromarray(mesh.visual.material.image)
			# Force convert to RGBA
			img = img.convert('RGBA')
			# Update the material with RGBA image
			mesh.visual.material.image = np.array(img)
			# Also update the material itself
			mesh.visual.material = trimesh.visual.material.PBRMaterial(
				baseColorTexture=np.array(img),
				metallicFactor=0.0,
				roughnessFactor=1.0
			)
		except Exception as e:
			print(f"Failed to convert texture for {mesh_path}: {e}")
			# Fallback to a simple material if texture conversion fails
			mesh.visual.material = trimesh.visual.material.PBRMaterial(
				baseColorFactor=[0.8, 0.8, 0.8, 1.0]
			)

def load_mesh_with_transform(mesh_path, position=None, rotation=None, scale=None):
	mesh = trimesh.load(mesh_path)

	# Convert any 2-channel textures to RGBA and make materials double-sided
	if isinstance(mesh, trimesh.Scene):
		# For scene with multiple geometries
		for m in mesh.geometry.values():
			fix_textures(m, mesh_path)
			# Make materials double-sided
			if hasattr(m, 'visual') and hasattr(m.visual, 'material'):
				m.visual.material.doubleSided = True
				# Ensure proper alpha mode for transparent materials
				if hasattr(m.visual.material, 'alphaMode'):
					m.visual.material.alphaMode = 'BLEND'
	else:
		# For single mesh
		fix_textures(mesh, mesh_path)
		# Make materials double-sided for single mesh
		if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
			mesh.visual.material.doubleSided = True
			# Ensure proper alpha mode for transparent materials
			if hasattr(mesh.visual.material, 'alphaMode'):
				mesh.visual.material.alphaMode = 'BLEND'

	if scale is not None:
		mesh.apply_scale(scale)
	
	if rotation is not None:
		# Convert [x,y,z,w] to [w,x,y,z] for trimesh
		quat_wxyz = [rotation[3], rotation[0], rotation[1], rotation[2]]
		rotation_matrix = quaternion_matrix(quat_wxyz)
		mesh.apply_transform(rotation_matrix)
	
	if position is not None:
		translation = translation_matrix(position)
		mesh.apply_transform(translation)
	
	return mesh

def setup_camera(scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span):
	fov = np.pi / 4.0
	camera = pyrender.PerspectiveCamera(yfov=np.pi/4.0, znear=0.05, zfar=100.0)
	scene_x, scene_y, scene_z = scene_span
	scene_aspect = scene_x / max(scene_z, 1e-5)
	
	if scene_aspect > 1.0:
		limiting_span = scene_x
	else:
		limiting_span = scene_z
	
	if view_type == "top":
		if camera_height == None:
			if use_dynamic_zoom:
				required_distance = (limiting_span/2) / np.tan(fov/2)
				camera_height = max(2.0, required_distance + 2.5)
			else:
				camera_height = 13.0

		camera_pose = np.array([
			[1.0, 0.0, 0.0, 0],
			[0.0, 0.0, 1.0, camera_height],
			[0.0, -1.0, 0.0, 0],
			[0.0, 0.0, 0.0, 1.0]
		])
	elif view_type == "diag":
		if camera_height == None:
			if use_dynamic_zoom:
				diagonal_length = np.sqrt(scene_x**2 + scene_y**2 + scene_z**2)
				required_distance = (diagonal_length/2) / np.tan(fov/2)
				camera_height = max(2.0, required_distance*0.8)
			else:
				camera_height = 10.0

		position = np.array([camera_height, camera_height, camera_height])
		
		# Compute look vectors
		target = np.array([0.0, 0.0, 0.0])
		
		forward = target - position
		forward = forward / np.linalg.norm(forward)
		
		world_up = np.array([0.0, 1.0, 0.0])
		
		right = np.cross(forward, world_up)
		right = right / np.linalg.norm(right)

		up = np.cross(right, forward)
		up = up / np.linalg.norm(up)
		
		# Construct view matrix
		camera_pose = np.eye(4)
		camera_pose[:3, 0] = right
		camera_pose[:3, 1] = up
		camera_pose[:3, 2] = -forward  # Note the negative forward
		camera_pose[:3, 3] = position
		
		# print("Camera position:", position)
		# print("Camera matrix:", camera_pose)

	scene.add(camera, pose=camera_pose)
	return camera_pose

def setup_lighting(scene, camera_pose):
	light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
	#light_pose = np.eye(4)
	#light_pose[:3, :3] = camera_pose[:3, :3]  # Keep camera orientation
	#light_pose[1, 3] = 3.0  # Move up by 3 units
	# light_pose = camera_pose.copy()
	# light_pose[1, 3] = 3.0
	light_pose = camera_pose.copy()
	light_pose[1, 3] = 3.0

	# If we're using OSMesa, adjust the transform to match Pyglet behavior
	# if os.environ.get('PYOPENGL_PLATFORM') == 'osmesa':
	# 	# Try inverting certain rotations to match Pyglet's interpretation
	# 	correction = np.array([
	# 		[ 1,  0,  0,  0],
	# 		[ 0,  0, -1,  3],  # Flip the forward direction
	# 		[ 0,  1,  0,  0],
	# 		[ 0,  0,  0,  1]
	# 	])
	# 	light_pose = correction @ light_pose

	# print("Light pose matrix:")
	# print(light_pose)
	
	scene.add(light, pose=light_pose)
	
	scene.ambient_light = np.array([0.6, 0.6, 0.6, 1.0])

def create_bbox(size, pos, rot, color=[0.0, 0.0, 1.0, 0.7]):
	bbox = trimesh.creation.box(extents=size)

	material = trimesh.visual.material.PBRMaterial(baseColorFactor=color, alphaMode='BLEND', doubleSided=False, metallicFactor=0.0, roughnessFactor=1.0)
	bbox.visual = trimesh.visual.TextureVisuals(material=material, uv=bbox.vertices[:, [0, 1]])
	bbox.fix_normals(multibody=True)

	bottom_center_transform = np.eye(4)
	bottom_center_transform[1, 3] = size[1] / 2
	bbox.apply_transform(bottom_center_transform)
	
	if rot is not None:
		# Convert [x,y,z,w] to [w,x,y,z] for trimesh
		quat_wxyz = [rot[3], rot[0], rot[1], rot[2]]
		rotation_matrix = quaternion_matrix(quat_wxyz)
		bbox.apply_transform(rotation_matrix)
	
	if pos is not None:
		translation = translation_matrix(pos)
		bbox.apply_transform(translation)
	
	return bbox

def create_floor_slab(bounds_bottom):
	# bounds_bottom = [[-5, 0, -5], [5, 0, -5], [5, 0, 5], [-5, 0, 5]]
	floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)
	
	floor_mesh = trimesh.creation.extrude_polygon(
		polygon=floor_plan_polygon,
		height=0.15
	)

	rotation = trimesh.transformations.rotation_matrix(
		angle=np.pi/2,
		direction=[1, 0, 0]
	)
	floor_mesh.apply_transform(rotation)
	
	try:
		img = Image.open('src/frontend/public/texture.png')
		if img.mode != 'RGBA':
			img = img.convert('RGBA')

		material = trimesh.visual.material.PBRMaterial(
			baseColorTexture=img
		)
		
		vertices = floor_mesh.vertices
		bounds = floor_mesh.bounds
		bounds_range = bounds[1] - bounds[0]
		dims = np.argsort(bounds_range)[-2:]
		
		uvs = np.zeros((len(vertices), 2))
		uvs[:, 0] = (vertices[:, dims[0]] - bounds[0][dims[0]]) / bounds_range[dims[0]]
		uvs[:, 1] = (vertices[:, dims[1]] - bounds[0][dims[1]]) / bounds_range[dims[1]]
		
		floor_mesh.visual = trimesh.visual.TextureVisuals(
			uv=uvs,
			material=material
		)
	except Exception as e:
		print(f"Failed to load texture: {e}")
		floor_mesh.visual.face_colors = [245, 222, 179, 178]
	
	return floor_mesh

def create_pyrender_scene_from_trimesh(trimesh_scene, bg_color=None):
	pyrender_scene = pyrender.Scene(bg_color=bg_color)
	for node_name in trimesh_scene.graph.nodes_geometry:
		geom_name = trimesh_scene.graph[node_name][1]
		transform = trimesh_scene.graph[node_name][0]
		geom = trimesh_scene.geometry[geom_name]

		try:
			pyrender_mesh = pyrender.Mesh.from_trimesh(geom)
		except ValueError as e:
			if hasattr(geom, 'visual'):
				geom.visual = trimesh.visual.ColorVisuals(geom)
			try:
				pyrender_mesh = pyrender.Mesh.from_trimesh(geom)
			except Exception as e2:
				print(f"Failed to convert {geom_name} even without texture: {e2}")
				continue

		pyrender_scene.add(pyrender_mesh, pose=transform)
	
	return pyrender_scene

def add_objects_to_trimesh_scene(trimesh_scene, objects, show_bboxes=False, show_assets=True, show_assets_voxelized=False, show_bounds=False, bounds_bottom=None, bounds_top=None, voxelized_objects_cache=None):
	for i, obj in enumerate(objects):
		if show_bboxes and obj.get("size") is not None:
			bbox_mesh = create_bbox(size=obj.get("size"), pos=obj.get("pos"), rot=obj.get("rot"))
			trimesh_scene.add_geometry(bbox_mesh)
		if show_assets and not show_assets_voxelized:
			jid = obj["jid"] if obj.get("jid") is not None else obj["sampled_asset_jid"]
			mesh = load_mesh_with_transform(get_pth_mesh(jid), obj.get("pos"), obj.get("rot"), obj.get("scale"))
			trimesh_scene.add_geometry(mesh)
		elif show_assets and show_assets_voxelized:
			if voxelized_objects_cache is not None and i < len(voxelized_objects_cache):
				voxel_mesh = voxelized_objects_cache[i]
				trimesh_scene.add_geometry(voxel_mesh)
			else:
				# Fallback to computing voxelization on-the-fly (original behavior)
				jid = obj["jid"] if obj.get("jid") is not None else obj["sampled_asset_jid"]
				mesh = load_mesh_with_transform(get_pth_mesh(jid), obj.get("pos"), obj.get("rot"), obj.get("scale"))
				voxel_size = 0.05
				print("VOXELIZING")
				if isinstance(mesh, trimesh.Scene):
					mesh = mesh.to_geometry()
				else:
					mesh = mesh
				voxelized = mesh.voxelized(pitch=voxel_size).fill()
				voxel_points = voxelized.points
				voxel_mesh = trimesh.Trimesh()
				for point in voxel_points:
					cube = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
					transform = np.eye(4)
					transform[:3, 3] = point
					cube.apply_transform(transform)
					voxel_mesh = trimesh.util.concatenate([voxel_mesh, cube])
				trimesh_scene.add_geometry(voxel_mesh)

	if show_bounds:
		if bounds_bottom is not None:
			# for each item in bounds_bottom, add a small red cube
			for bound in bounds_bottom:
				bound_mesh = create_bbox(size=[0.1, 0.1, 0.1], pos=bound, rot=None, color=[1.0, 0.0, 0.0, 0.7])
				trimesh_scene.add_geometry(bound_mesh)
		if bounds_top is not None:
			# for each item in bounds_top, add a small red cube
			for bound in bounds_top:
				bound_mesh = create_bbox(size=[0.1, 0.1, 0.1], pos=bound, rot=None, color=[1.0, 0.0, 0.0, 0.7])
				trimesh_scene.add_geometry(bound_mesh)

def render_single_frame(pyrender_scene, resolution, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL, max_attempts=3):
	attempt = 0
	# OpenGL.GL.glGenTextures = patched_glGenTextures
	import os
	os.environ['PYOPENGL_PLATFORM'] = 'egl'
	while attempt < max_attempts:
		renderer = None
		try:
			attempt += 1
			#print("before renderer")
			renderer = pyrender.OffscreenRenderer(*resolution)
			flags = (pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.RGBA)
			color, depth = renderer.render(pyrender_scene, flags=flags)
			#print("after renderer")
			return color
		
		except Exception as e:
			print(f"Render attempt {attempt} failed: {e}")
			traceback.print_exc()
			time.sleep(0.5)
			if attempt >= max_attempts:
				raise
		
		finally:
			if renderer is not None:
				renderer.delete()
	
	raise RuntimeError(f"Failed to render frame after {max_attempts} attempts")

def render_with_retry(pyrender_scene, resolution, pth_output, filename):
	# print("render_with_retry...")
	attempt = 0
	while True:  # Infinite attempts
		renderer = None
		try:
			# print("attempt:", attempt)
			attempt += 1

			# import pyglet
			# pyglet.options['headless'] = True
			
			# ml load py-pyopengl/3.1.5_py39


			### ***

			# import os
			# import pyrender
			# from pyrender.constants import (TARGET_OPEN_GL_MAJOR, TARGET_OPEN_GL_MINOR,
			# 							MIN_OPEN_GL_MAJOR, MIN_OPEN_GL_MINOR)
			# from importlib import import_module
			# pyglet_platform = import_module('pyrender.platforms.pyglet_platform')

			# os.environ['DISPLAY'] = ':9925'
			
			# import pyglet
			# pyglet.options['shadow_window'] = False
			# pyglet.options['headless'] = True
			# pyglet.options['headless_device'] = 0

			# def patched_init_context(self):
			# 	import pyglet
			# 	pyglet.options['shadow_window'] = False
			# 	pyglet.options['debug_x11'] = True  # Enable X11 debugging
			# 	pyglet.options['debug_gl'] = True   # Enable OpenGL debugging
				
			# def patched_init_context(self):
			# 	import pyglet
			# 	pyglet.options['shadow_window'] = False

			# 	try:
			# 		pyglet.lib.x11.xlib.XInitThreads()
			# 	except Exception:
			# 		pass

			# 	self._window = None
			# 	confs = [pyglet.gl.Config(sample_buffers=1, samples=4,
			# 							depth_size=24,
			# 							double_buffer=True,
			# 							major_version=TARGET_OPEN_GL_MAJOR,
			# 							minor_version=TARGET_OPEN_GL_MINOR),
			# 			pyglet.gl.Config(depth_size=24,
			# 							double_buffer=True,
			# 							major_version=TARGET_OPEN_GL_MAJOR,
			# 							minor_version=TARGET_OPEN_GL_MINOR),
			# 			pyglet.gl.Config(sample_buffers=1, samples=4,
			# 							depth_size=24,
			# 							double_buffer=True,
			# 							major_version=MIN_OPEN_GL_MAJOR,
			# 							minor_version=MIN_OPEN_GL_MINOR),
			# 			pyglet.gl.Config(depth_size=24,
			# 							double_buffer=True,
			# 							major_version=MIN_OPEN_GL_MAJOR,
			# 							minor_version=MIN_OPEN_GL_MINOR)]
			# 	error = None
			# 	for conf in confs:
			# 		try:
			# 			self._window = pyglet.window.Window(config=conf, visible=False,
			# 												resizable=False,
			# 												width=1, height=1)
			# 			break
			# 		except pyglet.window.NoSuchConfigException as e:
			# 			error = e
			# 			traceback.print_exc()

			# 	if not self._window:
			# 		raise ValueError(
			# 			'Failed to initialize Pyglet window with an OpenGL >= 3+ '
			# 			'context. If you\'re logged in via SSH, ensure that you\'re '
			# 			'running your script with vglrun (i.e. VirtualGL). The '
			# 			'internal error message was "{}"'.format(error)
			# 		)

			# pyglet_platform.PygletPlatform.init_context = patched_init_context

			### ****
			import os
			os.environ['PYOPENGL_PLATFORM'] = 'egl'
			# os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

			try:
				# print("before offscreen renderer")
				renderer = pyrender.OffscreenRenderer(*resolution)
				# print("after offscreen renderer")
			except Exception as exc:
				print(exc)
				traceback.print_exc()
				# exit()

			# print("OpenGL context:", pyglet.gl.current_context)
			# print("Current OpenGL Platform:", os.environ.get('PYOPENGL_PLATFORM', 'default'))
			# print("Renderer type:", type(renderer._renderer).__name__)
			# print("Platform type:", type(renderer._platform).__name__)
			
			# flags = (pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
			flags = (pyrender.RenderFlags.SHADOWS_DIRECTIONAL)

			# print("Render flags:", flags)
			# print("Ambient light:", pyrender_scene.ambient_light)

			# lights = [node for node in pyrender_scene.nodes if isinstance(node.light, pyrender.DirectionalLight)]
			# for i, light_node in enumerate(lights):
				# print(f"Light {i} intensity:", light_node.light.intensity)
				# print(f"Light {i} color:", light_node.light.color)
				# print(f"Light {i} pose:", light_node.matrix)
			
			color, depth = renderer.render(pyrender_scene, flags=flags)
			
			os.makedirs(pth_output, exist_ok=True)
			output_file = os.path.join(pth_output, filename)

			if ".png" in output_file:
				file_type = 'PNG'
				alpha = (color.sum(axis=2) > 0).astype(np.uint8) * 255
				rgba = np.dstack((color, alpha))
				img = Image.fromarray(rgba, 'RGBA')
			else:
				file_type = 'JPEG'
				img = Image.fromarray(color)

			img.save(output_file, file_type, quality=95)

			pth_file = pth_output / filename
			if pth_file.exists():
				break
		
		except Exception as e:
			print(f"Render attempt {attempt} failed: {e}")
			time.sleep(0.5)  # Small delay between attempts
			
		finally:
			if renderer is not None:
				renderer.delete()

def remove_pyrender_nodes(pyrender_scene):
	# Remove camera and light
	nodes_to_remove = []
	for node in pyrender_scene.nodes:
		if isinstance(node.camera, pyrender.PerspectiveCamera) or isinstance(node.light, pyrender.DirectionalLight):
			nodes_to_remove.append(node)
	
	for node in nodes_to_remove:
		pyrender_scene.remove_node(node)

def render_both_views(pyrender_scene, resolution, pth_output, base_filename, use_dynamic_zoom, camera_height, scene_span):
	# First, collect nodes to remove
	remove_pyrender_nodes(pyrender_scene)
	# Render top view
	camera_pose_top = setup_camera(pyrender_scene, resolution, 'top', use_dynamic_zoom, camera_height, scene_span)
	setup_lighting(pyrender_scene, camera_pose_top)
	render_with_retry(pyrender_scene, resolution, pth_output / "top", f"{base_filename}.jpg")

	remove_pyrender_nodes(pyrender_scene)
	# Render diagonal view
	camera_pose = setup_camera(pyrender_scene, resolution, 'diag', use_dynamic_zoom, camera_height, scene_span)
	setup_lighting(pyrender_scene, camera_pose_top)
	# JPG
	render_with_retry(pyrender_scene, resolution, pth_output / "diag", f"{base_filename}.jpg")
	# PNG
	# render_with_retry(pyrender_scene, resolution, pth_output / "diag", f"{base_filename}.png")

def render_scene_and_export(scene_with_assets, filename, pth_output, resolution=(1024, 1024), show_bboxes=False, show_assets=True, show_assets_voxelized=False, show_bounds=False, use_dynamic_zoom=True, camera_height=None, bg_color=None):
	bounds_bottom = scene_with_assets["bounds_bottom"]
	trimesh_scene, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)

	add_objects_to_trimesh_scene(trimesh_scene, scene_with_assets["objects"], show_bboxes, show_assets, show_assets_voxelized, show_bounds, scene_with_assets.get("bounds_bottom"), scene_with_assets.get("bounds_top"))
	pyrender_scene = create_pyrender_scene_from_trimesh(trimesh_scene, bg_color=bg_color)

	# render both top and diag views
	render_both_views(pyrender_scene, resolution, pth_output, filename, use_dynamic_zoom, camera_height, scene_span)

def render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span, bg_color=None):
	pyrender_scene = create_pyrender_scene_from_trimesh(trimesh_scene, bg_color=bg_color)

	# Add camera and lighting
	camera_pose = setup_camera(pyrender_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
	setup_lighting(pyrender_scene, camera_pose)
	
	return render_single_frame(pyrender_scene, resolution)

def setup_trimesh_scene_with_floor(bounds_bottom):
	trimesh_scene = trimesh.Scene()
	floor_slab = create_floor_slab(bounds_bottom)
	trimesh_scene.add_geometry(floor_slab)

	x_span = np.array(bounds_bottom)[:, 0].max() - np.array(bounds_bottom)[:, 0].min()
	y_span = np.array(bounds_bottom)[:, 1].max() - np.array(bounds_bottom)[:, 1].min()
	z_span = np.array(bounds_bottom)[:, 2].max() - np.array(bounds_bottom)[:, 2].min()
	scene_span = (x_span, y_span, z_span)

	return trimesh_scene, scene_span

def create_progressive_gif(scene_with_assets, filename, pth_output, view_type, resolution=(1024, 1024), use_dynamic_zoom=True, camera_height=None, duration=0.8):
	
	# Create output directory
	gif_output_dir = pth_output / f"{view_type}-gif"
	os.makedirs(gif_output_dir, exist_ok=True)
	
	# Add floor
	bounds_bottom = scene_with_assets["bounds_bottom"]
	trimesh_scene, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
	
	# Collect frames
	frames = []
	
	# First frame with just the floor
	try:
		base_frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
		frames.append(base_frame)
	except Exception as e:
		print(f"Failed to render base frame: {e}")
		return
	
	# Add objects one by one
	for i, obj in enumerate(scene_with_assets["objects"]):
		try:
			# Add this object to the scene
			jid = obj["jid"] if obj.get("jid") is not None else obj["sampled_asset_jid"]
			mesh = load_mesh_with_transform(get_pth_mesh(jid), obj.get("pos"), obj.get("rot"), obj.get("scale"))
			trimesh_scene.add_geometry(mesh)
			
			# Render current state
			frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
			frames.append(frame)
			
		except Exception as e:
			print(f"Failed to add object {i}: {e}")
			traceback.print_exc()
			continue
	
	# Save as GIF if we have at least two frames
	if len(frames) >= 2:
		gif_path = os.path.join(gif_output_dir, f"{filename}.gif")
		durations = [duration*1000] * len(frames)
		imageio.mimsave(gif_path, frames, duration=durations, loop=0)

def render_full_scene_and_export_with_gif(scene_with_assets, filename, pth_output, resolution=(1024, 1024), show_bboxes=False, show_assets=True, show_assets_voxelized=False, show_bounds=False, use_dynamic_zoom=True, camera_height=None, create_gif=True, gif_duration=0.6, show_bboxes_also=False, bg_color=None):
	# render assets only
	render_scene_and_export(scene_with_assets, filename, pth_output, resolution, show_bboxes, show_assets, show_assets_voxelized, show_bounds, use_dynamic_zoom, camera_height, bg_color=bg_color)
	
	# if show_bboxes_also:
		# render_scene_and_export(scene_with_assets, f"{filename}-bboxes", pth_output, resolution, show_bboxes=True, show_assets=False, use_dynamic_zoom=use_dynamic_zoom, camera_height=camera_height)

	if create_gif:
		create_progressive_gif(scene_with_assets, filename, pth_output, "top", resolution, use_dynamic_zoom, camera_height, gif_duration)
		create_progressive_gif(scene_with_assets, filename, pth_output, "diag", resolution, use_dynamic_zoom, camera_height, gif_duration)

def create_instr_before_after_gif(scene_after, filename, pth_output, view_type, resolution=(1024, 1024), use_dynamic_zoom=True, camera_height=None, duration=0.8):
	# Create output directory
	gif_output_dir = pth_output / f"{view_type}-gif"
	os.makedirs(gif_output_dir, exist_ok=True)
	
	# setup scene
	bounds_bottom = scene_after["bounds_bottom"]
	trimesh_scene, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
	
	# Collect frames
	frames = []

	# get frame for "before" scene
	add_objects_to_trimesh_scene(trimesh_scene, scene_after["objects"][:-1])
	before_frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
	frames.append(before_frame)
	
	# get frame for "after" scene
	last_objects = [scene_after["objects"][-1]]
	add_objects_to_trimesh_scene(trimesh_scene, last_objects)
	after_frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
	frames.append(after_frame)

	# Save as GIF with two frames
	if len(frames) == 2:
		gif_path = os.path.join(gif_output_dir, f"{filename}.gif")
		durations = [duration*1000] * len(frames)
		imageio.mimsave(gif_path, frames, duration=durations, loop=0)

def render_frame_at_angle(trimesh_scene, angle_degrees, resolution, camera_height, scene_span, bg_color=None):
	"""
	Renders a single frame of the scene from a specific angle by rotating the scene
	instead of moving the camera.
	
	Args:
		trimesh_scene: The trimesh scene to render
		angle_degrees: Rotation angle in degrees
		resolution: Output resolution (width, height)
		camera_height: Height of camera (used directly)
		scene_span: Size of the scene for calculating distances
		bg_color: Background color
	
	Returns:
		numpy array: The rendered image
	"""
	# Create a copy of the scene to rotate
	scene_copy = copy.deepcopy(trimesh_scene)
	
	# Calculate rotation matrix for the scene
	rotation_matrix = trimesh.transformations.rotation_matrix(
		angle=math.radians(angle_degrees),
		direction=[0, 1, 0],  # Rotate around Y axis
		point=[0, 0, 0]       # Rotate around the origin
	)
	
	# Apply rotation to the entire scene
	scene_copy.apply_transform(rotation_matrix)
	
	# Convert trimesh scene to pyrender scene
	pyrender_scene = create_pyrender_scene_from_trimesh(scene_copy, bg_color=bg_color)

	for node in pyrender_scene.mesh_nodes:
		if node.mesh is not None:
			for primitive in node.mesh.primitives:
				offset = np.eye(4)
				offset[:3, 3] += 0.0001 * np.random.randn(3)
				node.matrix = np.matmul(node.matrix, offset)
	
				if primitive.material is not None:
					primitive.material.baseColorFactor = [*primitive.material.baseColorFactor[:3], 1.0]
					primitive.material.alphaMode = 'OPAQUE'

	camera_pose_top = setup_camera(pyrender_scene, resolution, 'top', False, camera_height, scene_span)
	
	remove_pyrender_nodes(pyrender_scene)

	camera_pose = setup_camera(pyrender_scene, resolution, 'diag', False, camera_height, scene_span)
	setup_lighting(pyrender_scene, camera_pose_top)
	
	# # Add a fixed camera looking at the center of the scene
	# camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
	
	# # Camera position - fixed, using provided height
	# max_span = max(scene_span)
	# camera_distance = max_span * 2.5
	# camera_pos = np.array([0.0, camera_height, camera_distance])  # Camera on Z axis, looking toward origin
	
	# # Calculate view vectors
	# target = np.array([0.0, 0.0, 0.0])  # Look at center
	
	# forward = target - camera_pos
	# forward = forward / np.linalg.norm(forward)
	
	# world_up = np.array([0.0, 1.0, 0.0])
	
	# right = np.cross(forward, world_up)
	# right = right / np.linalg.norm(right)
	
	# up = np.cross(right, forward)
	# up = up / np.linalg.norm(up)
	
	# # Construct camera pose matrix
	# camera_pose = np.eye(4)
	# camera_pose[:3, 0] = right
	# camera_pose[:3, 1] = up
	# camera_pose[:3, 2] = -forward  # Note the negative forward
	# camera_pose[:3, 3] = camera_pos
	
	# # Add camera to scene
	# pyrender_scene.add(camera, pose=camera_pose)
	
	# # Add lighting
	# light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
	# light_pose = camera_pose.copy()
	# light_pose[1, 3] += 3.0  # Move light up relative to camera
	# pyrender_scene.add(light, pose=light_pose)
	
	# # Add ambient light
	# pyrender_scene.ambient_light = np.array([0.6, 0.6, 0.6, 1.0])
	
	# Render the frame
	return render_single_frame(pyrender_scene, resolution, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)

def create_360_video_instr(scene_with_assets, filename, room_type, pth_output, 
						  resolution=(1536, 1024),  # 3:2 aspect ratio
						  camera_height=None, fps=30, video_duration=5.0, 
						  visibility_time=1.0, bg_color=None):
	
	# Create output directory for video
	video_output_dir = pth_output
	os.makedirs(video_output_dir, exist_ok=True)
	
	# Setup scene basics
	bounds_bottom = scene_with_assets["bounds_bottom"]

	# fix flickering for selected samples (weird mesh issue)
	# if room_type in ["bedroom", "all"]:
	# 	for obj in scene_with_assets["objects"]:
	# 		if "lamp" in obj.get("desc", "").lower() or "vase" in obj.get("desc", "").lower():
	# 			print("found obj!")
	# 			obj["pos"][1] -= 0.01
		
	trimesh_scene_before, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
	
	# Add all objects except the last one to the "before" scene
	add_objects_to_trimesh_scene(trimesh_scene_before, scene_with_assets["objects"][:-1])
	
	# Create the "after" scene with all objects
	trimesh_scene_after = copy.deepcopy(trimesh_scene_before)
	add_objects_to_trimesh_scene(trimesh_scene_after, [scene_with_assets["objects"][-1]])
	
	# Calculate frames for a single full rotation
	total_frames = int(fps * video_duration)
	
	# Calculate how many frames to show each state (before/after)
	# Based on visibility_time parameter
	frames_visible = int(fps * visibility_time)
	
	# Prepare video writer
	video_path = os.path.join(video_output_dir, f"{filename}_360.mp4")
	writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
	
	print(f"Creating {total_frames} frames for {video_duration}s 360° instruction video with blinking effect...")
	
	# Generate frames with progress bar
	for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
		# Calculate angle for continuous rotation through entire video
		angle_progress = frame_idx / total_frames
		angle_degrees = angle_progress * 360
		
		# Determine whether to show the "after" scene (with added object)
		# We want to blink between before/after periodically
		cycle_position = frame_idx % (frames_visible * 2)  # Cycle between before and after
		show_after = cycle_position >= frames_visible  # Show "before" first, then "after"
		
		# Select the appropriate scene
		current_scene = trimesh_scene_after if show_after else trimesh_scene_before
		
		# Render the frame from the current angle
		frame = render_frame_at_angle(
			current_scene, 
			angle_degrees, 
			resolution, 
			camera_height, 
			scene_span, 
			bg_color
		)
		
		# Convert from RGB to BGR (OpenCV uses BGR)
		frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		writer.write(frame_bgr)
	
	writer.release()
	print(f"Created 360° video at {video_path}")
	return video_path

def create_360_video_full(scene_with_assets, filename, pth_output, room_type=None,
						  resolution=(1536, 1024),  # 3:2 aspect ratio
						  camera_height=None, fps=30, video_duration=4.0, 
						  step_time=0.5, bg_color=None):
	
	# Create output directory for video
	video_output_dir = pth_output
	os.makedirs(video_output_dir, exist_ok=True)
	
	# Setup scene basics
	bounds_bottom = scene_with_assets["bounds_bottom"]
	trimesh_scene_base, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
	
	# Calculate timing parameters
	num_objects = len(scene_with_assets["objects"])
	objects_per_rotation = int(video_duration / step_time)  # e.g., 8 objects per 4-second rotation
	
	# Calculate minimum rotations needed to show all objects
	min_rotations_needed = math.ceil(num_objects / objects_per_rotation)
	
	# Add one extra rotation at the end to show complete scene
	total_rotations = min_rotations_needed + 1
	total_video_duration = total_rotations * video_duration
	
	# Calculate frames
	total_frames = int(fps * total_video_duration)
	frames_per_step = int(fps * step_time)
	frames_per_rotation = int(fps * video_duration)
	
	print(f"Creating 360° full scene video:")
	print(f"- {num_objects} objects to place")
	print(f"- {objects_per_rotation} objects per {video_duration}s rotation")
	print(f"- {min_rotations_needed} rotations needed for placement")
	print(f"- {total_rotations} total rotations (including final complete scene)")
	print(f"- {total_video_duration}s total duration")
	print(f"- {total_frames} total frames")

	all_objects = copy.deepcopy(scene_with_assets["objects"])
	
	# fix flickering for selected samples (weird mesh issue)
	# if room_type in ["bedroom", "livingroom"]:
	# 	for obj in all_objects:
	# 		if "lamp" in obj.get("desc", "").lower() or "vase" in obj.get("desc", "").lower() or "plant" in obj.get("desc", "").lower():
	# 			print("found obj!")
	# 			obj["pos"][1] -= 0.01
	
	# Prepare video writer
	video_path = os.path.join(video_output_dir, f"{filename}_360.mp4")
	writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
	
	# Generate frames with progress bar
	for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
		# Calculate which rotation we're in
		current_rotation = frame_idx // frames_per_rotation
		frame_in_rotation = frame_idx % frames_per_rotation
		
		# Calculate angle for continuous rotation
		angle_progress = frame_in_rotation / frames_per_rotation
		angle_degrees = angle_progress * 360
		
		# Determine how many objects to show based on time
		if current_rotation < min_rotations_needed:
			# During placement rotations
			objects_shown_from_prev_rotations = current_rotation * objects_per_rotation
			additional_objects_this_rotation = min(
				frame_in_rotation // frames_per_step,
				objects_per_rotation
			)
			total_objects_to_show = min(
				objects_shown_from_prev_rotations + additional_objects_this_rotation,
				num_objects
			)
		else:
			# Final rotation - show all objects
			total_objects_to_show = num_objects
		
		# Create scene with appropriate number of objects
		current_scene = copy.deepcopy(trimesh_scene_base)
		objects_to_add = all_objects[:total_objects_to_show]
		add_objects_to_trimesh_scene(current_scene, objects_to_add)
		
		# Render the frame from the current angle
		frame = render_frame_at_angle(
			current_scene, 
			angle_degrees, 
			resolution, 
			camera_height, 
			scene_span, 
			bg_color
		)
		
		# Convert from RGB to BGR (OpenCV uses BGR)
		frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		writer.write(frame_bgr)
	
	writer.release()
	print(f"Created 360° full scene video at {video_path}")
	return video_path

def render_instr_scene_and_export_with_gif(scene_after, filename, pth_output, resolution=(1024, 1024), show_bboxes=False, show_assets=True, show_assets_voxelized=False, show_bounds=False, use_dynamic_zoom=True, camera_height=None, create_gif=True, gif_duration=0.8, bg_color=None):
	render_scene_and_export(scene_after, filename, pth_output, resolution, show_bboxes, show_assets, show_assets_voxelized, show_bounds, use_dynamic_zoom, camera_height, bg_color=bg_color)
	# render_scene_and_export(scene_after, f"{filename}-bboxes", pth_output, resolution, show_bboxes=True, show_assets=False, use_dynamic_zoom=use_dynamic_zoom, camera_height=camera_height)
	if create_gif:
		create_instr_before_after_gif(scene_after, filename, pth_output, "top", resolution, use_dynamic_zoom, camera_height, gif_duration)
		create_instr_before_after_gif(scene_after, filename, pth_output, "diag", resolution, use_dynamic_zoom, camera_height, gif_duration)

def render_full_scenes_for_room_type(room_type, pth_root, pth_folder_prefix, pth_output):

	folder_name = f"{pth_folder_prefix}-{room_type}"
	pth_output_full = pth_output / folder_name
	remove_and_recreate_folder(pth_output_full)

	# we take the full train split if less than 5K, otherwise we sample 5K
	all_pths = get_pths_dataset_split(room_type, "train")
	if len(all_pths) > 5000:
		all_pths = np.random.choice(all_pths, 5000, replace=False)
	
	# test only
	# all_pths = all_pths[:5]

	cnt = 0
	pbar = tqdm(all_pths)
	for pth in pbar:
		scene = json.load(open(os.path.join(pth_root, pth), "r"))
		scene_id = pth.split(".")[0]
		render_full_scene_and_export_with_gif(scene, filename=scene_id, pth_output=pth_output_full, create_gif=False)
		cnt += 1
		pbar.set_description(f"Rendering scenes (# {cnt})")

	precompute_fid_scores_for_caching(f"{folder_name}-top", str(pth_output_full / "top"))
	precompute_fid_scores_for_caching(f"{folder_name}-diag", str(pth_output_full / "diag"))
	
	print(f"rendered all scenes for room type: {room_type}, total: {cnt}")

def get_assets_from_gt_for_scene(scene, scene_id):
	# for each object in scene, get the asset from full_scene_with_assets via "jid" that matches "desc"
	full_scene_with_assets = json.load(open(f"{os.getenv('PTH_STAGE_2_DEDUP')}/{scene_id}.json"))
	for obj in scene.get("objects"):
		desc = obj.get("desc")
		for asset in full_scene_with_assets.get("objects"):
			if asset.get("desc") == desc:
				obj["jid"] = asset.get("jid")
				break

def render_instr_scenes_for_room_type(room_type, pth_root, pth_folder_prefix, pth_output_base):
	print("=== Starting file reading phase ===")
	
	import gc
	gc.collect()
	
	folder_name = f"{pth_folder_prefix}-{room_type}"
	pth_output_full = pth_output_base / folder_name
	remove_and_recreate_folder(pth_output_full)
	
	dataset_train, _, _ = load_train_val_test_datasets(lambda_instr_exp=None, use_cached_dataset=True, room_type=room_type, do_sanity_check=False, seed=42)

	all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
	all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
	
	# Track failures for reporting
	failed_renders = []

	# Add a buffer to account for potential failures
	np.random.seed(42)
	target_size = min(5000, len(dataset_train))
	dataset_train = dataset_train.select(range(target_size))

	model, tokenizer, max_seq_length = get_model("meta-llama/Llama-3.2-1B-Instruct", use_gpu=True, accelerator=None)
	
	print("=== Starting rendering phase ===")
	cnt = 0
	batch_size = 100
	for i in range(0, len(dataset_train), batch_size):
		batch = dataset_train.select(range(i, min(i + batch_size, len(dataset_train))))
		for sample in tqdm(batch, desc=f"Rendering batch {i//batch_size + 1}/{len(dataset_train)//batch_size}"):
			try:
				
				_, _, _, instr_sample = process_scene_sample(sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False, do_augm=False, do_full_sg_outputs=False)
				# instr_sample = create_instruction_from_scene(sample, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False)
				
				scene_id = sample["pth_orig_file"].split(".")[0]
				scene_after = create_full_scene_from_before_and_added(json.loads(instr_sample.get("sg_input")), json.loads(instr_sample.get("sg_output_add")))
				get_assets_from_gt_for_scene(scene_after, scene_id)

				render_instr_scene_and_export_with_gif(scene_after, filename=scene_id, pth_output=pth_output_full, create_gif=False)

				# Verify both files were actually created
				top_file = pth_output_full / "top" / f"{scene_id}.jpg"
				diag_file = pth_output_full / "diag" / f"{scene_id}.jpg"
				if top_file.exists() and diag_file.exists():
					cnt += 1
				else:
					failed_renders.append((scene_id, "Files not created after render"))
					print(f"Failed to create files for {scene_id} after seemingly successful render")
					if not top_file.exists():
						print(f"Missing top view: {top_file}")
					if not diag_file.exists():
						print(f"Missing diag view: {diag_file}")
				
			except Exception as exc:
				traceback.print_exc()
				print(f"Failed to render {scene_id}: {exc}")
				continue

			gc.collect()
		gc.collect()
	
	print(f"=== Completed rendering phase. Rendered {cnt} scenes ===")
	print(f"Failed to render {len(failed_renders)} scenes")
	
	# Verify output files exist
	top_files = set(os.listdir(pth_output_full / "top"))
	diag_files = set(os.listdir(pth_output_full / "diag"))
	
	print(f"Files in top directory: {len(top_files)}")
	print(f"Files in diag directory: {len(diag_files)}")
	
	# Only proceed with FID computation if we have files
	if len(top_files) > 0 and len(diag_files) > 0:
		precompute_fid_scores_for_caching(f"{folder_name}-top", str(pth_output_full / "top"))
		precompute_fid_scores_for_caching(f"{folder_name}-diag", str(pth_output_full / "diag"))
	
	print(f"Completed processing for room type: {room_type}")

def create_360_video_voxelization(scene_teaser, pth_folder_fig):
	resolution = (1536, 1024)  # 3:2 aspect ratio
	fps = 30
	camera_height = 5.5
	bg_color = np.array([240, 240, 240]) / 255.0
	step_time = 0.8  # Time between object additions/replacements
	still_time = 4.0  # Time to hold still between phases
	
	# Calculate timing parameters
	num_objects = len(scene_teaser["objects"])
	empty_scene_duration = 2.0  # 2 seconds empty scene
	bounds_only_duration = 4.0  # 4 seconds with bounds only
	bbox_placement_duration = num_objects * step_time  # 0.8s per bbox addition
	bbox_still_duration = still_time  # 4s still with all bboxes
	asset_replacement_duration = num_objects * step_time  # 0.8s per bbox->asset replacement
	asset_still_duration = still_time  # 4s still with all assets
	voxel_replacement_duration = num_objects * step_time  # 0.8s per asset->voxel replacement
	
	# Calculate how much extra time needed to complete full rotation
	base_duration = (empty_scene_duration + bounds_only_duration + 
					bbox_placement_duration + bbox_still_duration +
					asset_replacement_duration + asset_still_duration + 
					voxel_replacement_duration)
	
	# Add time to complete the rotation (so we end where we started for looping)
	# We want at least 4 seconds of voxels, plus enough to complete the circle
	voxel_still_minimum = still_time
	total_for_full_rotation = base_duration + voxel_still_minimum
	
	# Calculate how much more time needed to complete exactly one full rotation
	# If we're past 360°, add time until we reach the next full rotation
	extra_time_for_loop = 0
	if total_for_full_rotation % 360 != 0:
		# Add time to reach next "clean" rotation point for seamless looping
		extra_time_for_loop = 1.0  # Add 1 second buffer for clean loop
	
	voxel_still_duration = voxel_still_minimum + extra_time_for_loop
	
	total_video_duration = base_duration + voxel_still_duration
	total_frames = int(fps * total_video_duration)
	
	# Calculate frame ranges for each phase
	empty_frames = int(fps * empty_scene_duration)
	bounds_frames = int(fps * bounds_only_duration)
	bbox_placement_frames = int(fps * bbox_placement_duration)
	bbox_still_frames = int(fps * bbox_still_duration)
	asset_replacement_frames = int(fps * asset_replacement_duration)
	asset_still_frames = int(fps * asset_still_duration)
	voxel_replacement_frames = int(fps * voxel_replacement_duration)
	voxel_still_frames = int(fps * voxel_still_duration)
	frames_per_step = int(fps * step_time)
	
	print(f"Creating voxelization 360° video:")
	print(f"- Phase 1 (Empty scene): {empty_frames} frames ({empty_scene_duration}s)")
	print(f"- Phase 2 (Bounds only): {bounds_frames} frames ({bounds_only_duration}s)")
	print(f"- Phase 3 (Bbox placement): {bbox_placement_frames} frames ({bbox_placement_duration}s)")
	print(f"- Phase 3b (Bbox still): {bbox_still_frames} frames ({bbox_still_duration}s)")
	print(f"- Phase 4 (Asset replacement): {asset_replacement_frames} frames ({asset_replacement_duration}s)")
	print(f"- Phase 4b (Asset still): {asset_still_frames} frames ({asset_still_duration}s)")
	print(f"- Phase 5 (Voxel replacement): {voxel_replacement_frames} frames ({voxel_replacement_duration}s)")
	print(f"- Phase 5b (Voxel still): {voxel_still_frames} frames ({voxel_still_duration}s)")
	print(f"- {num_objects} objects, {step_time}s per step")
	print(f"- Total: {total_frames} frames ({total_video_duration}s)")
	
	# Setup scene basics
	bounds_bottom = scene_teaser["bounds_bottom"]
	bounds_top = scene_teaser["bounds_top"]
	all_objects = copy.deepcopy(scene_teaser["objects"])
	
	# Fix flickering for selected samples (weird mesh issue)
	# for obj in all_objects:
	# 	if "lamp" in obj.get("desc", "").lower() or "plant" in obj.get("desc", "").lower():
	# 		obj["pos"][1] -= 0.01
	
	# Pre-compute voxelized objects for caching (expensive operation)
	print("Pre-computing voxelized objects for caching...")
	voxelized_objects_cache = []
	for i, obj in enumerate(tqdm(all_objects, desc="Voxelizing objects")):
		try:
			jid = obj["jid"] if obj.get("jid") is not None else obj["sampled_asset_jid"]
			mesh = load_mesh_with_transform(get_pth_mesh(jid), obj.get("pos"), obj.get("rot"), obj.get("scale"))
			
			voxel_size = 0.05
			if isinstance(mesh, trimesh.Scene):
				mesh = mesh.to_geometry()
			
			voxelized = mesh.voxelized(pitch=voxel_size).fill()
			voxel_points = voxelized.points
			voxel_mesh = trimesh.Trimesh()
			
			for point in voxel_points:
				cube = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
				transform = np.eye(4)
				transform[:3, 3] = point
				cube.apply_transform(transform)
				voxel_mesh = trimesh.util.concatenate([voxel_mesh, cube])
			
			voxelized_objects_cache.append(voxel_mesh)
		except Exception as e:
			print(f"Failed to voxelize object {i}: {e}")
			# Fallback to empty mesh
			voxelized_objects_cache.append(trimesh.Trimesh())
	
	print(f"Pre-computed {len(voxelized_objects_cache)} voxelized objects")
	
	# Create output directory
	pth_folder_fig = Path("./eval/viz/360videos-voxelization")
	remove_and_recreate_folder(pth_folder_fig)
	
	# Prepare video writer
	video_path = pth_folder_fig / "voxelization_360_demo.mp4"
	writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
	
	# Setup base scene with floor
	bounds_bottom = scene_teaser["bounds_bottom"]
	trimesh_scene_base, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
	
	# Helper function to add objects with mixed visualization (including voxels)
	def add_objects_mixed_visualization(trimesh_scene, objects, bbox_indices, asset_indices, voxel_indices, bounds_bottom=None, bounds_top=None, voxelized_cache=None):
		"""Add objects with some as bboxes, some as assets, and some as voxels"""
		# Add bounds if specified
		if bounds_bottom is not None or bounds_top is not None:
			add_objects_to_trimesh_scene(
				trimesh_scene, [], 
				show_bboxes=False, 
				show_assets=False, 
				show_bounds=True,
				show_assets_voxelized=False,
				bounds_bottom=bounds_bottom,
				bounds_top=bounds_top,
				voxelized_objects_cache=voxelized_cache
			)
		
		# Add bounding boxes for specified indices
		if bbox_indices:
			bbox_objects = [objects[i] for i in bbox_indices]
			add_objects_to_trimesh_scene(
				trimesh_scene, bbox_objects,
				show_bboxes=True,
				show_assets=False,
				show_bounds=False,
				show_assets_voxelized=False,
				bounds_bottom=None,
				bounds_top=None,
				voxelized_objects_cache=voxelized_cache
			)
		
		# Add assets for specified indices
		if asset_indices:
			asset_objects = [objects[i] for i in asset_indices]
			add_objects_to_trimesh_scene(
				trimesh_scene, asset_objects,
				show_bboxes=False,
				show_assets=True,
				show_bounds=False,
				show_assets_voxelized=False,
				bounds_bottom=None,
				bounds_top=None,
				voxelized_objects_cache=voxelized_cache
			)
		
		# Add voxelized assets for specified indices
		if voxel_indices:
			voxel_objects = [objects[i] for i in voxel_indices]
			add_objects_to_trimesh_scene(
				trimesh_scene, voxel_objects,
				show_bboxes=False,
				show_assets=True,
				show_bounds=False,
				show_assets_voxelized=True,
				bounds_bottom=None,
				bounds_top=None,
				voxelized_objects_cache=voxelized_cache
			)
	
	# Generate frames with progress bar
	for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
		# Calculate angle for continuous rotation through entire video
		angle_progress = frame_idx / total_frames
		angle_degrees = angle_progress * 360
		
		# Determine which phase we're in
		if frame_idx < empty_frames:
			# Phase 1: Empty scene (just floor)
			current_scene = copy.deepcopy(trimesh_scene_base)
			
		elif frame_idx < empty_frames + bounds_frames:
			# Phase 2: Bounds only
			current_scene = copy.deepcopy(trimesh_scene_base)
			add_objects_to_trimesh_scene(
				current_scene, [], 
				show_bboxes=False, 
				show_assets=False, 
				show_bounds=True,
				show_assets_voxelized=False,
				bounds_bottom=bounds_bottom,
				bounds_top=bounds_top
			)
			
		elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames:
			# Phase 3: Incremental bbox placement
			placement_frame = frame_idx - empty_frames - bounds_frames
			bboxes_to_show = min(placement_frame // frames_per_step + 1, num_objects)
			
			current_scene = copy.deepcopy(trimesh_scene_base)
			bbox_indices = list(range(bboxes_to_show))
			asset_indices = []
			voxel_indices = []
			
			add_objects_mixed_visualization(
				current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
				bounds_bottom, bounds_top
			)
			
		elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames:
			# Phase 3b: All bboxes still
			current_scene = copy.deepcopy(trimesh_scene_base)
			bbox_indices = list(range(num_objects))
			asset_indices = []
			voxel_indices = []
			
			add_objects_mixed_visualization(
				current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
				bounds_bottom, bounds_top
			)
			
		elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames + asset_replacement_frames:
			# Phase 4: Incremental bbox->asset replacement
			replacement_frame = frame_idx - empty_frames - bounds_frames - bbox_placement_frames - bbox_still_frames
			assets_to_show = min(replacement_frame // frames_per_step + 1, num_objects)
			
			current_scene = copy.deepcopy(trimesh_scene_base)
			bbox_indices = list(range(assets_to_show, num_objects))  # Remaining bboxes
			asset_indices = list(range(assets_to_show))  # Already replaced with assets
			voxel_indices = []
			
			add_objects_mixed_visualization(
				current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
				bounds_bottom, bounds_top, voxelized_objects_cache
			)
			
		elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames + asset_replacement_frames + asset_still_frames:
			# Phase 4b: All assets still
			current_scene = copy.deepcopy(trimesh_scene_base)
			bbox_indices = []
			asset_indices = list(range(num_objects))
			voxel_indices = []
			
			add_objects_mixed_visualization(
				current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
				bounds_bottom, bounds_top, voxelized_objects_cache
			)
			
		elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames + asset_replacement_frames + asset_still_frames + voxel_replacement_frames:
			# Phase 5: Incremental asset->voxel replacement
			replacement_frame = frame_idx - empty_frames - bounds_frames - bbox_placement_frames - bbox_still_frames - asset_replacement_frames - asset_still_frames
			voxels_to_show = min(replacement_frame // frames_per_step + 1, num_objects)
			
			current_scene = copy.deepcopy(trimesh_scene_base)
			bbox_indices = []
			asset_indices = list(range(voxels_to_show, num_objects))  # Remaining assets
			voxel_indices = list(range(voxels_to_show))  # Already replaced with voxels
			
			add_objects_mixed_visualization(
				current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
				bounds_bottom, bounds_top, voxelized_objects_cache
			)
			
		else:
			# Phase 5b: All voxels still (until we complete rotation for looping)
			current_scene = copy.deepcopy(trimesh_scene_base)
			bbox_indices = []
			asset_indices = []
			voxel_indices = list(range(num_objects))
			
			add_objects_mixed_visualization(
				current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
				bounds_bottom, bounds_top, voxelized_objects_cache
			)
		
		# Render the frame from the current angle
		frame = render_frame_at_angle(
			current_scene, 
			angle_degrees, 
			resolution, 
			camera_height, 
			scene_span, 
			bg_color
		)
		
		# Convert from RGB to BGR (OpenCV uses BGR)
		frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		writer.write(frame_bgr)
	
	writer.release()
	print(f"Created voxelization demonstration video at {video_path}")
	return video_path

def create_360_videos_assets(scene_example, camera_height, pth_folder_fig):
    # Setup video parameters
    resolution = (1536, 1024)  # 3:2 aspect ratio
    fps = 30
    camera_height = 5.5
    bg_color = np.array([240, 240, 240]) / 255.0
    step_time = 0.8  # Time per asset sample
    
    # Calculate timing parameters
    scene_before_duration = 4.0  # 4 seconds showing scene before
    bbox_duration = 4.0  # 4 seconds showing blue bounding box
    num_asset_samples = 10  # Number of different assets to sample
    asset_sampling_duration = num_asset_samples * step_time  # 0.8s per asset sample
    
    total_video_duration = scene_before_duration + bbox_duration + asset_sampling_duration
    total_frames = int(fps * total_video_duration)
    
    # Calculate frame ranges for each phase
    before_frames = int(fps * scene_before_duration)
    bbox_frames = int(fps * bbox_duration)
    sampling_frames = int(fps * asset_sampling_duration)
    frames_per_step = int(fps * step_time)
    
    print(f"Creating asset sampling 360° video:")
    print(f"- Phase 1 (Scene before): {before_frames} frames ({scene_before_duration}s)")
    print(f"- Phase 2 (Blue bbox): {bbox_frames} frames ({bbox_duration}s)")
    print(f"- Phase 3 (Asset sampling): {sampling_frames} frames ({asset_sampling_duration}s)")
    print(f"- {num_asset_samples} asset samples, {step_time}s per sample")
    print(f"- Total: {total_frames} frames ({total_video_duration}s)")
    print(f"- Total rotations: {total_video_duration / 8.0:.1f} (2 full rotations)")
    
    # Extract scene components
    bounds_bottom = scene_example["bounds_bottom"]
    bounds_top = scene_example["bounds_top"]
    all_objects = copy.deepcopy(scene_example["objects"])
    
    # Fix flickering for selected samples (weird mesh issue)
    # for obj in all_objects:
    #     if "lamp" in obj.get("desc", "").lower() or "plant" in obj.get("desc", "").lower() or "vase" in obj.get("desc", "").lower():
    #         obj["pos"][1] -= 0.01
    
    # Separate the scene before (all objects except last) and the target object
    scene_before_objects = all_objects[:-1] if len(all_objects) > 0 else []
    target_object = all_objects[-1] if len(all_objects) > 0 else None
    
    if target_object is None:
        print("Error: No target object found for asset sampling")
        return
    
    # Initialize sampling engine
    sampling_engine = AssetRetrievalModule(
		lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, 
        asset_size_threshold=0.5, rand_seed=1234, do_print=False
    )
    
    # Pre-sample different assets for the target object
    print("Pre-sampling different assets for target object...")
    sampled_assets = []
    for i in tqdm(range(num_asset_samples), desc="Sampling assets"):
        # Create a temporary scene with the target object
        temp_scene = {
            "room_type": scene_example["room_type"],
            "bounds_bottom": bounds_bottom,
            "bounds_top": bounds_top,
            "objects": scene_before_objects + [copy.deepcopy(target_object)]
        }
        
        # Sample a new asset for the last object
        try:
            sampled_scene = sampling_engine.sample_last_asset(temp_scene, is_greedy_sampling=False)
            sampled_target_object = sampled_scene["objects"][-1]
            sampled_assets.append(sampled_target_object)
        except Exception as e:
            print(f"Failed to sample asset {i}: {e}")
            # Fallback to original object
            sampled_assets.append(copy.deepcopy(target_object))
    
    print(f"Pre-sampled {len(sampled_assets)} different assets")
    
    # Prepare video writer
    video_path = pth_folder_fig / "asset_sampling_360_demo.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
    
    # Setup base scene with floor
    trimesh_scene_base, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
    
    # Generate frames with progress bar
    for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
        # Calculate angle for continuous rotation through entire video
        angle_progress = frame_idx / total_frames
        angle_degrees = angle_progress * 360
        
        # Determine which phase we're in
        if frame_idx < before_frames:
            # Phase 1: Scene before (without target object)
            current_scene = copy.deepcopy(trimesh_scene_base)
            if scene_before_objects:
                add_objects_to_trimesh_scene(
                    current_scene, scene_before_objects,
                    show_bboxes=False,
                    show_assets=True,
                    show_bounds=False,
                    show_assets_voxelized=False,
                    bounds_bottom=None,
                    bounds_top=None
                )
            
        elif frame_idx < before_frames + bbox_frames:
            # Phase 2: Scene before + blue bounding box for target object
            current_scene = copy.deepcopy(trimesh_scene_base)
            
            # Add scene before objects
            if scene_before_objects:
                add_objects_to_trimesh_scene(
                    current_scene, scene_before_objects,
                    show_bboxes=False,
                    show_assets=True,
                    show_bounds=False,
                    show_assets_voxelized=False,
                    bounds_bottom=None,
                    bounds_top=None
                )
            
            # Add blue bounding box for target object
            add_objects_to_trimesh_scene(
                current_scene, [target_object],
                show_bboxes=True,
                show_assets=False,
                show_bounds=False,
                show_assets_voxelized=False,
                bounds_bottom=None,
                bounds_top=None
            )
            
        else:
            # Phase 3: Asset sampling - show different sampled assets
            sampling_frame = frame_idx - before_frames - bbox_frames
            current_asset_index = min(sampling_frame // frames_per_step, num_asset_samples - 1)
            
            current_scene = copy.deepcopy(trimesh_scene_base)
            
            # Add scene before objects
            if scene_before_objects:
                add_objects_to_trimesh_scene(
                    current_scene, scene_before_objects,
                    show_bboxes=False,
                    show_assets=True,
                    show_bounds=False,
                    show_assets_voxelized=False,
                    bounds_bottom=None,
                    bounds_top=None
                )
            
            # Add current sampled asset
            current_sampled_object = sampled_assets[current_asset_index]
            add_objects_to_trimesh_scene(
                current_scene, [current_sampled_object],
                show_bboxes=False,
                show_assets=True,
                show_bounds=False,
                show_assets_voxelized=False,
                bounds_bottom=None,
                bounds_top=None
            )
        
        # Render the frame from the current angle
        frame = render_frame_at_angle(
            current_scene, 
            angle_degrees, 
            resolution, 
            camera_height, 
            scene_span, 
            bg_color
        )
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    print(f"Created asset sampling demonstration video at {video_path}")
    return video_path

def run_viz_for_full_training_dataset():

	# load_dotenv(".env.local")
	load_dotenv(".env.stanley")
	# load_dotenv(".env.sherlock")

	# dataset_train, dataset_val, dataset_test = load_train_val_test_datasets()

	# sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=1234, do_print=False)

	# pth_output = Path("./eval/viz/3d-front-train/")
	# remove_and_recreate_folder(pth_output)
	# for idx in tqdm(range(len(dataset_train))):
	#     sample = dataset_train.select([idx])[0]
	#     scene = json.loads(sample.get("sg_input"))
	#     scene_with_assets = sampling_engine.sample_all_assets(scene)
	#     render_scene_and_export(scene_with_assets, idx, pth_output=pth_output)

	# pth_output = Path("./eval/viz/3d-front-train-full-scenes-v2/")
	# remove_and_recreate_folder(pth_output)
	# max_obj_cnt = {}
	# for idx in tqdm(range(len(dataset_train))):
	#     sample = dataset_train.select([idx])[0]
	#     scene = json.loads(sample.get("sg_output"))
	#     scene_id = sample.get("pth_orig_file").split("/")[-2]
	#     n_objects = len(scene.get("objects"))
	#     if max_obj_cnt.get(scene_id) is not None and n_objects <= max_obj_cnt[scene_id]:
	#         print("skipping scene as not more objects")
	#         continue
	#     max_obj_cnt[scene_id] = n_objects
	#     scene_with_assets = sampling_engine.sample_all_assets(scene)
	#     render_scene_and_export(scene_with_assets, scene_id, pth_output=pth_output)

	pth_output_base = Path(os.getenv("PTH_EVAL_VIZ_CACHE"))
	pth_root = os.getenv("PTH_STAGE_2_DEDUP")

	# render just a single image into PTH_EVAL_VIZ_CACHE
	# get random scene from PTH_STAGE_2_DEDUP if json file
	# all_pths = [f for f in os.listdir(pth_root) if f.endswith(".json")]
	# pth = all_pths[0]
	# pth = "9bf7779c-3afd-474d-8343-05df08fda70c-6838264d-6da5-4aae-bc11-b539d0042e14.json"
	# scene_id = pth.split(".")[0]
	# scene = json.load(open(os.path.join(pth_root, pth), "r"))
	# render_scene_and_export(scene, filename=scene_id, pth_output=pth_output_base)

	# render full scenes
	# pth_folder_prefix = "3d-front-train-full-scenes"
	# render_full_scenes_for_room_type("bedroom", pth_root, pth_folder_prefix, pth_output_base)
	# render_full_scenes_for_room_type("livingroom", pth_root, pth_folder_prefix, pth_output_base)
	# render_full_scenes_for_room_type("all", pth_root, pth_folder_prefix, pth_output_base)

	# # render instr scenes
	pth_folder_prefix = "3d-front-train-instr-scenes"
	render_instr_scenes_for_room_type("bedroom", pth_root, pth_folder_prefix, pth_output_base)
	render_instr_scenes_for_room_type("livingroom", pth_root, pth_folder_prefix, pth_output_base)
	render_instr_scenes_for_room_type("all", pth_root, pth_folder_prefix, pth_output_base)
		
if __name__ == "__main__":

	# load_dotenv(".env.local")
	load_dotenv(".env.stanley")
	
	# run_viz_for_full_training_dataset()
	# xvfb-run -a python src/viz.py

	# metrics_raw = json.load(open("/home/martinbucher/git/stan-24-sgllm/eval/metrics-raw/eval_samples_respace_instr_bedroom_qwen1.5B_raw.json"))
	# metrics_raw = json.load(open("/home/martinbucher/git/stan-24-sgllm/eval/metrics-raw/eval_samples_respace_instr_livingroom_qwen1.5B_raw.json"))
	# for seed in range(3):
	# 	for i, elem in enumerate(metrics_raw[seed]):
	# 		if elem.get("txt_pms_score") == float('inf') or elem.get("txt_pms_score") is float("nan") or elem.get("txt_pms_score") is None or not isinstance(elem.get("txt_pms_score"), float):
	# 			print(seed, i)
	# 			break
