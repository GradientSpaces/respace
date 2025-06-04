import os
import numpy as np
import trimesh
import pdb
import matplotlib.pyplot as plt
import os
import glob
from shapely.geometry import Point, Polygon
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

def load_ply_files(input_folder):
	"""Load all PLY files from the folder."""
	meshes = []
	
	for file_name in os.listdir(input_folder):
	# for file_name in ["floor.ply"]:
	#for file_name in ["wall_01.ply", "floor.ply", ]: # os.listdir(input_folder):
		if file_name.endswith('.ply') and not file_name.startswith(".") and "_mesh.ply" not in file_name:
			file_path = os.path.join(input_folder, file_name)
			# print(f"Loading {file_path}")
			try:
				mesh = trimesh.load(file_path)
			except Exception as exc:
				print(exc)
				# exit()
			meshes.append(mesh)
	
	return meshes

def plot_mesh_3d(vertices, is_xyz=False):
	"""Visualize the mesh using matplotlib."""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	if is_xyz:
		ax.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], c='b', marker='o')
	else:
		ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o')
	plt.show()

def plot_mesh_2d(vertices, curr_point=None, draw_edges=False):
	fig, ax = plt.subplots()
	ax.scatter(vertices[:, 0], vertices[:, 1], c='b', marker='o')

	if curr_point is not None:
		ax.scatter(curr_point[0], curr_point[1], c='r', marker='o')

	if draw_edges:
		for i in range(len(vertices)): # 14 = 0 ... 13
			x1, y1 = vertices[i]
			x2, y2 = vertices[i-1]
			ax.plot([x1, x2], [y1, y2], 'r--')  # Red dashed line for edges
		
	plt.show()

def save_as_ply(vertices, output_file):
	"""Save vertices and faces as a PLY file."""
	#vertices = np.array(vertices)
	#faces = np.array(faces)
	mesh = trimesh.Trimesh(vertices=vertices, faces=[])
	mesh.export(output_file)

def axis_variance(verts):
	return np.var(verts[:, 0]), np.var(verts[:, 1]), np.var(verts[:, 2])

def get_same_ys(points, curr_point, direction):
	# print("get same ys: ", curr_point, points)
	y_same = points[points[:, 1] == curr_point[1]]
	y_same = y_same[np.argsort(y_same[:, 0]), :]

	if direction == "w" or direction == "s":
		y_same = y_same[::-1]

	idx = np.where(y_same[:, 0] == curr_point[0])[0]
	
	if IS_DEBUG:
		print("y idx (which x)", idx)

	return y_same, idx[0]

def get_same_xs(points, curr_point, direction):
	# print("get same xs: ", curr_point, points)
	x_same = points[points[:, 0] == curr_point[0]]
	x_same = x_same[np.argsort(x_same[:, 1]), :]

	if direction == "s" or direction == "w":
		x_same = x_same[::-1]

	idx = np.where(x_same[:, 1] == curr_point[1])[0]
	
	if IS_DEBUG:
		print("x idx (which y)", idx)

	return x_same, idx[0]

def is_corner_point(points, curr_point, direction):

	if IS_DEBUG:
		print("is_corner_point? ", curr_point)
		plot_mesh_2d(points, curr_point=curr_point)

	if direction == "n" or direction == "s":
		x_same, x_idx = get_same_xs(points, curr_point, direction)

		if x_idx == (x_same.shape[0] - 1): # last point at the end of y-axis (top for "n" / bottom for "s")
			is_corner_point = True
		else:
			y_same, y_idx = get_same_ys(points, curr_point, direction)

			if direction == "n" and np.min(y_same[:, 0]) < curr_point[0]: # more points to the left, assume it's a corner
				is_corner_point = True
			
			elif direction == "s" and np.max(y_same[:, 0]) > curr_point[0]: # more points to the right, assume it's a corner
				is_corner_point = True
			
			elif y_idx == (y_same.shape[0] - 1):
				if direction == "s" and y_same.shape[0] > 1: # if moving south and there is stuff to the right, we need to change dir
					is_corner_point = True
				else:
					is_corner_point = False

			elif y_idx == 0: # first elem on x-axis (at the "end")
				if direction == "n" and np.max(x_same[:, 1]) > curr_point[1]: # more points up north, follow dir
					is_corner_point = False
				elif direction == "s" and np.min(x_same[:, 1]) < curr_point[1]: # more points south, follow dir
					is_corner_point = False
				else:
					is_corner_point = True

			else:
				is_corner_point = False

	elif direction == "e" or direction == "w":
		y_same, y_idx = get_same_ys(points, curr_point, direction)
		
		if y_idx == (y_same.shape[0] - 1): # last point on x-axis
			is_corner_point = True
		else:
			x_same, x_idx = get_same_xs(points, curr_point, direction)

			if direction == "e" and np.max(x_same[:, 1]) > curr_point[1]: # more points up north, change dir
				is_corner_point = True

			elif direction == "w" and np.min(x_same[:, 1]) < curr_point[1]: # more points south, change dir
				is_corner_point = True
			
			elif x_idx == (x_same.shape[0] - 1):
				if direction == "e":   
					is_corner_point = False # we are at the "top" so it's an edge
				else: 
					is_corner_point = False # last point on y-axis, but we are heading west, so it's an edge

			elif x_idx == 0:
				if direction == "e" and np.max(x_same[:, 1]) <= curr_point[1]:
					is_corner_point = False
				elif direction == "e" and np.max(x_same[:, 1]) > curr_point[1]:
					is_corner_point = True
				else:
					is_corner_point = False
			
			else:
				is_corner_point = True

	if IS_DEBUG:
		print(f"=== {is_corner_point} ===")
		print("")

	return is_corner_point

def extract_corners(points, resolution=0.01):

	points = np.unique(points, axis=0)

	if IS_DEBUG:
		print("points:", points.shape)
		print(points)

	xs_min = points[points[:, 0] == np.min(points[:, 0])]
	start_x, start_y = xs_min[np.argmin(xs_min[:, 1]), :]

	direction='n'
	curr_point = (start_x, start_y)

	orig_points = points.copy()
	corners = []

	while True:

		if direction == 'n':
			x_same, idx = get_same_xs(points, curr_point, direction)
			idx += 1
			while is_corner_point(points, x_same[idx, :], direction) is False:
				idx += 1
			curr_point = (x_same[idx, 0], x_same[idx, 1])
				
			y_same, _ = get_same_ys(points, curr_point, direction)
			if y_same.shape[0] > 1 and np.min([y_same[:, 0]]) < curr_point[0]:
				direction = "w"
			else:
				direction = "e"

		elif direction == 'e':
			y_same, idx = get_same_ys(points, curr_point, direction)
			idx += 1
			while is_corner_point(points, y_same[idx, :], direction) is False:
				idx += 1
			curr_point = (y_same[idx, 0], y_same[idx, 1])

			# if there are points north go north, else go south
			x_same, _ = get_same_xs(points, curr_point, direction)
			if x_same.shape[0] > 1 and np.max([x_same[:, 1]]) > curr_point[1]:
				direction = "n"
			else:
				direction = "s"

		elif direction == "s":
			x_same, idx = get_same_xs(points, curr_point, direction)
			idx += 1
			while is_corner_point(points, x_same[idx, :], direction) is False:
				idx += 1
			curr_point = (x_same[idx, 0], x_same[idx, 1])

			y_same, _ = get_same_ys(points, curr_point, direction)
			if y_same.shape[0] > 1 and np.max([y_same[:, 0]]) > curr_point[0]:
				direction = "e"
			else:
				direction = "w"

		elif direction == "w":
			y_same, idx = get_same_ys(points, curr_point, direction)
			idx += 1
			while is_corner_point(points, y_same[idx, :], direction) is False:
				idx += 1
			curr_point = (y_same[idx, 0], y_same[idx, 1])

			x_same, idx = get_same_xs(points, curr_point, direction)
			if x_same.shape[0] > 1 and (np.min([x_same[:, 1]]) < curr_point[1]):
				direction = "s"
			else:
				direction = "n"

		if IS_DEBUG:		
			print("")
			print("add corner:", curr_point)
			print("new dir:", direction)
			print("")

		corners.append(curr_point)

		# we finish if we arrive back at the starting point
		if curr_point[0] == start_x and curr_point[1] == start_y:
			break

	corners = np.array(corners)

	if IS_DEBUG:
		print("corners:", corners.shape)
		print(corners)
		plot_mesh_2d(corners, draw_edges=True)

	return corners


def simplify_geometry(points_3d):

	points_3d = np.round(points_3d, decimals=1)

	# raw data is in XZY, not XYZ
	points_2d = points_3d[:, [0, 1]].copy()

	corners_2d = extract_corners(points_2d)

	# put back into 3d coords
	tmp_3d = np.zeros((corners_2d.shape[0], 3))
	tmp_3d[:, [0, 1]] = corners_2d
	tmp_3d[:, 2] = points_3d[0, 2]
	points_3d = tmp_3d

	# swap axis: XZY => XYZ
	tmp_3d_xyz = points_3d.copy()
	tmp_3d_xyz[:, 1] = points_3d[:, 2]
	tmp_3d_xyz[:, 2] = points_3d[:, 1]

	# plot_mesh_3d(tmp_3d_xyz, is_xyz=True)

	return tmp_3d_xyz


def extract_all_corners(all_points):

	y_top = np.max(all_points[:, 2])
	y_bottom = np.min(all_points[:, 2])
	
	# all_points_top = all_points[all_points[:, 2] == y_top, :]
	# corners_top = simplify_geometry(all_points_top)

	all_points_bottom = all_points[all_points[:, 2] == y_bottom, :]
	corners_bottom = simplify_geometry(all_points_bottom)

	corners_top = corners_bottom.copy()
	corners_top[:, 1] = y_top

	if IS_DEBUG:
		plot_mesh_2d(corners_bottom[:, [0, 2]])

	bounds = {
		"bounds_top": corners_top,
		"bounds_bottom": corners_bottom
	}

	return bounds

def add_room_to_blacklist(input_folder):
	room_id = input_folder.split("/")[-1]
	with open(os.getenv("PTH_INVALID_ROOMS"), "a") as fp:
		fp.write(room_id + "\n")

def save_corners(input_folder, cnt):
	all_points = []
	meshes = load_ply_files(input_folder)
	for mesh in meshes:	
		all_points.append(mesh.vertices)
	
	all_points = np.concatenate(all_points)
	all_points = np.unique(all_points, axis=0)

	if IS_DEBUG:
		plot_mesh_3d(all_points, is_xyz=False)

	try:
		all_bounds_dict = extract_all_corners(all_points)
		
		all_points_bottom = all_bounds_dict.get("bounds_bottom")

		# sanity check if 2D polygon can be built
		polygon_area = Polygon(np.array(all_points_bottom)[:, [0, 2]].tolist()).area

		# x = all_points_bottom[:, 0]
		# z = all_points_bottom[:, 2]

		# if x.shape[0] > 10 or x.shape[0] < 4 or (np.max(x) - np.min(x)) > 10.0 or (np.max(z) - np.min(z)) > 10.0: 
		# if x.shape[0] > 10 or x.shape[0] < 4 or (np.max(x) - np.min(x)) > 10.0 or (np.max(z) - np.min(z)) > 10.0:
			# if IS_DEBUG:
				# print("⛔️ NOT VALID")
			# add_room_to_blacklist(input_folder)
		# else:
			# if IS_DEBUG:
				# print("✅ VALID")
				
		with open(f'{input_folder}/room_vertices_simplified.pickle', 'wb') as fp:
			pickle.dump(all_bounds_dict, fp)
		cnt += 1

		return cnt

	except IndexError as exc:
		print(exc)
		add_room_to_blacklist(input_folder)
		print(cnt)
		return cnt

	except Exception as exc:
		print(exc)
		print(type(exc))
		print(cnt)
		return cnt


def is_valid_room(room_id, invalid_rooms):
	room_id = room_id.split("/")[-1]
	if room_id in invalid_rooms:
		return False
	# for word in ["bedroom", "livingroom", "diningroom", "library"]:
	# 	if word in room_id.lower():
	# 		return True
	return True


if __name__ == "__main__":

	load_dotenv(".env.local")

	with open(os.getenv("PTH_INVALID_ROOMS")) as fp:
		invalid_rooms = [line.rstrip() for line in fp]

	# RUN ONE ROOM
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/83b4b802-7830-451f-8242-2981799aab61/Bedroom-308"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/5c337554-ee3f-4c1b-80c0-f8ee49d87e8d/LivingDiningRoom-16531"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/1e15c70d-e32b-4822-b9fa-c81b53a45969/LivingRoom-18069"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/3c3866c8-a167-4e66-8b73-ee663bd79f07/LivingDiningRoom-515"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/9d7a01d8-d1a0-4e8f-89b2-70933f44fa46/LivingDiningRoom-1105"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/a0b72140-8c42-4d52-858b-c4046f6655f4/Bedroom-9106"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/c25d7531-99de-4709-94ce-a22e01740e09/LivingRoom-4200"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/fdda8473-43ac-4012-9735-6b9ad6109b7d/MasterBedroom-30069"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/b5a27a58-2a2f-4c97-b182-7d68c63b2d6c/LivingRoom-41245"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/fdda8473-43ac-4012-9735-6b9ad6109b7d/MasterBedroom-30069"
	# folder = "/Volumes/apollo11/data/front3d-room_meshes/fed8960a-081d-4c4c-94dd-1255c2a21297/Bedroom-10773"
	# IS_DEBUG = True
	# cnt = 0
	# print("doing folder: ", folder)
	# output_file = f"{folder}/simplified_mesh.ply"
	# if is_valid_room(folder, invalid_rooms):
	# 	save_corners(folder, output_file, cnt)
	# else:
	# 	print("SKIPPING ROOM WITH ID:", folder)
	# exit()

	# RUN SCENE (all rooms)
	# scene_id = "d9853b19-5834-49f5-8f78-9975aa5f5575"
	# scene_id = "296aef4e-47fa-447a-b1a6-80d64e5c07d0"
	# scene_id = "103cce55-24d5-4c71-9856-156962e30511"
	# scene_id = "0c1f82d4-6aa2-401f-ae0d-f053a024aa3b"
	# IS_DEBUG = True
	# cnt = 0
	# root_folder = f"/Volumes/apollo11/data/front3d-room_meshes/{scene_id}"
	# for folder in tqdm([f.path for f in os.scandir(root_folder) if f.is_dir()]):
	# 	output_file = f"{folder}/simplified_mesh.ply"
	# 	if is_valid_room(folder, invalid_rooms):
	# 		print("doing folder: ", folder)
	# 		cnt = save_corners(folder, output_file, cnt)
	# exit()

	# # RUN ALL
	IS_DEBUG = False
	cnt = 0

	# sanity check
	# all_pths = [os.path.normpath(f) for f in glob.glob(f"{root_folder}/*/*/") if os.path.isdir(f)]
	# while True:
	# 	try:
	# 		pth = np.random.choice(all_pths)
	# 		corners = pd.read_pickle(f"{pth}/room_vertices_simplified.pickle").get("bounds_bottom")
	# 		print("corners:", corners.shape)
	# 		plot_mesh_2d(corners[:, [0, 2]], draw_edges=True)
	# 	except Exception as exc:
	# 		print(exc)
	# 		continue

	all_folders = [f.path for f in os.scandir(os.getenv("PTH_3DFRONT_BOUNDS")) if f.is_dir()]

	# all_folders = [ os.path.join(os.getenv("PTH_3DFRONT_BOUNDS"), "6b774494-78d2-4def-a1df-24e4d907e796") ]
	# all_folders = [ os.path.join(os.getenv("PTH_3DFRONT_BOUNDS"), "0a9c667d-033d-448c-b17c-dc55e6d3c386") ]

	for folder in tqdm(all_folders):
		for subfolder in [f.path for f in os.scandir(folder) if f.is_dir()]:
			if is_valid_room(subfolder, invalid_rooms):
				print(f"doing folder: {subfolder} (# {cnt})")
				cnt = save_corners(subfolder, cnt)
	print(f"total rooms: {cnt}")
