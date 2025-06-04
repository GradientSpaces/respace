import json
import os
from tqdm import tqdm
import uuid
import shutil
import pdb
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, Delaunay
import scipy
import pandas as pd
from shapely.geometry import Polygon
from dotenv import load_dotenv
import copy

# pth_scene = "/Volumes/apollo11/data/3D-FRONT/5c0a1757-e14e-4901-a3a3-498537689821.json"
# pth_scene = "/Volumes/apollo11/data/3D-FRONT/8ad8bd52-f739-4021-8ad9-9e02f2cb7456.json"


def shift_scene_by_center(scene_subset):

	# all_bboxes_min = np.min(all_bboxes, axis=0).tolist()
	# all_bboxes_center = all_bboxes_min + np.divide((np.max(all_bboxes, axis=0) - all_bboxes_min), 2)

	# we assume the center is always at y=0 and only shift scene on XZ plane
	if scene_subset.get("bounds_bottom") is None:
		return False

	bounds = np.array(scene_subset.get("bounds_bottom"))
	# bounds_xz = np.array(bounds)[:, [0, 2]]
	bounds_min_xz = np.min(bounds, axis=0)
	bounds_max_xz = np.max(bounds, axis=0)
	bounds_center_xz = bounds_min_xz + (bounds_max_xz - bounds_min_xz) / 2.0
	center_3d = np.array([bounds_center_xz[0], 0, bounds_center_xz[2]])
	
	#print(center_3d)

	#else:
	# all_bboxes = []
	# for obj in scene_subset.get("objects"):
	# 	all_bboxes.append(obj.get("pos"))
	# all_bboxes = np.array(all_bboxes)
	# all_bboxes_min_xz = np.min(all_bboxes[:, [0, 2]], axis=0)
	# all_bboxes_max_xz = np.max(all_bboxes[:, [0, 2]], axis=0)
	# all_bboxes_center_xz = all_bboxes_min_xz + (all_bboxes_max_xz - all_bboxes_min_xz) / 2.0
	# center_3d = np.array([all_bboxes_center_xz[0], 0, all_bboxes_center_xz[1]])
	
	#print(center_3d)
	#exit()

	if np.any(np.isnan(center_3d)):
		return False
	else:
		for obj in scene_subset.get("objects"):
			obj_pos_shifted = np.array(obj["pos"]) - center_3d
			obj["pos"] = [ round(elem, 2) for elem in obj_pos_shifted.tolist() ]

		if scene_subset.get("walls") is not None:
			for wall in scene_subset.get("walls"):
				for i in range(int(len(wall.get("xyz"))/3)):
					# print(wall.get("xyz")[3*i], center_3d[0], type(wall.get("xyz")[3*i]), type(center_3d[0]))
					wall.get("xyz")[3*i] = float(wall.get("xyz")[3*i]) - center_3d[0]
					wall.get("xyz")[3*i + 2] = float(wall.get("xyz")[3*i + 2]) - center_3d[2]

		# if scene_subset.get("windows") is not None:
		# 	for window in scene_subset.get("windows"):
		# 		for i in range(int(len(window.get("points")))):
		# 			window.get("points")[i][0] = window.get("points")[i][0] - center_3d[0]
		# 			window.get("points")[i][2] = window.get("points")[i][2] - center_3d[2]

		if scene_subset.get("bounds_bottom") is not None:
			for i, bound in enumerate(scene_subset.get("bounds_bottom")):
				#print(bound)
				bound_shifted = np.array(bound) - center_3d
				#print(bound_shifted)
				scene_subset["bounds_bottom"][i] = [ round(elem, 2) for elem in bound_shifted.tolist() ]

		if scene_subset.get("bounds_top") is not None:
			for i, bound in enumerate(scene_subset.get("bounds_top")):
				#print(bound)
				bound_shifted = np.array(bound) - center_3d
				#print(bound_shifted)
				scene_subset["bounds_top"][i] = [ round(elem, 2) for elem in bound_shifted.tolist() ]

		return True

def save_file(scene_subset, idx=None):

	# pth_file = f"{os.getenv('PTH_STAGE_1')}/{scene_subset.get('orig_scene_uid')}.json" if DO_FULL_SCENES else f"{os.getenv('PTH_STAGE_1')}/{scene_subset.get('orig_scene_uid')}-{scene_subset.get('uid')}.json"
	
	# pth_file = f"{os.getenv('PTH_STAGE_1')}/{scene_subset.get('orig_scene_uid')}-{scene_subset.get('uid')}.json"
	
	pth_file = f"/Users/mnbucher/Downloads/test-doors-windows/{scene_subset.get('orig_scene_uid')}-{idx}.json"
	
	# pth_file = f"{PTH_TARGET}/example.json"

	with open(pth_file, "w") as write_file:
		json.dump(scene_subset, write_file, indent=4)

def compute_bounding_box_sizes(scene):
	if isinstance(scene, trimesh.Scene):
		all_bounds = np.array([geom.bounds for geom in scene.geometry.values()])
		min_bounds = np.min(all_bounds[:, 0, :], axis=0)
		max_bounds = np.max(all_bounds[:, 1, :], axis=0)
	elif isinstance(scene, trimesh.Trimesh):
		min_bounds, max_bounds = scene.bounds
	else:
		raise ValueError("The input scene is neither a trimesh.Scene nor a trimesh.Trimesh.")
	
	bbox_size = (max_bounds - min_bounds).tolist()

	bbox_size = [ round(elem, 2) for elem in bbox_size ]

	return bbox_size

def scale_asset(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id):

	print(f"scaling asset with id {pth_scaled_asset_id}")

	mesh = trimesh.load(f"{pth_orig_asset_id}/raw_model.glb")
	scale_matrix = trimesh.transformations.compose_matrix(scale=np.array(scale))

	if isinstance(mesh, trimesh.Scene):
		for geometry in mesh.geometry.values():
			geometry.apply_transform(scale_matrix)
	else:
		mesh.apply_transform(scale_matrix)

	mesh.export(f"{pth_scaled_asset_id}/raw_model.glb")

	bbox_size = extract_metadata(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id, mesh)

	return bbox_size

def extract_metadata(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id, mesh=None):
	# print(f"extracting metadata for {pth_scaled_asset_id}...")

	orig_mesh = trimesh.load(f"{pth_orig_asset_id}/raw_model.glb")
	orig_bbox_size = compute_bounding_box_sizes(orig_mesh)

	if mesh is None:
		mesh = trimesh.load(f"{pth_scaled_asset_id}/raw_model.glb")
	bbox_size = compute_bounding_box_sizes(mesh)

	all_assets_metadata_scaled = {}
	if os.path.isfile(os.getenv("PTH_ASSETS_METADATA_SCALED")):
		all_assets_metadata_scaled = json.load(open(os.getenv("PTH_ASSETS_METADATA_SCALED")))

	all_assets_metadata_scaled[jid_scaled_asset] = {
		"jid": jid,
		"jid_scaled_asset": jid_scaled_asset,
		"scale": scale,
		"orig_size": orig_bbox_size,
		"size": bbox_size,
		"pth_orig_asset_id": pth_orig_asset_id,
		"pth_scaled_asset_id": pth_scaled_asset_id,
	}

	with open(os.getenv("PTH_ASSETS_METADATA_SCALED"), 'w', encoding='utf-8') as f:
		json.dump(all_assets_metadata_scaled, f, ensure_ascii=False, indent=4)

	return bbox_size

def get_bbox_from_metadata_file(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id):

	all_assets_metadata_scaled = {}
	if os.path.isfile(os.getenv("PTH_ASSETS_METADATA_SCALED")):
		all_assets_metadata_scaled = json.load(open(os.getenv("PTH_ASSETS_METADATA_SCALED")))
	
	asset = all_assets_metadata_scaled.get(jid_scaled_asset)

	if asset is not None:
		return asset.get("size")
	else:
		return extract_metadata(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id, mesh=None)

def is_valid_room(room_id, invalid_rooms):
	room_id = room_id.split("/")[-1]
	if room_id in invalid_rooms:
		return False
	# for word in ["bedroom", "livingroom", "diningroom", "library"]:
	# 	if word in room_id.lower():
	# 		return True
	return True

def has_invalid_values(data_dict, keys_to_check=["pos", "size", "rot"]):
	try:
		for key in keys_to_check:
			value = data_dict.get(key)
			if value is None:
				return True
			if np.any(np.isnan(np.array(value, dtype=np.float64))):
				return True
		return False
	except (ValueError, TypeError):
		return True

def add_room_to_blacklist(input_folder):
	room_id = input_folder.split("/")[-1]
	with open(os.getenv("PTH_INVALID_ROOMS"), "a") as fp:
		fp.write(room_id + "\n")

def convert_scene(pth_scene, all_assets_metadata, invalid_rooms, cnt):

	print(f"converting scene: {pth_scene} (# {cnt})")

	scene = json.load(open(pth_scene))

	unique_objs = {}
	# doors_windows = {}

	for furniture in scene.get("furniture"):

		# if furniture.get('valid') != None:
			# print(f"Furniture valid flag: {furniture.get('valid')}, jid: {furniture.get('jid')}")
		
		jid = furniture.get("jid")
		asset_metadata = all_assets_metadata.get(jid)

		if asset_metadata is not None:			
			uid = furniture.get("uid")
			obj = {
				"uid": uid,
				"jid": jid,
				"size": [ round(elem, 2) for elem in asset_metadata.get("size") ],
				"desc": asset_metadata.get("summary"), 
				"is_scaled": False,
			}
			unique_objs[uid] = obj
		# else:
		# 	uid = furniture.get("uid")
		# 	title = furniture.get("title")
		# 	if "window" in title:
		# 		doors_windows[uid] = {
		# 			"uid": uid,
		# 			"type": "window",
		# 		}
		# 	elif "door" in title:
		# 		uid = furniture.get("uid")
		# 		doors_windows[uid] = {
		# 			"uid": uid,
		# 			"type": "door",
		# 		}

	all_rooms = {}
	for room in scene.get("scene").get("room"):
		all_objs_room = []
		room_id = room.get("instanceid")
		
		print(room_id)
		
		if not is_valid_room(room_id, invalid_rooms):
			print("not a valid room!", room_id)
			continue

		# print("room id:", room.get('instanceid'))
		for furniture in room.get("children"): 
			
			ref = furniture.get("ref") # ref is a foreign key to uid for assets above
			
			if (ref is not None) and ((ref in unique_objs) is True):
				
				obj = copy.deepcopy(unique_objs[ref])
				
				scale = [ round(elem, 2) for elem in furniture.get("scale") ]
				if scale is not None and (scale[0] != 1 or scale[1] != 1 or scale[2] != 1):
					# print("scale is not identity vector!", scale)

					jid = obj.get("jid")
					pth_orig_asset_id = f"{os.getenv('PTH_3DFUTURE_ASSETS')}/{jid}"
					jid_scaled_asset = f"{jid}-({scale[0]})-({scale[1]})-({scale[2]})"
					pth_scaled_asset_id = f"{os.getenv('PTH_3DFUTURE_ASSETS')}/{jid_scaled_asset}"

					if os.path.exists(pth_scaled_asset_id) and os.path.isfile(pth_scaled_asset_id + "/raw_model.glb"):
						# print(f"scaled asset already found for id {pth_scaled_asset_id}")
						bbox_size = get_bbox_from_metadata_file(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id)
					else:
						os.makedirs(pth_scaled_asset_id, exist_ok=True)
						bbox_size = scale_asset(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id)

					# overwrite existing props for scaled objs
					obj["size"] = [ round(elem, 2) for elem in bbox_size ]
					obj["jid"] = jid_scaled_asset
					obj["is_scaled"] = True

				if has_invalid_values(furniture, ["pos", "rot"]) or has_invalid_values(obj, ["size"]):
					continue

				# if one object has some dim > 20 let's skip the asset
				elif obj["size"][0] > 20 or obj["size"][1] > 20 or obj["size"][2] > 20:
					continue

				obj["rot"] = rotation = furniture.get("rot")
				obj["pos"] = [ round(elem, 2) for elem in furniture.get("pos") ]
				obj["instance_id"] = furniture.get("instanceid")
				all_objs_room.append(obj)

		# let's skip scenes with less than 3 objects
		# if len(all_objs_room) < 3:
			# add_room_to_blacklist(room_id)

		if len(all_objs_room) > 0:
			try:
				bounds = pd.read_pickle(f"{os.getenv('PTH_3DFRONT_BOUNDS')}/{scene.get('uid')}/{room_id}/room_vertices_simplified.pickle")

				if np.array(bounds.get("bounds_bottom")).shape[0] != np.array(bounds.get("bounds_top")).shape[0]:
					print("not same top and bottom corner shapes!")
					continue

				if not Polygon(np.array(bounds.get("bounds_bottom"))[:, [0, 2]].tolist()).is_valid:
					print("invalid polygon found!")
					continue

				# x = bounds.get("bounds_bottom")[:, 0]
				# z = bounds.get("bounds_bottom")[:, 2]

				# if x.shape[0] > 10 or x.shape[0] < 4:
					# print(f"number of corners is too noisy! skipping room for {room_id}")
					# add_room_to_blacklist(room_id)
				
				# elif (np.max(x) - np.min(x)) > 10.0 or (np.max(z) - np.min(z)) > 10.0:
				
					# print(f"bounds exceeded! skipping room for {room_id}")
					# add_room_to_blacklist(room_id)

				# else:
				all_rooms[room_id] = {
					"bounds_top": [ elem.tolist() for elem in bounds.get("bounds_top") ],
					"bounds_bottom": [ elem.tolist() for elem in bounds.get("bounds_bottom") ],
					"objects": all_objs_room,
				}
			except FileNotFoundError as exc:
				print(exc)
			except Exception as exc:
				print(exc)
		else:
			print(f"got ZERO objects for {room_id}")
			add_room_to_blacklist(room_id)

	scene_subset = {
		"orig_scene_uid": scene.get("uid"),
	}

	# windows = []
	# for mesh in scene.get("mesh"):
	# 	if mesh.get("type") == "Window" or mesh.get("type") == "Door":
	# 		points = mesh.get("xyz") # this is a flat list of points in xyz format
	# 		windows.append({
	# 			"type": "window",
	# 			# get all points as list of lists with 3d points
	# 			"points": [ points[i:i+3] for i in range(0, len(points), 3) ],
	# 			# "bounds": 
	# 		})
	# scene_subset["windows"] = windows

	# walls = []
	# for mesh in scene.get("mesh"):
	# 	wall = {
	# 		"uid": mesh.get("uid"),
	# 		"xyz": mesh.get("xyz"),
	# 		"faces": mesh.get("faces"),
	# 		"normals": mesh.get("normal"),
	# 		"uv": mesh.get("uv"),
	# 	}
	# 	walls.append(wall)
	# scene_subset["walls"] = walls

	for idx, room_id in enumerate(all_rooms):

		scene_subset_room = scene_subset.copy()
		scene_subset_room["uid"] = str(uuid.uuid4())
		scene_subset_room["room_id"] = room_id

		scene_subset_room["objects"] = all_rooms[room_id].get("objects")

		scene_subset_room["bounds_top"] = all_rooms[room_id].get("bounds_top")
		scene_subset_room["bounds_bottom"] = all_rooms[room_id].get("bounds_bottom")

		is_center_valid = shift_scene_by_center(scene_subset_room)

		if is_center_valid:
			if DO_SAVE_FILES:
				# save_file(scene_subset_room)
				save_file(scene_subset_room, idx=idx)
			cnt += 1
		else:
			add_room_to_blacklist(room_id)

	return cnt		

# **********************************************************************************************************

load_dotenv(".env.local")

with open(os.getenv("PTH_INVALID_ROOMS")) as fp:
	invalid_rooms = [line.rstrip() for line in fp]   

# DO_FULL_SCENES = False

DO_SAVE_FILES = True

# if os.path.exists(os.getenv("PTH_STAGE_1")):
# 	print("removing existing json files...")
# 	shutil.rmtree(os.getenv("PTH_STAGE_1"))
# os.makedirs(os.getenv("PTH_STAGE_1"), exist_ok=True)

# **********************************************************************************************************

# build dictionary from metadata instead of list for O(1) access
# all_assets_metadata = {}
# for asset in json.load(open("/Volumes/apollo11/data/3D-FUTURE-assets/model_info.json")):
	# all_assets_metadata[asset.get("model_id")] = asset

all_assets_metadata = json.load(open("/Volumes/apollo11/data/3D-FUTURE-assets/model_info_martin.json"))

# all_pths = [f for f in os.listdir(os.getenv("PTH_3DFRONT_SCENES")) if f.endswith('.json') and not f.startswith(".")]

# all_pths = [ "6b774494-78d2-4def-a1df-24e4d907e796.json" ]
all_pths = [ "0a9c667d-033d-448c-b17c-dc55e6d3c386.json" ]

# all_pths = [ "06543b99-4fe3-4be8-be81-f62a83bbee97.json" ]
# all_pths = [ "103cce55-24d5-4c71-9856-156962e30511.json" ]
# all_pths = [ "8abd4a65-e5c5-4edb-b494-7ebd05ff84b9.json" ]
# all_pths = [ "0a77986b-5c04-41b9-8ee8-1ea301ccda3e.json" ]
# all_pths = [ "b01fbcd3-a49b-4cdb-a04e-743db03d3d87.json" ]

cnt = 0
for pth_scene in tqdm(all_pths):
	cnt = convert_scene(os.path.join(os.getenv("PTH_3DFRONT_SCENES"), pth_scene), all_assets_metadata, invalid_rooms, cnt)

print(f"total # of rooms: {cnt}")
