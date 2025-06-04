import json
import os
import numpy as np
import trimesh
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import pdb

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
	bbox_size = [round(elem, 2) for elem in bbox_size]
	return bbox_size

def scale_asset(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id):
	print(f"Scaling asset with id {jid_scaled_asset}")
	
	# Load original mesh
	mesh_path = os.path.join(pth_orig_asset_id, "raw_model.glb")
	if not os.path.exists(mesh_path):
		print(f"Warning: Original asset not found at {mesh_path}")
		return None
	
	try:
		mesh = trimesh.load(mesh_path)
	except Exception as e:
		print(f"Error loading mesh {mesh_path}: {e}")
		return None
	
	# Apply scaling transformation
	scale_matrix = trimesh.transformations.compose_matrix(scale=np.array(scale))
	
	if isinstance(mesh, trimesh.Scene):
		for geometry in mesh.geometry.values():
			geometry.apply_transform(scale_matrix)
	else:
		mesh.apply_transform(scale_matrix)

	pdb.set_trace()
	
	# Save scaled mesh
	os.makedirs(pth_scaled_asset_id, exist_ok=True)
	output_path = os.path.join(pth_scaled_asset_id, "raw_model.glb")
	
	try:
		mesh.export(output_path)
	except Exception as e:
		print(f"Error saving scaled mesh to {output_path}: {e}")
		return None
	
	# Compute and return bounding box size
	bbox_size = compute_bounding_box_sizes(mesh)
	return bbox_size

def extract_metadata(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id, mesh=None):
	# Load original mesh to get original bbox size
	orig_mesh_path = os.path.join(pth_orig_asset_id, "raw_model.glb")
	if not os.path.exists(orig_mesh_path):
		return None
	
	try:
		orig_mesh = trimesh.load(orig_mesh_path)
		orig_bbox_size = compute_bounding_box_sizes(orig_mesh)
	except Exception as e:
		print(f"Error loading original mesh {orig_mesh_path}: {e}")
		return None
	
	# Use provided mesh or load scaled mesh to get scaled bbox size
	if mesh is None:
		scaled_mesh_path = os.path.join(pth_scaled_asset_id, "raw_model.glb")
		if not os.path.exists(scaled_mesh_path):
			return None
		
		try:
			mesh = trimesh.load(scaled_mesh_path)
		except Exception as e:
			print(f"Error loading scaled mesh {scaled_mesh_path}: {e}")
			return None
	
	bbox_size = compute_bounding_box_sizes(mesh)
	
	# Load existing metadata
	assets_metadata_scaled_file = os.getenv("PTH_ASSETS_METADATA_SCALED")
	all_assets_metadata_scaled = {}
	if os.path.isfile(assets_metadata_scaled_file):
		try:
			with open(assets_metadata_scaled_file, 'r') as f:
				all_assets_metadata_scaled = json.load(f)
		except Exception as e:
			print(f"Error loading metadata file {assets_metadata_scaled_file}: {e}")
	
	# Add new metadata
	all_assets_metadata_scaled[jid_scaled_asset] = {
		"jid": jid,
		"jid_scaled_asset": jid_scaled_asset,
		"scale": scale,
		"orig_size": orig_bbox_size,
		"size": bbox_size,
		"pth_orig_asset_id": pth_orig_asset_id,
		"pth_scaled_asset_id": pth_scaled_asset_id,
	}

	try:
		with open(assets_metadata_scaled_file, 'w', encoding='utf-8') as f:
			json.dump(all_assets_metadata_scaled, f, ensure_ascii=False, indent=4)
	except Exception as e:
		print(f"Error saving metadata file {assets_metadata_scaled_file}: {e}")
	
	return bbox_size

def get_bbox_from_metadata_file(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id):
	assets_metadata_scaled_file = os.getenv("PTH_ASSETS_METADATA_SCALED")
	all_assets_metadata_scaled = {}
	
	if os.path.isfile(assets_metadata_scaled_file):
		try:
			with open(assets_metadata_scaled_file, 'r') as f:
				all_assets_metadata_scaled = json.load(f)
		except Exception as e:
			print(f"Error loading metadata file {assets_metadata_scaled_file}: {e}")
	
	asset = all_assets_metadata_scaled.get(jid_scaled_asset)
	
	if asset is not None:
		return asset.get("size")
	else:
		return extract_metadata(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id, mesh=None)

def is_valid_room(room_id, invalid_rooms):
	room_id = room_id.split("/")[-1]
	if room_id in invalid_rooms:
		return False
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

def process_scene_file(scene_path, processed_assets, invalid_rooms, all_assets_metadata, cnt):	
	try:
		with open(scene_path, 'r') as f:
			scene = json.load(f)
	except Exception as e:
		print(f"Error loading scene file {scene_path}: {e}")
		return
	
	# Process each room in the scene
	for room in scene.get("scene", {}).get("room", []):
		room_id = room.get("instanceid")
			
		# Skip invalid rooms
		if not is_valid_room(room_id, invalid_rooms):
			continue

		for furniture in room.get("children"):
			ref = furniture.get("ref")
			scale = furniture.get("scale")
			
			if ref is None or scale is None:
				continue
			
			# Check if scale is non-identity
			scale = [round(elem, 2) for elem in scale]
			if scale[0] == 1.0 and scale[1] == 1.0 and scale[2] == 1.0:
				continue

			if has_invalid_values(furniture, ["scale"]):
				continue
			
			# Find the corresponding furniture item to get the jid
			jid = None
			for furn in scene.get("furniture", []):
				if furn.get("uid") == ref:
					jid = furn.get("jid")
					break

			if jid is None:
				continue

			# Check if asset metadata exists (like original script)
			asset_metadata = all_assets_metadata.get(jid)
			if asset_metadata is None:
				continue

			# Apply validation checks BEFORE scaling (like original script)
			if has_invalid_values(furniture, ["pos", "rot"]):
				continue

			obj = { "size": [ round(elem, 2) for elem in asset_metadata.get("size") ] }
			if has_invalid_values(obj, ["size"]):
				continue
			
			# Generate scaled asset identifier
			jid_scaled_asset = f"{jid}-({scale[0]})-({scale[1]})-({scale[2]})"
			
			# Skip if already processed
			if jid_scaled_asset in processed_assets:
				# print(f"Scaled asset entry already exists in metadata_scaled: {jid_scaled_asset}")
				continue
			
			pth_orig_asset_id = os.path.join(os.getenv("PTH_3DFUTURE_ASSETS"), jid)
			pth_scaled_asset_id = os.path.join(os.getenv("PTH_3DFUTURE_ASSETS"), jid_scaled_asset)
			
			# Check if original asset exists
			if not os.path.exists(os.path.join(pth_orig_asset_id, "raw_model.glb")):
				continue
			
			# Check if scaled asset already exists
			if os.path.exists(pth_scaled_asset_id) and os.path.isfile(os.path.join(pth_scaled_asset_id, "raw_model.glb")):
				bbox_size = get_bbox_from_metadata_file(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id)
				pass
			else:
				cnt += 1
				# print(cnt)
				bbox_size = scale_asset(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id)
				if bbox_size is not None:
					extract_metadata(jid, jid_scaled_asset, scale, pth_orig_asset_id, pth_scaled_asset_id)
			
			processed_assets.add(jid_scaled_asset)

	return cnt

def main():
	# Load environment variables from .env file
	load_dotenv(".env")
	
	# Validate required environment variables
	required_env_vars = ["PTH_3DFRONT_SCENES", "PTH_3DFUTURE_ASSETS", "PTH_ASSETS_METADATA_SCALED", "PTH_ASSETS_METADATA", "PTH_INVALID_ROOMS"]
	for var in required_env_vars:
		if not os.getenv(var):
			print(f"Error: Environment variable {var} not set. Please check your .env file.")
			return
	
	scenes_dir = Path(os.getenv("PTH_3DFRONT_SCENES"))
	assets_dir = Path(os.getenv("PTH_3DFUTURE_ASSETS"))
	assets_metadata_scaled_file = os.getenv("PTH_ASSETS_METADATA_SCALED")
	invalid_rooms_file = os.getenv("PTH_INVALID_ROOMS")
	
	# Validate directories
	if not scenes_dir.exists():
		print(f"Error: Scenes directory does not exist: {scenes_dir}")
		return
	
	if not assets_dir.exists():
		print(f"Error: Assets directory does not exist: {assets_dir}")
		return
	
	# Load invalid rooms list
	invalid_rooms = []
	if os.path.exists(invalid_rooms_file):
		try:
			with open(invalid_rooms_file, 'r') as fp:
				invalid_rooms = [line.rstrip() for line in fp]
			print(f"Loaded {len(invalid_rooms)} invalid rooms to skip")
		except Exception as e:
			print(f"Error loading invalid rooms file {invalid_rooms_file}: {e}")
	else:
		print(f"Warning: Invalid rooms file not found at {invalid_rooms_file}")

	# Load assets metadata (required for filtering like original script)
	assets_metadata_file = os.getenv("PTH_ASSETS_METADATA")
	if not os.path.exists(assets_metadata_file):
		print(f"Error: Assets metadata file not found at {assets_metadata_file}")
		return
	try:
		all_assets_metadata = json.load(open(assets_metadata_file))
		print(f"Loaded metadata for {len(all_assets_metadata)} assets")
	except Exception as e:
		print(f"Error loading assets metadata: {e}")
		return
	
	# Find all scene files
	scene_files = [f for f in scenes_dir.glob("*.json") if not f.name.startswith(".")]
	if not scene_files:
		print(f"No scene files found in {scenes_dir}")
		return
	print(f"Found {len(scene_files)} scene files to process")
	
	# Track processed assets to avoid duplicates
	processed_assets = set()
	if os.path.exists(assets_metadata_scaled_file):
		try:
			with open(assets_metadata_scaled_file, 'r') as f:
				existing_metadata = json.load(f)
				processed_assets.update(existing_metadata.keys())
				print(f"Found {len(processed_assets)} already processed assets in metadata")
		except Exception as e:
			print(f"Error loading existing metadata: {e}")

	# Process each scene file
	cnt = 0
	print("Processing scenes to generate scaled assets...")
	for scene_file in tqdm(scene_files, desc="Processing scenes"):
		cnt = process_scene_file(scene_file, processed_assets, invalid_rooms, all_assets_metadata, cnt)

	print(f"Assets effectively scaled in this run: {cnt}")
	print(f"Asset scaling complete. Metadata saved to {assets_metadata_scaled_file}")
	print(f"Total unique scaled assets processed: {len(processed_assets)}")


if __name__ == "__main__":
	main()