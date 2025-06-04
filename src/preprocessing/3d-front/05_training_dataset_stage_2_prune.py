import json
from tqdm import tqdm
import os
import shutil
import math
import pdb
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
import hashlib
import copy

from src.eval import eval_scene
from src.utils import remove_and_recreate_folder, get_room_type_from_id

def is_invalid_scene_by_heuristics(scene, pth_scene):

	room_id = scene.get("room_id")

	# filter out scenes that are not bedrooms
	# if not "bedroom" in room_id.lower():
		# print("room id type does not match!")
		# return True

	x = np.array(scene.get("bounds_bottom"))[:, 0]
	z = np.array(scene.get("bounds_bottom"))[:, 2]

	# if x.shape[0] > 16 or x.shape[0] < 4:
	if x.shape[0] < 4:
		print(f"number of corners is < 4 ... ({x.shape[0]}) skipping room {room_id}", pth_scene)
		return True
	
	# elif (np.max(x) - np.min(x)) > 10.0 or (np.max(z) - np.min(z)) > 10.0:
		# print(f"bounds exceeded! skipping room {room_id}")
		# return True

	n_objs = len(scene.get("objects"))
	if n_objs < 2:
		print(f"number of objects too small: ({n_objs}) ... skipping room {room_id}")
		return True
	
	# if n_objs > 40:
		# print(f"number of objects > 40 ({n_objs})... skipping room {room_id}", pth_scene)
	
	if n_objs > 50:
		print(f"number of objects too large ({n_objs})... skipping room {room_id}", pth_scene)
		return True

	return False

def get_scene_hash(scene):
	scene_str = json.dumps(scene, sort_keys=True)
	scene_hash = hashlib.md5(scene_str.encode()).hexdigest()
	return scene_hash

def is_invalid_scene_by_mesh_pbl(scene, idx):
	scene_cp = copy.deepcopy(scene)
	metrics = eval_scene(scene_cp, is_debug=False, idx=idx)

	# if single object has already high pbl loss then remove object + eval again => if valid then keep partial scene
	if metrics['obj_with_highest_pbl_loss']['pbl'] > 0.1:
		obj_idx = metrics['obj_with_highest_pbl_loss']['idx']
		scene_cp["objects"].pop(obj_idx)
		old_pbl_loss = metrics['total_pbl_loss']
		metrics = eval_scene(scene_cp, is_debug=False, idx=idx)
		if metrics["is_valid_scene"]:
			print(f">> valid scene with 1 object removed! {old_pbl_loss} -> {metrics['total_pbl_loss']}")
			print(f"replacing scene ({len(scene.get('objects'))} objs) with new scene ({len(scene_cp.get('objects'))} objs)")
			return False, scene_cp

	elif metrics["is_valid_scene"]:
		print(f"valid scene! {metrics['total_pbl_loss']}")
		return False, scene

	print(f"invalid scene! {metrics['total_pbl_loss']}")
	return True, None

def convert_scene(pth_scene, cnt, room_counts, idx):
	scene = json.load(open(pth_scene))

	if is_invalid_scene_by_heuristics(scene, pth_scene):
		print("⛔️ INVALID scene according to heuristics", pth_scene)
		return cnt, room_counts, idx+1

	is_invalid, scene = is_invalid_scene_by_mesh_pbl(scene, idx)
	if is_invalid:
		print(f"⛔️ INVALID scene according to eval metrics", pth_scene)
		return cnt, room_counts, idx+1
	else:
		# after removing the worst object, the scene can still become invalid by heuristics
		if is_invalid_scene_by_heuristics(scene, pth_scene):
			print("⛔️ INVALID scene according to heuristics", pth_scene)
			return cnt, room_counts, idx+1
		else:
			scene_pruned = {
				"bounds_top": scene.get("bounds_top"),
				"bounds_bottom": scene.get("bounds_bottom"),
				"room_type": get_room_type_from_id(scene.get("room_id")),
				"room_id": scene.get("room_id"),
				"objects": []
			}

			room_type = scene_pruned["room_type"]
			room_counts[room_type] += 1

			for obj in scene.get("objects"):
				obj_pruned = {
					"desc": obj.get("desc"),
					"size": [ round(elem, 2) for elem in obj.get("size") ],
					"pos": [ round(elem, 2) for elem in obj.get("pos") ],
					"rot": obj.get("rot"),
					"jid": obj.get("jid"),
				}
				scene_pruned["objects"].append(obj_pruned)

			pth_file = os.getenv("PTH_STAGE_2") + "/" + scene.get("orig_scene_uid") + "-" + scene.get("uid") + ".json"
			with open(pth_file, "w") as write_file:
				print("✅ exporting VALID scene", pth_scene)
				json.dump(scene_pruned, write_file, indent=4)

			return cnt+1, room_counts, idx+1

def check_for_duplicates(processed_scenes):    
	scene_hashes = {}
	total_duplicate_files = 0  # Add counter for individual duplicate files
	
	for (scene, pth_scene) in processed_scenes:
		scene_hash = get_scene_hash(scene)
		
		if scene_hashes.get(scene_hash) is not None:
			scene_hashes[scene_hash].append(pth_scene)
			total_duplicate_files += 1  # Increment for each duplicate file
		else:
			scene_hashes[scene_hash] = [pth_scene]

	duplicate_groups = [{"scene_hash": scene_hash, "pth_scenes": pth_scenes} for scene_hash, pth_scenes in scene_hashes.items() if len(pth_scenes) > 1]
			
	return duplicate_groups, total_duplicate_files

# **********************************************************************************************************

load_dotenv(".env.local")
# load_dotenv(".env.stanley")

# # scene = json.load(open("/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/103cce55-24d5-4c71-9856-156962e30511-57950022-d81f-4a1d-ada2-7a1d9d256b9d.json"))
# scene = json.load(open("/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/3935a020-6a26-4e14-ba45-39062fc8aed0-7f130e0e-e75a-44a5-95a7-1d1aff561eb9.json"))
# # scene_2 = json.load(open("/Volumes/apollo11/data/3D-FRONT-martin-rooms-stage-1/0773253b-c93b-4fea-85a3-220c4edce8e1-9b5dcd4e-8e96-4f16-8b20-a66476ef1635.json"))

# # print(get_scene_hash(scene_1))
# # print(get_scene_hash(scene_2))
# # exit()

# metrics = eval_scene(scene, is_debug=True)
# exit()

# **********************************************************************************************************

all_pths = [f for f in os.listdir(os.getenv("PTH_STAGE_1")) if f.endswith('.json') and not f.startswith(".")]

remove_and_recreate_folder(os.getenv("PTH_STAGE_2"))

room_counts = defaultdict(int)
cnt = 0
idx = 0
pbar = tqdm(all_pths)
for pth_scene in pbar:
	cnt, room_counts, idx = convert_scene(os.path.join(os.getenv("PTH_STAGE_1"), pth_scene), cnt, room_counts, idx)
	pbar.set_description(f"keeping {cnt} scenes")
print(f"total # of scenes: {cnt}")
print(f"\nroom type distribution:")
for room_type, count in sorted(room_counts.items()):
	print(f"\t{room_type}: {count}")

# **********************************************************************************************************

# Check for duplicates
all_pths = [f for f in os.listdir(os.getenv("PTH_STAGE_2")) if f.endswith('.json') and not f.startswith(".")]
scenes = []
for pth_scene in tqdm(all_pths):
	scene = json.load(open(os.path.join(os.getenv("PTH_STAGE_2"), pth_scene)))
	scenes.append((scene, pth_scene))

duplicates, total_duplicate_files = check_for_duplicates(scenes)
if duplicates:
	print(f"\nfound {len(duplicates)} duplicate groups with {total_duplicate_files} total duplicate files!")
else:
	print("\n✅ no duplicate scenes found")

# Create new folder without duplicates
stage_2_dedup = f"{os.getenv('PTH_STAGE_2')}-dedup"
os.makedirs(stage_2_dedup, exist_ok=True)
copied_hashes = set()
for scene, pth_scene in tqdm(scenes):
	scene_hash = get_scene_hash(scene)
	# Only copy if we haven't seen this hash before
	if scene_hash not in copied_hashes:
		shutil.copy2(
			os.path.join(os.getenv("PTH_STAGE_2"), pth_scene),
			os.path.join(stage_2_dedup, pth_scene)
		)
		copied_hashes.add(scene_hash)

print(f"\nCreated deduplicated dataset at {stage_2_dedup}")
print(f"Original dataset size: {len(scenes)} scenes")
print(f"Deduplicated dataset size: {len(copied_hashes)} scenes")