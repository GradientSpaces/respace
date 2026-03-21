# pipeline.py

import json
import torch
import re
import numpy as np
import argparse
from dotenv import load_dotenv
from accelerate import Accelerator
import copy
import traceback
from tqdm import tqdm
import os
import pdb
import gc
from pathlib import Path
from collections import defaultdict
import logging
import sys
import time
import matplotlib.pyplot as plt
from collections import Counter
import uuid

from src.utils import get_test_instrs_all, StreamToLogger, remove_and_recreate_folder, create_category_lookup, find_removed_objects, classify_confusion_type
from src.dataset import load_train_val_test_datasets, get_pths_dataset_split, create_full_scene_from_before_and_added, create_sequential_editing_dataset
from src.utils import set_seeds, is_rectangular_room
from src.respace import ReSpace
from src.viz import render_full_scene_and_export_with_gif

def generate_full_scene(idx, respace, pth_scene, pth_output, scene_id=None):
	# prep empty scene with fixed bounds from test set sample
	with open(os.path.join(os.getenv("PTH_STAGE_2_DEDUP"), pth_scene), "r", encoding='utf-8') as f:
		scene = json.loads(f.read())

	scene_bounds_only = {
		"room_type": scene.get("room_type"),
		"bounds_top": scene.get("bounds_top"),
		"bounds_bottom": scene.get("bounds_bottom"),
		"objects": []
	}

	final_scene, is_success, _ = respace.generate_full_scene(
		room_type=scene.get("room_type"),
		scene_bounds_only=scene_bounds_only,
		scene_id=scene_id,
		do_skip_removals=True,
	)

	num_objects = len(final_scene.get("objects", []))

	return final_scene, is_success, num_objects
	
def generate_instr_scene(idx, respace, rand_seed, sample, pth_output, all_test_instrs, desc_to_category, do_removal_test=False):
	# prep sample
	instr_sample = all_test_instrs.get(sample.get("pth_orig_file"))[rand_seed]
	prompt = instr_sample.get("prompt")
	scene_before_gt_with_jids = json.loads(instr_sample.get("sg_input"))

	pth_viz_output = Path(pth_output) / str(rand_seed) / "best-of-n" / str(idx)
	do_rendering_with_object_count = False
	analysis_data = None

	if do_removal_test:
		# we add the object intended for addition first to create the "before" scene, then ask it to remove it with the given prompt
		obj_add = json.loads(instr_sample.get("sg_output_add"))
		scene_merged = create_full_scene_from_before_and_added(scene_before_gt_with_jids, obj_add)

		for obj in scene_merged["objects"]:
			if "uuid" not in obj:
				obj["uuid"] = str(uuid.uuid4())

		scene_length = len(json.dumps(scene_merged).split())
		target_category = desc_to_category.get(obj_add.get("desc"), "unknown")
		
		final_scene, _ = respace.remove_object(prompt, scene_merged, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, idx=idx)

		removed_objects = find_removed_objects(scene_merged, final_scene)
		removed_categories = [desc_to_category.get(obj.get("desc"), "unknown") for obj in removed_objects]
		confusion_type = classify_confusion_type(target_category, removed_categories)

		if obj_add.get("desc") not in [ obj.get("desc") for obj in final_scene.get("objects") ]:
			is_success = True
		else:
			is_success = False

		analysis_data = {
			"idx": idx,
			"seed": rand_seed,
			"prompt": prompt,
			"prompt_length": len(prompt.split()),
			"scene_length": scene_length,
			"target_object": {
				"desc": obj_add.get("desc"),
				"category": target_category,
			},
			"removed_objects": [
				{
					"desc": obj.get("desc"),
					"category": desc_to_category.get(obj.get("desc"), "unknown"),
				} for obj in removed_objects
			],
			"is_success": is_success,
			"confusion_type": confusion_type
		}
	else:
		temp = 0.7
		final_scene, is_success = respace.add_object(prompt, scene_before_gt_with_jids, do_sample_assets_for_input_scene=True, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, temp=temp)

	return final_scene, is_success, analysis_data, 1

def generate_and_eval_full_sequence(idx, respace, seq_samples):
	sample = seq_samples[idx]

	# print list of seq_length for all items in seq_sample at once
	# seq_lengths = [s["seq_length"] for s in seq_samples]
	# print(seq_lengths)
	
	# Build all_commands from steps (already decomposed in dataset)
	all_commands = []
	for step in sample["steps"]:
		if step["op"] == "add":
			all_commands.append(f"<add>{step['add_prompt']}</add>")
		else:
			all_commands.append(f"<remove>{step['remove_prompt']}</remove>")
	
	final_scene, is_success, step_results = respace.handle_prompt(
		prompt=None,
		current_scene=sample["partial_scene"],
		all_commands=all_commands,
		gt_steps=sample["steps"],   # triggers metric computation inside
	)
	
	metrics = aggregate_seq_metrics(step_results)
	
	metrics["seq_length"] = sample["seq_length"]
	metrics["sample_idx"] = idx
	
	return final_scene, metrics

def aggregate_seq_metrics(step_results: list) -> dict:
    add_results = [r for r in step_results if r["op"] == "add"]
    rem_results = [r for r in step_results if r["op"] == "remove"]

    acc_add = float(np.mean([r["is_success"] for r in add_results])) if add_results else None
    acc_rem = float(np.mean([r["is_success"] for r in rem_results])) if rem_results else None
    acc_seq = float(np.mean([r["is_success"] for r in step_results])) if step_results else None

    return {
        "acc_seq":     acc_seq,
        "acc_add":     acc_add,
        "acc_rem":     acc_rem,
        "n_steps":     len(step_results),
        "n_add_steps": len(add_results),
        "n_rem_steps": len(rem_results),
    }

def print_and_save_seq_eval_results(all_seq_metrics, pth_output, room_type, bon_llm, do_bon_rotation):
    all_samples_flat = [m for seed_metrics in all_seq_metrics for m in seed_metrics]

    def nanmean_none(vals):
        vals = [v for v in vals if v is not None]
        return np.mean(vals) if vals else float("nan")

    # Per-seed aggregation
    seed_means = []
    for seed_metrics in all_seq_metrics:
        seed_means.append({
            "acc_seq": nanmean_none([m["acc_seq"] for m in seed_metrics]),
            "acc_add": nanmean_none([m["acc_add"] for m in seed_metrics]),
            "acc_rem": nanmean_none([m["acc_rem"] for m in seed_metrics]),
        })

    def fmt(key):
        vals = [s[key] for s in seed_means]
        return f"{np.mean(vals):.4f} (+/- {np.std(vals):.4f})"

    print(f"acc_seq : {fmt('acc_seq')}")
    print(f"acc_add : {fmt('acc_add')}")
    print(f"acc_rem : {fmt('acc_rem')}")

    # Breakdown by seq_length
    bins = sorted(set(m["seq_length"] for m in all_samples_flat))
    print("\nBreakdown by seq_length:")
    for b in bins:
        bin_samples = [m for m in all_samples_flat if m["seq_length"] == b]
        acc_vals = [m["acc_seq"] for m in bin_samples if m["acc_seq"] is not None]
        print(f"  L={b:2d}: {np.mean(acc_vals):.4f} (n={len(acc_vals)})")

    with open(f"./plots/seq_eval_raw_{room_type}_bon_{bon_llm}{'_rot' if do_bon_rotation else ''}.json", "w") as f:
        json.dump(all_seq_metrics, f, indent=4)

# def do_sample_custom(respace):
# 	pth_viz_output = Path("./eval/viz/misc/custom")
# 	remove_and_recreate_folder(pth_viz_output)
# 	do_rendering_with_object_count = True
# 	# scene_before = '{"room_type": "bedroom", "bounds_top": [[-1.45, 2.6, 2.45], [0.45, 2.6, 2.45], [0.45, 2.6, 1.45], [1.45, 2.6, 1.45], [1.45, 2.6, -2.45], [-1.45, 2.6, -2.45]], "bounds_bottom": [[-1.45, 0.0, 2.45], [0.45, 0.0, 2.45], [0.45, 0.0, 1.45], [1.45, 0.0, 1.45], [1.45, 0.0, -2.45], [-1.45, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "size": [0.57, 1.21, 0.63], "pos": [1.1, 0.0, 1.05], "rot": [0, 0, 0, 1], "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6-(0.68)-(0.68)-(0.68)"}, {"desc": "Elegant wooden wardrobe with three geometric-patterned glass doors, two drawers, and modern metal handles.", "size": [1.45, 2.28, 0.62], "pos": [0.67, 0.0, -2.1], "rot": [0, 0, 0, 0], "sampled_asset_jid": "a0b67c64-15a4-4969-91a6-89e365d87d12"}, {"desc": "Modern contemporary pendant lamp featuring white fabric conical shades on a geometric gold metal frame with multiple light sources.", "size": [1.06, 1.03, 0.47], "pos": [0.02, 2.08, -0.44], "rot": [0, -0.71254, 0, 0.70164], "sampled_asset_jid": "5a72093d-b9e5-4823-906b-331ced5e08d7"}, {"desc": "Modern beige upholstered king-size bed with minimalist design and neatly tailored edges.", "size": [1.9, 1.11, 2.23], "pos": [-0.29, 0.0, -0.3], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "6c7bf8e0-37a2-4661-a554-3af2b1e242d6"}, {"desc": "A modern-traditional nightstand in dark brown wood with a gold geometric patterned front, featuring two drawers and sleek elevated legs.", "size": [0.58, 0.59, 0.46], "pos": [-1.31, 0.0, -1.71], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "8b8cdbde-57e3-432a-a46a-89a77f8e6294"}]}'
# 	# scene_before = '{"room_type": "bedroom", "bounds_top": [[-1.45, 2.6, 2.45], [0.45, 2.6, 2.45], [0.45, 2.6, 1.45], [1.45, 2.6, 1.45], [1.45, 2.6, -2.45], [-1.45, 2.6, -2.45]], "bounds_bottom": [[-1.45, 0.0, 2.45], [0.45, 0.0, 2.45], [0.45, 0.0, 1.45], [1.45, 0.0, 1.45], [1.45, 0.0, -2.45], [-1.45, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "size": [0.57, 1.21, 0.63], "pos": [1.1, 0.0, 1.05], "rot": [0, 0, 0, 1], "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6-(0.68)-(0.68)-(0.68)"}, {"desc": "Elegant wooden wardrobe with three geometric-patterned glass doors, two drawers, and modern metal handles.", "size": [1.45, 2.28, 0.62], "pos": [0.67, 0.0, -2.1], "rot": [0, 0, 0, 0], "sampled_asset_jid": "a0b67c64-15a4-4969-91a6-89e365d87d12"}, {"desc": "Modern contemporary pendant lamp featuring white fabric conical shades on a geometric gold metal frame with multiple light sources.", "size": [1.06, 1.03, 0.47], "pos": [0.02, 2.08, -0.44], "rot": [0, -0.71254, 0, 0.70164], "sampled_asset_jid": "5a72093d-b9e5-4823-906b-331ced5e08d7"}, {"desc": "Modern beige upholstered king-size bed with minimalist design and neatly tailored edges.", "size": [1.9, 1.11, 2.23], "pos": [-0.29, 0.0, -0.3], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "6c7bf8e0-37a2-4661-a554-3af2b1e242d6"}, {"desc": "A modern-traditional nightstand in dark brown wood with a gold geometric patterned front, featuring two drawers and sleek elevated legs.", "size": [0.58, 0.59, 0.46], "pos": [-1.31, 0.0, -1.71], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "8b8cdbde-57e3-432a-a46a-89a77f8e6294"}, {"desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "pos": [-1.1, 0.0, 1.38], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [1.1, 1.36, 0.81], "prompt": "modern dark wooden desk", "sampled_asset_jid": "ec9190d1-cc42-4a85-bb1e-730ed7642f51", "sampled_asset_desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "sampled_asset_size": [1.1008340120315552, 1.3596680217888206, 0.8073000013828278], "uuid": "51b03ac6-941c-4beb-a8c1-84d69f8a41c1"}]}'
	
# 	# scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}]}'
	
# 	scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}]}'
# 	scene_before_teaser = json.loads(scene_before_teaser)
# 	temp = 0.7

# 	# final_scene, is_success = respace.add_object("modern dark wooden desk", scene_before, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, temp=temp)
# 	# final_scene, is_success = respace.add_object("office chair", scene_before, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, temp=temp)
# 	# final_scene, is_success = respace.add_object("modern dark wooden desk", scene_before_teaser, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, temp=temp)
# 	# final_scene, is_success = respace.add_object("round wooden coffee table", scene_before_teaser, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output, temp=temp)

# 	# final_scene, is_success = respace.handle_prompt("add large pendant lamp and large indoor plant", scene_before_teaser, system_prompt=None, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)
	
# 	# final_scene, is_success = respace.handle_prompt("add large round wooden coffee table", scene_before_teaser, system_prompt=None, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)

# 	# scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"}, {"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "pos": [-0.5, 0.0, -1.96], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.84, 1.78, 0.93], "prompt": "large artificial green plant", "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6", "sampled_asset_desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "sampled_asset_size": [0.8378599882125854, 1.7756899947231837, 0.9323999881744385], "uuid": "02a55e98-2eb7-4a36-b03b-ea063c22b9f7"}]}'
# 	# scene_before_teaser = json.loads(scene_before_teaser)
# 	# final_scene, is_success = respace.handle_prompt("remove plant with black ceramic planter", scene_before_teaser, system_prompt=None, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)
	
# 	# scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}]}'
# 	# scene_before_teaser = json.loads(scene_before_teaser)
# 	# final_scene, is_success = respace.handle_prompt("add dark mid century sofa", scene_before_teaser, system_prompt=None, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)

# 	# scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"}]}'
# 	scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"}, {"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "pos": [-0.5, 0.0, -1.96], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.84, 1.78, 0.93], "prompt": "large artificial green plant", "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6", "sampled_asset_desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "sampled_asset_size": [0.8378599882125854, 1.7756899947231837, 0.9323999881744385], "uuid": "02a55e98-2eb7-4a36-b03b-ea063c22b9f7"}]}'
# 	scene_before_teaser = json.loads(scene_before_teaser)

# 	# render_full_scene_and_export_with_gif(final_scene, filename="0-2", pth_output=Path("./eval/viz/misc/custom"), create_gif=False, bg_color=None, camera_height=6.5)

# 	render_full_scene_and_export_with_gif(scene_before_teaser, filename="0-0", pth_output=pth_viz_output, create_gif=False, bg_color=None, camera_height=6.5)
# 	# render_full_scene_and_export_with_gif(final_scene, filename="0-5", pth_output=pth_viz_output, create_gif=False, bg_color=None, camera_height=6.5)

# 	# final_scene, is_success = respace.handle_prompt("add round coffee table", scene_before_teaser, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)
# 	# final_scene, is_success = respace.handle_prompt("swap black couch with white sofa and add coffee table", scene_before_teaser, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)

# 	# final_scene, is_success = respace.handle_prompt("remove that one plant with a black ceramic planter", scene_before_teaser, system_prompt=None, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)
	
# 	# remove key/values with "sampled_" from the last object in objects list
# 	scene_2 = scene_before_teaser.copy()
# 	scene_2["objects"][-1] = {k: v for k, v in scene_2["objects"][-1].items() if not k.startswith("sampled_")}

# 	scene_2 = respace.sampling_engine.sample_last_asset(scene_2, is_greedy_sampling=True, )

# 	# final_scene, is_success = respace.handle_prompt("remove plant with black ceramic planter", scene_before_teaser, system_prompt=None, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)

# 	# scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}]}'
# 	# scene_before_teaser = json.loads(scene_before_teaser)
# 	# final_scene, is_success = respace.handle_prompt("replace bookcase with a wooden wardrobe", scene_before_teaser, system_prompt=None, do_rendering_with_object_count=do_rendering_with_object_count, pth_viz_output=pth_viz_output)

# 	# print(final_scene)

def print_summary(args):
	print("="*20)
	print(f"room_type: {args.room_type}")
	print(f"rand_seeds: {args.seed_only if args.seed_only is not None else [1234, 3456, 5678]}")
	print(f"do_full_scenes: {args.do_full_scenes}")
	print(f"do_removal_test: {args.do_removal_test}")
	print(f"do_seq_test: {args.do_seq_test}")
	print("")
	print(f"n_test_scenes: {args.n_test_scenes}")
	print(f"bon_llm: {args.bon_llm}")
	print(f"do_bon_rotation: {args.do_bon_rotation}")
	print(f"do_bon_shuffling: {args.do_bon_shuffling}")
	print(f"bon_shuffling: {args.bon_shuffling}")
	print("")
	print(f"pth_output: {args.pth_output}")
	print("="*20)

def main(args):
	env_file = f"{args.env}"
	load_dotenv(env_file)

	accelerator = Accelerator()
	dvc = accelerator.device

	if args.use_logfile:
		print(f"switching to logfile in the folder ./logs/{args.jid}")
		jid = args.jid
		os.makedirs(f"./logs/{jid}", exist_ok=True)
		logging.basicConfig(format=f"%(asctime)s — [device {torch.cuda.current_device()}] — %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p — ", filename=f"./logs/{jid}/main.log", filemode="a", level=logging.INFO)
		sys.stdout = StreamToLogger(logging.getLogger('stdout'), dvc)
		sys.stderr = StreamToLogger(logging.getLogger('stderr'), dvc, log_level=logging.ERROR)

	# set ckpt path
	if args.model_id is not None:
		model_id = f"./ckpts/{args.model_id}"
	else:
		model_id = None
	
	rand_seeds = [1234, 3456, 5678]
	# rand_seeds = [1234]

	if args.seed_only is not None:
		rand_seeds = [args.seed_only]

	print_summary(args)
	
	# get instructions based on CLI arg if set (otherwise use default room_type one)
	test_instr_room_type = args.room_type
	if args.do_bedroom_testset:
		test_instr_room_type = "bedroom"
	elif args.do_livingroom_testset:
		test_instr_room_type = "livingroom"

	all_test_instrs = get_test_instrs_all(test_instr_room_type)
	_, _, dataset_test = load_train_val_test_datasets(room_type=test_instr_room_type, use_cached_dataset=True, do_sanity_check=False, accelerator=accelerator)
	
	all_pths = get_pths_dataset_split(test_instr_room_type, "test")

	all_assets_metadata = json.load(open(os.getenv("PTH_ASSETS_METADATA")))
	all_assets_metadata_orig = json.load(open(os.path.join(os.getenv("PTH_3DFUTURE_ASSETS"), "model_info.json")))
	desc_to_category = create_category_lookup(all_assets_metadata_orig, all_assets_metadata)

	path_to_sample_map = {}
	for i, sample in enumerate(dataset_test):
		path = sample.get("pth_orig_file")
		if path in all_pths:
			path_to_sample_map[path] = i

	save_prompts_to = None
	if args.pth_save_prompts_file:
		save_prompts_to = args.pth_save_prompts_file
	
	load_prompts_from = None
	if args.pth_load_prompts_file:
		load_prompts_from = args.pth_load_prompts_file

	if args.do_seq_test:
		seq_dataset = create_sequential_editing_dataset(room_type="all", seeds=[1234, 3456, 5678], n_per_bin=20, max_seq_length=10, p_add=0.8, use_cached_dataset=False)

	respace = ReSpace(
		model_id=model_id,
		dataset_room_type=args.room_type,
		use_gpu=args.use_gpu,
		accelerator=accelerator,
		n_bon_sgllm=args.bon_llm,
		n_bon_assets=1,
		do_prop_sampling_for_prompt=args.do_prop_sampling_for_prompt,
		do_icl_for_prompt=args.do_icl_for_prompt,
		do_class_labels_for_prompt=args.do_class_labels_for_prompt,
		k_few_shot_samples=args.icl_k,
		use_vllm=args.use_vllm,
		do_removal_only=args.do_removal_test,
		save_prompts_to=save_prompts_to,
		load_prompts_from=load_prompts_from,
		do_bon_shuffling=args.do_bon_shuffling,
		bon_shuffling=args.bon_shuffling,
		do_sort_add_asc=args.do_sort_add_asc,
		do_sort_add_desc=args.do_sort_add_desc,
		num_workers=16,
		do_bon_rotation=args.do_bon_rotation or args.do_debug_rotation,
		do_debug_rotation=args.do_debug_rotation,
	)

	all_seeds_cnt_is_success = defaultdict(int)
	all_analysis_data = [] if args.do_removal_test else None
	all_seq_metrics = [] if args.do_seq_test else None

	for rand_seed in rand_seeds:
		set_seeds(rand_seed)

		scene_generation_times = []
		per_object_times = []

		if not args.resume:		
			scenes_processed = 0
		else:
			# count how many files are already in the output folder for this seed and set scenes_processed to that number
			output_folder = os.path.join(args.pth_output, str(rand_seed))
			if os.path.exists(output_folder):
				scenes_processed = len([name for name in os.listdir(output_folder) if name.endswith(".json")])
				print(f"Resuming from seed {rand_seed} - already processed {scenes_processed} scenes")
			else:
				scenes_processed = 0

		all_analysis_data_seed = [] if args.do_removal_test else None
		all_seq_metrics_seed = [] if args.do_seq_test else None

		# for idx, pth_scene in tqdm(enumerate(all_pths)):
		for idx, pth_scene in enumerate(all_pths):
			if scenes_processed >= args.n_test_scenes:
				break

			torch.cuda.empty_cache()
			gc.collect()

			pth_output_file = os.path.join(args.pth_output, str(rand_seed), f"{idx}_{rand_seed}.json")
			if args.resume and os.path.exists(pth_output_file):
				print(f"skipping {pth_output_file} since it already exists")
				continue

			with open(os.path.join(os.getenv("PTH_STAGE_2_DEDUP"), pth_scene), "r", encoding='utf-8') as f:
				scene = json.loads(f.read())

			# Check if room is rectangular (only if flag is set)
			if args.do_rectangular_only:
				bounds_bottom = scene.get("bounds_bottom", [])
				if not is_rectangular_room(bounds_bottom):
					print(f"Skipping non-rectangular room: {pth_scene}")
					continue

			start_time = time.time()

			if args.do_full_scenes:
				final_scene, is_success, num_objects = generate_full_scene(idx, respace, pth_scene, args.pth_output, scene_id=f"scene_{idx}_seed_{rand_seed}")
			elif args.do_seq_test:
				final_scene, sample_metrics = generate_and_eval_full_sequence(idx, respace, seq_dataset[rand_seed])
				all_seq_metrics_seed.append(sample_metrics)
				is_success = sample_metrics["acc_seq"] > 0.5
				num_objects = sample_metrics["seq_length"]
			else:
				sample_idx = path_to_sample_map[pth_scene]
				sample = dataset_test[sample_idx]
				final_scene, is_success, analysis_data, num_objects = generate_instr_scene(idx, respace, rand_seed, sample, args.pth_output, all_test_instrs, desc_to_category, args.do_removal_test)
				
				if args.do_removal_test:
					all_analysis_data_seed.append(analysis_data)

			if is_success:
				all_seeds_cnt_is_success[rand_seed] += 1

			elapsed_time = time.time() - start_time

			print(f"\n>> Scene {idx} (#{scenes_processed}) - Generation time: {elapsed_time:.2f} seconds - Success: {is_success} - Num objects: {num_objects}\n")

			scene_generation_times.append(elapsed_time)

			if num_objects > 0:
				per_object_time = elapsed_time / num_objects
				per_object_times.append(per_object_time)
			
			scenes_processed += 1

			if not args.do_removal_test:
				with open(pth_output_file, 'w') as f:
					json.dump(final_scene, f, indent=4)

		# if do_removal, then save cnt_is_success as json file under ./eval/misc
		if args.do_removal_test:
			pth_output_file = os.path.join(args.pth_output, f"{rand_seed}_cnt_is_success.json")
			with open(pth_output_file, 'w') as f:
				json.dump(all_seeds_cnt_is_success, f, indent=4)
			all_analysis_data.append(all_analysis_data_seed)
		elif args.do_seq_test:
			pth_output_file = os.path.join(args.pth_output, f"{rand_seed}_seq_metrics.json")
			with open(pth_output_file, 'w') as f:
				json.dump(all_seq_metrics_seed, f, indent=4)
			all_seq_metrics.append(all_seq_metrics_seed)

		print(f"Finished seed {rand_seed} - success rate : {np.mean(list(all_seeds_cnt_is_success.values())):.2f} (+/- {np.std(list(all_seeds_cnt_is_success.values())):.2f})\n")

		# Print average time
		avg_time = np.mean(scene_generation_times)
		print(f"Average scene generation time for seed {rand_seed}: {avg_time:.2f} s (+/- {np.std(scene_generation_times):.2f} s) over {len(scene_generation_times)} scenes")

		avg_per_object_time = np.mean(per_object_times)
		std_per_object_time = np.std(per_object_times)
		print(f"Average per-object latency for seed {rand_seed}: {avg_per_object_time:.2f} s (+/- {std_per_object_time:.2f} s) over {len(per_object_times)} scenes")

	# Save all_analysis_data to a JSON file
	if args.do_removal_test and all_analysis_data is not None and len(all_analysis_data) > 0:
		# Create plots directory if it doesn't exist
		os.makedirs("./plots", exist_ok=True)
		
		analysis_output_path = os.path.join("./plots", f"removal_analysis_data_{args.room_type}_test.json")
		with open(analysis_output_path, 'w') as f:
			json.dump(all_analysis_data, f, indent=4)
		print(f"\nSaved analysis data to {analysis_output_path}")
		print(f"Data structure: {len(all_analysis_data)} seeds, {[len(seed_data) for seed_data in all_analysis_data]} scenes per seed")
	elif args.do_seq_test:
		print_and_save_seq_eval_results(all_seq_metrics, args.pth_output, args.room_type, args.bon_llm, args.do_bon_rotation)

	all_successes = list(all_seeds_cnt_is_success.values())
	print(f"\nFINISHED! success rate: {np.mean(all_successes):.2f} (+/- {np.std(all_successes):.2f})\n")
	print(f"saved files to pth: {args.pth_output} with seeds [ {','.join([str(seed) for seed in rand_seeds])} ]")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Author: Martin Juan José Bucher')

	parser.add_argument('--env', dest='env', type=str)
	parser.add_argument('--pth-output', dest='pth_output', type=str)
	parser.add_argument("--room-type", type=str, default=None)
	parser.add_argument("--model-id", type=str, default=None)
	parser.add_argument("--n-test-scenes", type=int, default=1)
	parser.add_argument("--use-gpu", action="store_true", default=False)
	parser.add_argument('--use-logfile', action='store_true', default=False)
	parser.add_argument('--jid', type=str)
	
	parser.add_argument("--do-full-scenes", action="store_true", default=False)

	parser.add_argument("--do-icl-for-prompt", action="store_true", default=False)
	parser.add_argument("--do-class-labels-for-prompt", action="store_true", default=False)
	parser.add_argument("--do-prop-sampling-for-prompt", action="store_true", default=False)
	
	parser.add_argument("--do-bedroom-testset", action="store_true", default=False)
	parser.add_argument("--do-livingroom-testset", action="store_true", default=False)

	parser.add_argument("--resume", action="store_true", default=False)
	parser.add_argument("--seed-only", type=int)

	parser.add_argument("--do-removal-test", action="store_true", default=False)

	parser.add_argument("--do-seq-test", action="store_true", default=False)

	parser.add_argument("--do-rectangular-only", action="store_true", default=False, help="Only process rectangular rooms")

	parser.add_argument("--use-vllm", action="store_true", default=False, help="Use vLLM for faster inference")

	parser.add_argument("--icl-k", type=int, default=10)

	parser.add_argument("--bon-llm", type=int, default=8, help="Best-of-N (BoN) for SG-LLM")

	parser.add_argument("--do-bon-shuffling", action="store_true", default=False, help="Use shuffling best-k")
	parser.add_argument("--bon-shuffling", type=int, default=1, help="Best-of-N (BoN) for shuffling")

	parser.add_argument("--do-bon-rotation", action="store_true", default=False, help="Use test-time augmentation (TTA) with rotations for SG-LLM")
	parser.add_argument("--do-debug-rotation", action="store_true", default=False, help="Debug rotation augmentation: render all 4 rotations for first scene, then exit")

	parser.add_argument("--do-sort-add-asc", action="store_true", default=False, help="Sort add commands by estimated object size (smallest first)")
	parser.add_argument("--do-sort-add-desc", action="store_true", default=False, help="Sort add commands by estimated object size (largest first)")

	parser.add_argument("--pth-save-prompts-file", type=str, default=None, help="Path to save prompts to")
	parser.add_argument("--pth-load-prompts-file", type=str, default=None, help="Path to load prompts from")

	main(parser.parse_args())