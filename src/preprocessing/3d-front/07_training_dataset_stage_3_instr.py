import json
from openai import OpenAI
from tqdm import tqdm
import os
import uuid
import shutil
import numpy as np
import time
import pdb
from copy import deepcopy
from dotenv import load_dotenv
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import csv
import concurrent.futures
import threading
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

from src.utils import remove_and_recreate_folder, get_model
from src.dataset import create_instruction_from_scene, process_scene_sample

def load_and_categorize_scenes(stage_2_path):
    pths_scene_by_type = {room_type: [] for room_type in ROOM_TYPES}
    
    all_pths = [pth for pth in stage_2_path.glob("*") if pth.stem[0].isalnum()]
    # all_pths = all_pths[:50]

    for pth in tqdm(all_pths):
        with open(pth, 'r') as f:
            scene = json.load(f)
            room_type = scene["room_type"]
            if room_type in ROOM_TYPES:
                pths_scene_by_type[room_type].append(str(pth.name))
            pths_scene_by_type["all"].append(str(pth.name))
    
    return pths_scene_by_type

def create_splits(scenes, n_val_test):
    if len(scenes) < 2 * n_val_test:
        raise ValueError(f"Not enough scenes ({len(scenes)}) for val/test splits ({n_val_test} each)")
    
    train_val, test = train_test_split(scenes, test_size=n_val_test, random_state=42)
    train, val = train_test_split(train_val, test_size=n_val_test, random_state=42)
    
    return train, val, test

def export_split(stage_2_path, stage_3_path, room_type, splits):
    print(f"dataset: {room_type}")
    
    # Export CSV
    csv_path = stage_3_path / f"{room_type}_splits.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for split_name, scenes in splits.items():
            print(f"  {split_name}: {len(scenes)}")
            for scene_path in scenes:
                with open(stage_2_path / scene_path, 'r') as sf:
                    scene_data = json.load(sf)
                    # take first five blocks of scne_id that are dividide by "-"
                    orig_scene_id = "-".join(scene_path.split("-")[:5])
                    scene_id = scene_data.get("room_id") + "-" + orig_scene_id
                writer.writerow([scene_id, split_name])
    
    # Create and export pickle file with file names instead of scene_id
    split_filenames = { split_name: [scene_path for scene_path in scenes] for split_name, scenes in splits.items() }
    pickle_path = stage_3_path / f"{room_type}_splits.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(split_filenames, f)
    
    print(f"  Exported {room_type} splits to {csv_path} and {pickle_path}")

def load_existing_splits(stage_3_path, room_types=["bedroom", "livingroom"]):
    splits_by_type = {}
    for room_type in room_types:
        pickle_path = stage_3_path / f"{room_type}_splits.pkl"
        with open(pickle_path, 'rb') as f:
            splits_by_type[room_type] = pickle.load(f)
    return splits_by_type

def create_merged_dataset_splits(pths_scene_by_type, existing_splits, n_val_test):
    
    all_scenes = set(pths_scene_by_type["all"])
    
    # Get bedroom and livingroom test/val scenes
    bedroom_test = set(existing_splits["bedroom"]["test"])
    bedroom_val = set(existing_splits["bedroom"]["val"])
    livingroom_test = set(existing_splits["livingroom"]["test"])
    livingroom_val = set(existing_splits["livingroom"]["val"])

    clean_set = all_scenes - bedroom_test - bedroom_val - livingroom_test - livingroom_val
    
    test_pool = clean_set.union(bedroom_test).union(livingroom_test)
    test_pool = list(test_pool)
    random.shuffle(test_pool)
    test_set = test_pool[:n_val_test]
    
    val_pool = (clean_set - set(test_set)).union(bedroom_val).union(livingroom_val)
    val_pool = list(val_pool)
    random.shuffle(val_pool)
    val_set = val_pool[:n_val_test]

    train_set = list(clean_set - set(test_set) - set(val_set))
    
    print(f"All splits: {len(clean_set)} all, {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
    
    return {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

def generate_test_instructions(stage_2_path, splits_by_type, seeds):
    all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
    all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

    model, tokenizer, max_seq_length = get_model("meta-llama/Llama-3.2-1B-Instruct", use_gpu=True, accelerator=None)
    
    instrs_by_type = {}
    for room_type, splits in splits_by_type.items():
        print(f"generating instructions for {room_type} test scenes")
        test_scenes = splits["test"]
        scene_instrs = {}

        for scene_path in test_scenes:
            scene_instrs[scene_path] = {}
        
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            
            for scene_path in tqdm(test_scenes):
                with open(stage_2_path / scene_path, 'r') as f:
                    scene_data = json.load(f)

                sample = {
                    "scene": scene_data,
                    "n_objects": len(scene_data["objects"]),
                    "room_type": scene_data["room_type"],
                }
                
                # generate instructions for each seed
                # instr_sample = create_instruction_from_scene(sample, all_prompts=all_prompts, do_keep_jids=True)
                _, _, _, instr_sample = process_scene_sample(sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False, do_augm=False, do_full_sg_outputs=False, do_keep_jids=True)
                scene_instrs[scene_path][seed] = instr_sample
        
        instrs_by_type[room_type] = scene_instrs
    
    return instrs_by_type

def update_pickle_files(stage_3_path, splits_by_type, instrs_by_type):
    for room_type, splits in splits_by_type.items():
        pickle_path = stage_3_path / f"{room_type}_splits.pkl"

        # backup existing pickle file with postfix ".backup"
        shutil.copyfile(pickle_path, pickle_path.with_suffix(".backup"))
        
        # Load existing pickle data
        with open(pickle_path, 'rb') as f:
            split_data = pickle.load(f)
        
        # Add instructions data
        if room_type in instrs_by_type:
            split_data["test_instrs"] = instrs_by_type[room_type]
        
        # Save updated pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(split_data, f)
        
        print(f"Updated {room_type} pickle file with test instructions")

def main():
    load_dotenv(".env.stanley")
    # load_dotenv(".env.local")
    
    global ROOM_TYPES
    ROOM_TYPES = ["bedroom", "livingroom", "all"]
    
    N_VAL_TEST = 500
    
    stage_2_path = Path(os.getenv('PTH_STAGE_2_DEDUP'))
    stage_3_path = Path(os.getenv('PTH_STAGE_3'))

    # remove_and_recreate_folder(stage_3_path)

    # classify scenes by room type for splits later
    # pths_scene_by_type = load_and_categorize_scenes(stage_2_path)

    # print pths where we have the same room_id for different files
    # pths_bedrooms = pths_scene_by_type["bedroom"]
    # pths_for_room_id = defaultdict(list)
    # for pth in pths_bedrooms:
    #     with open(pth, 'r') as f:
    #         scene = json.load(f)
    #         room_id = scene["room_id"]
    #         if pths_for_room_id[room_id]:
    #             pths_for_room_id[room_id].append(pth)
    #             print(f"room_id: {room_id} has multiple files: {pths_for_room_id[room_id]}, {pth}")
    #         else:
    #             pths_for_room_id[room_id] = [pth]
    
    # create splits for bedroom and livingroom
    # for room_type in ROOM_TYPES[:2]:
    #     pths_room_type = pths_scene_by_type[room_type]
    #     train, val, test = create_splits(pths_room_type, N_VAL_TEST)
    #     splits = {
    #         "train": train,
    #         "val": val,
    #         "test": test
    #     }
    #     export_split(stage_2_path, stage_3_path, room_type, splits)

    # create clean split for "all" split
    # existing_splits = load_existing_splits(stage_3_path, room_types=["bedroom", "livingroom"])
    # merged_splits = create_merged_dataset_splits(pths_scene_by_type, existing_splits, N_VAL_TEST)
    # export_split(stage_2_path, stage_3_path, "all", merged_splits)

    # generate instructions for test scenes with 3 different random seeds
    all_splits = load_existing_splits(stage_3_path, room_types=["bedroom", "livingroom", "all"])
    instructions_by_type = generate_test_instructions(stage_2_path, all_splits, [1234, 3456, 5678])
    update_pickle_files(stage_3_path, all_splits, instructions_by_type)

if __name__ == "__main__":
    main()