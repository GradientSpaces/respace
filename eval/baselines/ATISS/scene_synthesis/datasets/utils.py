# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

from collections import defaultdict
import numpy as np
import json
import os
import pickle
import pdb
import math
import trimesh

from .threed_front_scene import Asset, ModelInfo, Room, ThreedFutureModel, \
    ThreedFutureExtra


def parse_threed_front_scenes(
    dataset_directory, path_to_model_info, path_to_models,
    path_to_room_masks_dir=None
):
    if os.getenv("PATH_TO_SCENES"):
        scenes = pickle.load(open(os.getenv("PATH_TO_SCENES"), "rb"))
    else:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        model_info = mf.model_info

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")
        ]

        # (2): black nightstand and lamp
        # path_to_scene_layouts = [pth for pth in path_to_scene_layouts if "6f3094b8-689f-4f0c-adb2-748fbc81ec8a" in pth]

        # (3): lamp, desk and chair
        # path_to_scene_layouts = [pth for pth in path_to_scene_layouts if "6b774494-78d2-4def-a1df-24e4d907e796" in pth]

        # from scene_synthesis.datasets import CSVSplitsBuilder
        # splits_builder = CSVSplitsBuilder("../../../../data/3D-FRONT-martin-rooms-stage-3/bedroom_splits.csv")
        # split_scene_ids = splits_builder.get_splits(["train", "val"])

        scenes = []
        unique_room_ids = set()
        total_num_objects = 0

        # Start parsing the dataset
        print("Loading dataset ", end="")
        for i, m in enumerate(path_to_scene_layouts):
            with open(m) as f:
                data = json.load(f)
                
                # Parse the furniture of the scene
                # furniture_in_scene = defaultdict()
                furniture_in_scene = {}
                
                for ff in data["furniture"]:
                    # print(f"UID: {ff['uid']}, JID: {ff['jid']}, Valid: {ff.get('valid', 'None')}")
                    # if "valid" in ff and ff["valid"]:
                    # if ff.get("valid") != None:
                        # print("valid")
                    if model_info.get(ff["jid"]) is not None:
                        # print(ff.get("uid"))
                        furniture_in_scene[ff["uid"]] = dict(
                            model_uid=ff["uid"],
                            model_jid=ff["jid"],
                            model_info=model_info[ff["jid"]]
                        )
                    # else:
                        # print("not valid")

                # Parse the extra meshes of the scene e.g walls, doors,
                # windows etc.
                meshes_in_scene = defaultdict()
                for mm in data["mesh"]:
                    meshes_in_scene[mm["uid"]] = dict(
                        mesh_uid=mm["uid"],
                        mesh_jid=mm["jid"],
                        mesh_xyz=np.asarray(mm["xyz"]).reshape(-1, 3),
                        mesh_faces=np.asarray(mm["faces"]).reshape(-1, 3),
                        mesh_type=mm["type"]
                    )

                # Parse the rooms of the scene
                scene = data["scene"]
                # Keep track of the parsed rooms
                rooms = []
                for rr in scene["room"]:
                    # Keep track of the furniture in the room
                    furniture_in_room = []
                    # Keep track of the extra meshes in the room
                    extra_meshes_in_room = []
                    # Flag to keep track of invalid scenes
                    is_valid_scene = True

                    scene_id_custom = rr["instanceid"] + "-" + data["uid"]

                    for cc in rr["children"]:
                        if cc["ref"] in furniture_in_scene:
                            tf = furniture_in_scene[cc["ref"]]

                            # Check for NaN values in position, rotation, and scale
                            has_nan = False
                            for value in cc["pos"] + cc["rot"] + cc["scale"]:
                                if math.isnan(value):
                                    has_nan = True
                                    break
                            # Skip this furniture if it has NaN values
                            if has_nan:
                                print("Skipping furniture with NaN values: ", tf["model_uid"])
                                continue

                            # If scale is very small/big ignore this scene
                            if any(si < 1e-5 for si in cc["scale"]):
                                print("Skipping furniture with small scale")
                                # print("scale too small !")
                                # is_valid_scene = False
                                # break
                                continue

                            # if any(si > 5 for si in cc["scale"]):
                            #     # print("scale too big !")
                            #     # is_valid_scene = False
                            #     # break
                            #     pass

                            # Check if the product of size values is too large
                            raw_model_path = f"/home/martinbucher/git/stan-24-sgllm/data/3D-FUTURE-assets/{tf['model_jid']}/raw_model.obj"
                            try:
                                raw_mesh = trimesh.load(raw_model_path, force="mesh", ignore_materials=True, process=False)
                            except Exception as e:
                                print(e)
                                print(f"error loading mesh: {raw_model_path}. loading with fallback...")
                                try:
                                    print(e)
                                    raw_mesh = trimesh.load(raw_model_path)
                                except Exception as e:
                                    print("loading glb file !!")
                                    raw_mesh = trimesh.load(raw_model_path.replace('.obj', '.glb'))

                            real_bbox = raw_mesh.bounds
                            real_size = (real_bbox[1] - real_bbox[0]).tolist()
                        
                            size_prod_threshold = 150.0
                            size_prod = real_size[0] * real_size[1] * real_size[2]
                            if size_prod > size_prod_threshold:
                                print("Skipping furniture with large size product. size:", real_size)
                                continue

                            # size_prod_threshold = 150.0
                            # # Check if the product of the scale values is too large
                            # size_prod = cc["scale"][0] * cc["scale"][1] * cc["scale"][2]
                            # if size_prod > size_prod_threshold:
                            #     print("Skipping furniture with large size product")
                            #     continue

                            furniture_in_room.append(ThreedFutureModel(
                               tf["model_uid"],
                               tf["model_jid"],
                               tf["model_info"],
                               cc["pos"],
                               cc["rot"],
                               cc["scale"],
                               path_to_models
                            ))
                        elif cc["ref"] in meshes_in_scene:
                            mf = meshes_in_scene[cc["ref"]]
                            extra_meshes_in_room.append(ThreedFutureExtra(
                                mf["mesh_uid"],
                                mf["mesh_jid"],
                                mf["mesh_xyz"],
                                mf["mesh_faces"],
                                mf["mesh_type"],
                                cc["pos"],
                                cc["rot"],
                                cc["scale"]
                            ))
                        else:
                            continue

                    # missing_id_1 = "SecondBedroom-10129-6f3094b8-689f-4f0c-adb2-748fbc81ec8a"
                    # missing_id_2 = "SecondBedroom-137066-6b774494-78d2-4def-a1df-24e4d907e796"

                    # if scene_id_custom == missing_id_1 or scene_id_custom == missing_id_2:
                    #     print("missing scene: ", scene_id_custom)
                    #     print(len(furniture_in_room))
                    #     print(is_valid_scene)

                    #     for furn in furniture_in_room:
                    #         print(furn.model_info)

                    if len(furniture_in_room) > 1 and is_valid_scene:
                        # Check whether a room with the same instanceid has
                        # already been added to the list of rooms
                        # if rr["instanceid"] not in unique_room_ids:
                        #     unique_room_ids.add(rr["instanceid"])
                        #     # Add to the list
                        #     rooms.append(Room(
                        #         rr["instanceid"],                # scene_id
                        #         rr["type"].lower(),              # scene_type
                        #         furniture_in_room,               # bounding boxes
                        #         extra_meshes_in_room,            # extras e.g. walls
                        #         m.split("/")[-1].split(".")[0],  # json_path
                        #         path_to_room_masks_dir
                        #     ))

                        # unique_room_ids.add(rr["instanceid"])

                        if scene_id_custom not in unique_room_ids:
                            unique_room_ids.add(scene_id_custom)

                            total_num_objects += len(furniture_in_room)

                            # Add to the list
                            rooms.append(Room(
                                scene_id_custom,                # scene_id
                                rr["type"].lower(),              # scene_type
                                furniture_in_room,               # bounding boxes
                                extra_meshes_in_room,            # extras e.g. walls
                                m.split("/")[-1].split(".")[0],  # json_path
                                path_to_room_masks_dir
                            ))
                        else:
                            print("duplicate scene: ", rr["instanceid"])

                scenes.append(rooms)
                
            s = "{:5d} / {:5d} / {:5d} / {:5d}".format(i, len(path_to_scene_layouts), len(scenes), total_num_objects)
            print(s, flush=True, end="\b"*len(s))
        print()

        scenes = sum(scenes, [])

        # for each room in scenes check how many are in split_scene_ids
        # scenes = [scene for scene in scenes if scene.scene_id in split_scene_ids]
        # print("Number of rooms in split: ", len(scenes))

        # get room ids in split_scene_ids that are not in scenes
        # missing_room_ids = [room_id for room_id in split_scene_ids if room_id not in [scene.scene_id for scene in scenes]]
        # print("Missing room ids: ", missing_room_ids)

        # for each missing room id check if its in train or val split
        # train_split_scene_ids = splits_builder.get_splits(["train"])
        # val_split_scene_ids = splits_builder.get_splits(["val"])
        # for missing_room_id in missing_room_ids:
        #     if missing_room_id in train_split_scene_ids:
        #         print("Missing room id in train split: ", missing_room_id)
        #     elif missing_room_id in val_split_scene_ids:
        #         print("Missing room id in val split: ", missing_room_id)
        #     else:
        #         print("Missing room id in unknown split: ", missing_room_id)

        print("Number of rooms: ", len(scenes))

        pickle.dump(scenes, open("/tmp/threed_front.pkl", "wb"))

    return scenes


def parse_threed_future_models(
    dataset_directory, path_to_models, path_to_model_info
):
    if os.getenv("PATH_TO_3D_FUTURE_OBJECTS"):
        furnitures = pickle.load(
            open(os.getenv("PATH_TO_3D_FUTURE_OBJECTS"), "rb")
        )
    else:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        model_info = mf.model_info

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")
        ]
        # List to keep track of all available furniture in the dataset
        furnitures = []
        unique_furniture_ids = set()

        # Start parsing the dataset
        print("Loading dataset ", end="")
        for i, m in enumerate(path_to_scene_layouts):
            with open(m) as f:
                data = json.load(f)
                # Parse the furniture of the scene
                furniture_in_scene = defaultdict()
                for ff in data["furniture"]:
                    if "valid" in ff and ff["valid"]:
                        furniture_in_scene[ff["uid"]] = dict(
                            model_uid=ff["uid"],
                            model_jid=ff["jid"],
                            model_info=model_info[ff["jid"]]
                        )
                # Parse the rooms of the scene
                scene = data["scene"]
                for rr in scene["room"]:
                    # Flag to keep track of invalid scenes
                    # is_valid_scene = True
                    
                    for cc in rr["children"]:
                        if cc["ref"] in furniture_in_scene:
                            tf = furniture_in_scene[cc["ref"]]
                            
                            # # If scale is very small/big ignore this scene
                            # if any(si < 1e-5 for si in cc["scale"]):
                            #     is_valid_scene = False
                            #     break
                            
                            # if any(si > 5 for si in cc["scale"]):
                            #     is_valid_scene = False
                            #     break

                            # martin: check for NaN values in position, rotation, and scale
                            has_nan = False
                            for value in cc["pos"] + cc["rot"] + cc["scale"]:
                                if math.isnan(value):
                                    has_nan = True
                                    break
                            # Skip this furniture if it has NaN values
                            if has_nan:
                                print("Skipping furniture with NaN values: ", tf["model_uid"])
                                continue

                            # If scale is very small/big ignore this scene
                            if any(si < 1e-5 for si in cc["scale"]):
                                print("Skipping furniture with small scale")
                                # print("scale too small !")
                                # is_valid_scene = False
                                # break
                                continue

                            # Check if the product of size values is too large
                            raw_model_path = f"/home/martinbucher/git/stan-24-sgllm/data/3D-FUTURE-assets/{tf['model_jid']}/raw_model.obj"
                            try:
                                raw_mesh = trimesh.load(raw_model_path, force="mesh", ignore_materials=True, process=False)
                            except Exception as e:
                                print(e)
                                print(f"error loading mesh: {raw_model_path}. loading with fallback...")
                                try:
                                    print(e)
                                    raw_mesh = trimesh.load(raw_model_path)
                                except Exception as e:
                                    print("loading glb file !!")
                                    raw_mesh = trimesh.load(raw_model_path.replace('.obj', '.glb'))

                            real_bbox = raw_mesh.bounds
                            real_size = (real_bbox[1] - real_bbox[0]).tolist()
                        
                            size_prod_threshold = 150.0
                            size_prod = real_size[0] * real_size[1] * real_size[2]
                            if size_prod > size_prod_threshold:
                                print("Skipping furniture with large size product. size:", real_size)
                                continue

                            # size_prod_threshold = 150.0
                            # # Check if the product of the scale values is too large
                            # size_prod = cc["scale"][0] * cc["scale"][1] * cc["scale"][2]
                            # if size_prod > size_prod_threshold:
                            #     print("Skipping furniture with large size product")
                            #     continue

                            if tf["model_uid"] not in unique_furniture_ids:
                                unique_furniture_ids.add(tf["model_uid"])
                                furnitures.append(ThreedFutureModel(
                                    tf["model_uid"],
                                    tf["model_jid"],
                                    tf["model_info"],
                                    cc["pos"],
                                    cc["rot"],
                                    cc["scale"],
                                    path_to_models
                                ))
                        else:
                            continue
            s = "{:5d} / {:5d}".format(i, len(path_to_scene_layouts))
            print(s, flush=True, end="\b"*len(s))
        print()

        pickle.dump(furnitures, open("/tmp/threed_future_model.pkl", "wb"))

    return furnitures

