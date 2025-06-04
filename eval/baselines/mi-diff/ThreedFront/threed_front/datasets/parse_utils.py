# 
# Modified from: 
#   https://github.com/nv-tlabs/ATISS.
# 

from collections import defaultdict
import numpy as np
import json
import os
import pickle
import math
import trimesh

from .threed_front_scene import ModelInfo, Room, ThreedFutureModel, \
     ThreedFutureExtra


def parse_threed_front_scenes(
    dataset_directory=None, path_to_model_info=None, path_to_models=None,
    path_to_room_masks_dir=None
):
    # The pickled dataset file
    output_path = os.getenv("PATH_TO_SCENES")
    if output_path is None:
        output_path = "/tmp/threed_front.pkl"

    # Load or compute
    if os.path.exists(output_path):
        scenes = pickle.load(open(output_path, "rb"))
    else:
        scenes = parse_threed_front_scenes_from_dataset(
            dataset_directory, path_to_model_info, path_to_models,
            path_to_room_masks_dir
        )
        pickle.dump(scenes, open(output_path, "wb"))
    
    return scenes


def parse_threed_front_scenes_from_dataset(
    dataset_directory, path_to_model_info, path_to_models,
    path_to_room_masks_dir=None
):
    # Parse the model info
    mf = ModelInfo.from_file(path_to_model_info)
    model_info = mf.model_info

    path_to_scene_layouts = [
        os.path.join(dataset_directory, f)
        for f in sorted(os.listdir(dataset_directory))
        if f.endswith(".json")
    ]
    scenes = []
    unique_room_ids = set()
    # Start parsing the dataset
    print("Loading dataset ", end="")
    for i, m in enumerate(path_to_scene_layouts):
        with open(m) as f:
            data = json.load(f)
            # Parse the furniture of the scene
            furniture_in_scene = defaultdict()

            for ff in data["furniture"]:

                # if "valid" in ff and ff["valid"]:
                if model_info.get(ff["jid"]) is not None:
                    
                    furniture_in_scene[ff["uid"]] = dict(
                        model_uid=ff["uid"],
                        model_jid=ff["jid"],
                        model_info=model_info[ff["jid"]]
                    )

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
                            # is_valid_scene = False
                            # break
                            # ignore furniture
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

                        # if any(si > 5 for si in cc["scale"]):
                            # is_valid_scene = False
                            # break
                            # pass

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
                if len(furniture_in_room) > 1 and is_valid_scene:
                    # Check whether a room with the same instanceid has
                    # already been added to the list of rooms
                    
                    if scene_id_custom not in unique_room_ids:
                    # if rr["instanceid"] not in unique_room_ids:
                        # unique_room_ids.add(rr["instanceid"])
                        unique_room_ids.add(scene_id_custom)
                        # Add to the list
                        rooms.append(Room(
                            # rr["instanceid"],              # scene_id
                            scene_id_custom,                 # scene_id
                            rr["type"].lower(),              # scene_type
                            furniture_in_room,               # bounding boxes
                            extra_meshes_in_room,            # extras e.g. walls
                            m.split("/")[-1].split(".")[0],  # json_path
                            path_to_room_masks_dir
                        ))
                    else:
                        print("duplicate scene: ", rr["instanceid"])

            scenes.append(rooms)

        # s = "{:5d} / {:5d}".format(i + 1, len(path_to_scene_layouts))
        s = "{:5d} / {:5d} / {:5d}".format(i, len(path_to_scene_layouts), len(scenes))
        print(s, flush=True, end="\b"*len(s))
    print()

    # Flatten the scenes list
    scenes = sum(scenes, [])

    print("Number of rooms: ", len(scenes))

    return scenes
