# 
# Modified from: 
#   https://github.com/nv-tlabs/ATISS.
# 
"""Script to parse the 3D-FRONT data scenes to numpy files in order to avoid 
I/O overhead in training.
"""
import argparse
import json
import os
import sys
import pdb

import numpy as np
from tqdm import tqdm
import seaborn as sns

from threed_front.datasets import filter_function
from threed_front.datasets.threed_front_encoding_base import get_basic_encoding
from threed_front.rendering import scene_from_args, get_floor_plan, \
    get_textured_objects_in_scene, render_projection
from threed_front.simple_3dviz_setup import ORTHOGRAPHIC_PROJECTION_SCENE
from utils import PATH_TO_PROCESSED_DATA, PATH_TO_PICKLED_3D_FRONT_DATASET, \
    PATH_TO_DATASET_FILES, PATH_TO_FLOOR_PLAN_TEXTURES, load_pickled_threed_front


def main(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )
    parser.add_argument(
        "dataset_filtering",
        choices=[
            "threed_front_bedroom",
            "threed_front_livingroom",
            "threed_front_diningroom",
            "threed_front_library",
            "no-filtering",
        ],
        help="The type of dataset filtering to be used"
    )
    parser.add_argument(
        "--output_directory",
        default=PATH_TO_PROCESSED_DATA,
        help="Path to output directory (default: output/3d_front_processed/<room_type>)"
    )
    parser.add_argument(
        "--path_to_pickled_3d_front_dataset",
        default=PATH_TO_PICKLED_3D_FRONT_DATASET,
        help="Path to pickled 3D-FRONT dataset (default: output/threed_front.pkl)"
    )
    parser.add_argument(
        "--path_to_dataset_files_directory",
        default=PATH_TO_DATASET_FILES,
        help="Path to directory storing black_list.txt, invalid_threed_front_rooms.txt, "
        "and <room_type>_threed_front_splits.csv",
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default=PATH_TO_FLOOR_PLAN_TEXTURES,
        help="Path to floor texture image directory"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=None,
        help="The size of the room along a side (default:3.1 or 6.1)"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="Filter out lamps when extracting objects in the scene"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="Color objects by semantic label, and set floor plan to white"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="Remove the floor plane"
    )
    # add objfeat
    parser.add_argument(
        "--add_objfeats",
        action="store_true",
        help="Add object point cloud features (make sure raw_model_norm_pc_lat.npz "
        "and raw_model_norm_pc_lat32.npz exist in the raw dataset directory)"
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene"
    )

    args = parser.parse_args(argv)

    room_type = args.dataset_filtering.split("_")[-1]
    print(f"Room type: {room_type}")

    # Create the scene
    scene_params = ORTHOGRAPHIC_PROJECTION_SCENE
    scene_params["camera_position"] = args.camera_position

    if args.room_side is None:
        scene_params["room_side"] = 3.1 if room_type in ["bedroom", "library"] \
            else 6.1
    else:
        scene_params["room_side"] = args.room_side
    if args.without_floor:
        scene_params["background"] = (1, 1, 1, 1)
    
    scene = scene_from_args(scene_params)

    print("Room side:", scene_params["room_side"])

    layout_image = "rendered_scene{}_{}{}.png".format(
        "_notexture" if args.no_texture else "",
        scene_params["window_size"][0],
        "_nofloor" if args.without_floor else ""
    )
    print("Layout image name: {}".format(layout_image))

    # Check if output directory exists and if it doesn't create it
    args.output_directory = args.output_directory.format(room_type)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
        # overwrite_subdirectory = 2
        overwrite_subdirectory = "2" # martin: change to string to fix boxes.npz ?
    elif len(os.listdir(args.output_directory)) > 0:
        print(f"Warning: a non-empty output directory {args.output_directory} exists.")
        overwrite_subdirectory = None
        while overwrite_subdirectory not in {"0", "1", "2"}:
            overwrite_subdirectory = \
                input("Do you want to overwrite existing subdirectories? "
                      "[0. skip; 1. add/overwrite rendered images; 2. overwrite all.] ")
    
    # Set floor texture/color (color has higher priority if args.no_texture)
    if args.without_floor:
        floor_color = None
        floor_textures = [None]
    elif args.no_texture:
        floor_color = (1, 1, 1, 1)  # white floor
        floor_textures = [None]
    else:
        floor_color = None
        floor_textures = \
            [os.path.join(args.path_to_floor_plan_textures, fi)
                for fi in os.listdir(args.path_to_floor_plan_textures)]

    config = {
        "filter_fn":                 args.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": f"{args.path_to_dataset_files_directory}/invalid_threed_front_rooms.txt",
        "path_to_invalid_bbox_jids": f"{args.path_to_dataset_files_directory}/black_list.txt",
        # "annotation_file": f"{args.path_to_dataset_files_directory}/{room_type}_threed_front_splits.csv"
        "annotation_file": args.annotation_file,
    }

    # Initially, we only consider the train split to compute the dataset
    # statistics, e.g the translations, sizes and angles bounds
    filter_fn = filter_function(config, ["train", "val"], args.without_lamps)
    train_dataset = load_pickled_threed_front(
        args.path_to_pickled_3d_front_dataset, filter_fn
    )
    print("Loaded dataset with {} training rooms".format(len(train_dataset)))

    # Compute the bounds for the translations, sizes and angles in the dataset.
    # This will then be used to properly align rooms.
    tr_bounds = train_dataset.centroids
    si_bounds = train_dataset.sizes
    train_dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        "bounds_angles": [float(b) for b in train_dataset.angles],
        "class_labels": train_dataset.class_labels,
        "object_types": train_dataset.object_types,
        "class_frequencies": train_dataset.class_frequencies,
        "class_order": train_dataset.class_order,
        "count_furniture": train_dataset.count_furniture
    }
    if args.add_objfeats:
        if train_dataset.objfeats is not None:
            train_dataset_stats.update({
                "bounds_objfeats": [float(b) for b in train_dataset.objfeats],
            })
        if train_dataset.objfeats_32 is not None:
            train_dataset_stats.update({
                "bounds_objfeats_32": [float(b) for b in train_dataset.objfeats_32]
            })

    path_to_json = os.path.join(args.output_directory, "dataset_stats.txt")
    with open(path_to_json, "w") as f:
        json.dump(train_dataset_stats, f)
    print("Saving training statistics for dataset with bounds: {} to {}".format(
            train_dataset.bounds, path_to_json))

    # Load full dataset
    filter_fn = filter_function(config, ["train", "val", "test"], args.without_lamps)
    dataset = load_pickled_threed_front(
        args.path_to_pickled_3d_front_dataset, filter_fn
    )
    print("Loaded full dataset with {} rooms".format(len(dataset)))

    encoded_dataset = get_basic_encoding(
        dataset, box_ordering=None, add_objfeats=args.add_objfeats
    )

    if args.no_texture:
        color_palette = sns.color_palette('hls', dataset.n_object_types)
    
    for es, ss in tqdm(zip(encoded_dataset, dataset)):
        # Create a separate folder for each room
        room_directory = os.path.join(args.output_directory, ss.uid)

        # Skip existing room directory if the user does not want to overwrite
        if os.path.exists(room_directory) and overwrite_subdirectory == "0":
            continue
        else:
            os.makedirs(room_directory, exist_ok=True)

        # 3D-FUTURE model ids
        uids = [bi.model_uid for bi in ss.bboxes]
        jids = [bi.model_jid for bi in ss.bboxes]

        floor_plan_vertices, floor_plan_faces = ss.floor_plan

        # Render and save the room mask as an image
        if args.without_floor:
            room_mask = None
        else:
            room_mask = render_projection(
                scene, [get_floor_plan(ss)[0]], (1.0, 1.0, 1.0), "flat",
                None if overwrite_subdirectory == "1" else \
                    os.path.join(room_directory, "room_mask.png")
            )[:, :, 0:1]

        # Save layout to boxes.npz
        if overwrite_subdirectory == "2":
            data_dict = dict(
                uids=uids,
                jids=jids,
                scene_id=ss.scene_id,
                scene_uid=ss.uid,
                scene_type=ss.scene_type,
                json_path=ss.json_path,
                room_layout=room_mask,
                floor_plan_vertices=floor_plan_vertices,
                floor_plan_faces=floor_plan_faces,
                floor_plan_centroid=ss.floor_plan_centroid,
                class_labels=es["class_labels"],
                translations=es["translations"],
                sizes=es["sizes"],
                angles=es["angles"]
            )
            if args.add_objfeats:
                data_dict.update(
                    dict(objfeats=es["objfeats"], objfeats_32=es["objfeats_32"])
                )
            np.savez_compressed(
                os.path.join(room_directory, "boxes"), **data_dict
            )

        # Render a top-down orthographic projection of the room at a
        # specific pixel resolutin
        path_to_image = os.path.join(room_directory, layout_image)
        
        # object renderables
        if args.no_texture:
            # read class labels and get the color of each object
            class_labels = es["class_labels"]
            class_index = class_labels.argmax(axis=1).tolist()
            cc = [color_palette[ind] for ind in class_index]
            renderables = get_textured_objects_in_scene(ss, colors=cc)
        else:
            # use default texture files                
            renderables = get_textured_objects_in_scene(ss)
        
        # floor plan renderable
        if not args.without_floor:
            texture = np.random.choice(floor_textures)
            floor_plan, _, _ = get_floor_plan(
                ss, texture=texture, color=floor_color, 
                with_trimesh=False, with_room_mask=False
            )
            renderables.append(floor_plan)

        render_projection(
            scene, renderables, color=None, mode="shading",
            frame_path=path_to_image
        )


if __name__ == "__main__":
    main(sys.argv[1:])
