# 
# Modified from: 
#   https://github.com/QiuhongAnnaWei/LEGO-Net/blob/main/data/preprocess_TDFront.py
# 
"""Script to add floor_plan_ordered_corners and floor_plan_boundary_points_normals
for each room given a dataset directory. The results will be added to "boxes.npz" 
in each room subdirectory, and the feature bounds to "dataset_stats.txt".
"""
import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
from threed_front.datasets.floorplan_utils import process_floorplan_iterative_closest_point, scene_sample_fpbp
from tqdm import tqdm
import pdb

def preprocess_floor_plan(room_data_dir, room_side, num_sampled_points=256, overwrite=False):
    """ Generates all 3 representations of floor plans from data in boxes npz and write them to boxes npz.
    """
    print("--preprocess_floor_plan: start--")
    max_nfpoc = 0
    scene_data_list = []
    for e in tqdm(os.listdir(room_data_dir)): # train, val, and test
        if not os.path.isdir(os.path.join(room_data_dir, e)): continue

        scene_data = np.load(os.path.join(room_data_dir, e, "boxes.npz"))
        scene_data = dict(scene_data)

        # Update "room_layout" in case it is not consistent with room_mask.png
        img = Image.open(os.path.join(room_data_dir, e, "room_mask.png"))
        scene_data["room_layout"] = np.asarray(img)[:, :, 0:1]

        # Find ordered corners in floor_plan_vertices using contour
        ordered_corners = process_floorplan_iterative_closest_point(scene_data, room_side) # centered, not normalized
        scene_data["floor_plan_ordered_corners"] = ordered_corners.astype(np.float32)  # [?, 2]
        max_nfpoc = max(max_nfpoc, scene_data["floor_plan_ordered_corners"].shape[0])
        
        # Sample boundary points and normals based on floor_plan_ordered_corners
        scene_fpbpn = scene_sample_fpbp(scene_data["floor_plan_ordered_corners"], num_sampled_points=num_sampled_points) # boundary centered, not normalized
        if np.isnan(np.min(scene_fpbpn.astype(np.float32))) == True: 
            print(scene_fpbpn.astype(np.float32))
        scene_data["floor_plan_boundary_points_normals"] = scene_fpbpn.astype(np.float32) # [nfpbp, 4]

        scene_data_list.append(scene_data)
        
        if overwrite:
            np.savez_compressed(os.path.join(room_data_dir, e, "boxes.npz"),  **scene_data)
            
    if overwrite:
        # Update "dataset_stats.txt"
        path_to_json = os.path.join(room_data_dir, "dataset_stats.txt")
        with open(path_to_json, "r") as f:
            train_stats = json.load(f)

        # pdb.set_trace()
        
        # fpoc and fpbpn[:2] should have the same bounds, fpbpn[2:] are normals bounded by [-1, 1]
        fpoc_min = np.min([scene_data["floor_plan_ordered_corners"].min(axis=0) for scene_data in scene_data_list], axis=0)
        fpoc_max = np.max([scene_data["floor_plan_ordered_corners"].max(axis=0) for scene_data in scene_data_list], axis=0)
        fpbpn_min = np.min([scene_data["floor_plan_boundary_points_normals"].min(axis=0) for scene_data in scene_data_list], axis=0)
        fpbpn_max = np.max([scene_data["floor_plan_boundary_points_normals"].max(axis=0) for scene_data in scene_data_list], axis=0)
        fp_min = np.minimum(fpoc_min, fpbpn_min[:2]).round(5)
        fp_max = np.maximum(fpoc_max, fpbpn_max[:2]).round(5)

        bounds_fp = {
            "bounds_fpoc": fp_min.tolist() + fp_max.tolist(),
            "bounds_fpbpn": fp_min.tolist() + [-1.0, -1.0, ] + fp_max.tolist() + [1.0, 1.0, ]
        }
        train_stats.update(bounds_fp)

        with open(path_to_json, "w") as f:
            json.dump(train_stats, f)
        print("Updated bounds in dataset_stats.txt", bounds_fp)
    
    print("--preprocess_floor_plan: done (max nfpoc: {})--".format(max_nfpoc))
    return scene_data_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Update boxes.npz with floor_plan_ordered_corners and floor_plan_boundary_points_normals"
    )
    parser.add_argument(
        "data_directory",
        help="Path to preprocessed data directory"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=None,
        help="Approximate size of a room along a side to align floor_plan_vertices in boxes.npz and room_mask.png "
        "(default:3.1 for bedroom and library, 6.1 for diningroom and livingroom)"
    )
    parser.add_argument(
        "--n_sampled_points",
        type=int,
        default=256,
        help="Number of floor plan boundary and normals to be sampled (default:256)"
    )

    args = parser.parse_args(argv)

    if args.room_side is None:
        room_type = next((
            type for type in ["diningroom", "livingroom", "bedroom", "library"] \
            if type in os.path.basename(os.path.normpath(args.data_directory))
            ), None)
        args.room_side = 3.1 if room_type in ["bedroom", "library"] else 6.1
        print("Use default room_side {} for {} dataset.".format(args.room_side, room_type))
    
    preprocess_floor_plan(args.data_directory, args.room_side, args.n_sampled_points, True)


if __name__ == '__main__':
    main(sys.argv[1:])
