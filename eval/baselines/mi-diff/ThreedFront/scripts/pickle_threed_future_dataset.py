#
# Modified from: 
#   https://github.com/nv-tlabs/ATISS.
#
"""Script used for pickling the 3D Future dataset in order to be subsequently
used by our scripts.
"""
import argparse
import os
import sys
import pickle

from threed_front.datasets import filter_function
from threed_front.datasets.threed_future_dataset import ThreedFutureDataset
from utils import PATH_TO_PICKLED_3D_FUTURE_MODEL, PATH_TO_PICKLED_3D_FRONT_DATASET, \
    PATH_TO_DATASET_FILES, load_pickled_threed_front


def main(argv):
    parser = argparse.ArgumentParser(
        description="Pickle the 3D Future dataset"
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
        "--output_path",
        default=PATH_TO_PICKLED_3D_FUTURE_MODEL,
        help="Path to output directory"
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
        "--without_lamps",
        action="store_true",
        help="Filter out lamps when extracting objects in the scene"
    )

    parser.add_argument(
        "--annotation-file",
        type=str,
    )

    args = parser.parse_args(argv)
    
    room_type = args.dataset_filtering.split('_')[-1]
    output_path = args.output_path.format(room_type)
    if os.path.exists(output_path):
        input(f"Warning: {output_path} exists. Press any key to overwrite...")
    
    # Set up config for filtering
    config = {
        "filter_fn":                 args.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": os.path.join(args.path_to_dataset_files_directory, "invalid_threed_front_rooms.txt"),
        "path_to_invalid_bbox_jids": os.path.join(args.path_to_dataset_files_directory, "black_list.txt"),
        # "annotation_file": os.path.join(args.path_to_dataset_files_directory, f"{room_type}_threed_front_splits.csv")
        "annotation_file": args.annotation_file,
    }

    # Extract scenes from train split
    filter_fn = filter_function(config, ["train", "val", "test"], args.without_lamps)
    scenes_dataset = load_pickled_threed_front(
        args.path_to_pickled_3d_front_dataset, filter_fn
    )
    print("Loading dataset with {} rooms".format(len(scenes_dataset)))

    # Collect the set of objects in the scenes
    objects = {}
    for scene in scenes_dataset:
        for obj in scene.bboxes:
            objects[obj.model_jid] = obj
    objects = [vi for vi in objects.values()]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    objects_dataset = ThreedFutureDataset(objects)
    with open(output_path, "wb") as f:
        pickle.dump(objects_dataset, f)
    print("Saved result to: {}".format(output_path))


if __name__ == "__main__":
    main(sys.argv[1:])
