"""Script used for parsing the 3D-FRONT data scenes into a list of Room objects.
The result file is threed_front.pkl.
"""
import argparse
import os
import sys
import pickle

from threed_front.datasets.parse_utils import parse_threed_front_scenes_from_dataset
from utils import PATH_TO_PICKLED_3D_FRONT_DATASET


def main(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_3d_future_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "--output_path",
        default=PATH_TO_PICKLED_3D_FRONT_DATASET,
        help="Output path (default: output/threed_front.pkl)"
    )

    args = parser.parse_args(argv)

    if os.path.exists(args.output_path):
        input(f"Warning: {args.output_path} exists. Press any key to overwrite...")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    scenes = parse_threed_front_scenes_from_dataset(
        dataset_directory=args.path_to_3d_front_dataset_directory,
        path_to_model_info=args.path_to_3d_future_model_info,
        path_to_models=args.path_to_3d_future_dataset_directory,
    )
    pickle.dump(scenes, open(args.output_path, "wb"))
    print("Saved result to:", args.output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
