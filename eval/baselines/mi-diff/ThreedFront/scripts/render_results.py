"""Script to generate synthesized orthographic projection images from result pickle file.
These images are needed to evaluate the model in terms of FID scores and real/fake classification accuracy.
"""
import argparse
import numpy as np
import os
import sys
import seaborn as sns
import pickle
from tqdm import tqdm

from threed_front.datasets import ThreedFutureDataset
from threed_front.rendering import scene_from_args
from threed_front.evaluation import ThreedFrontResults
from threed_front.simple_3dviz_setup import ORTHOGRAPHIC_PROJECTION_SCENE
from utils import PATH_TO_PICKLED_3D_FUTURE_MODEL, PATH_TO_FLOOR_PLAN_TEXTURES


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate synthetic layout images from predicted results"
    )        
    parser.add_argument(
        "result_file",
        help="Path to a pickled result (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--output_directory",
        help="Path to the output directory (default: result_file directory)"
    )
    parser.add_argument(
        "--path_to_pickled_3d_future_model",
        default=PATH_TO_PICKLED_3D_FUTURE_MODEL,
        help="Path to pickled 3d future model"
        "(default: output/threed_future_model_<room_type>.pkl)"
    )
    parser.add_argument(
        "--retrieve_by_size",
        action="store_true",
        help="Ignore objfeat and use size to retrieve most similar 3D-FUTURE models "
        "(default: use objfeat instead of sizes if available)"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="Color objects by semantic label, and set floor plan to white"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="Remove the floor plane (will be set to True if the model is not trained with floor plans)"
    )
    parser.add_argument(
        "--floor_color",
        type=lambda x: tuple(map(float, x.split(","))) if x!= None else None,
        help="Set floor color of generated images (and override path_to_floor_plan_textures)"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default=PATH_TO_FLOOR_PLAN_TEXTURES,
        help="Path to floor texture image directory or a single image file "
        "(default: demo/floor_plan_texture_images)"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=None,
        help="The size of the room along a side "
        "(default:3.1 for bedroom and library, 6.1 for diningroom and livingroom)"
    )

    args = parser.parse_args(argv)

    # Load results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    room_type = next((
        type for type in ["diningroom", "livingroom", "bedroom", "library"] \
        if type in os.path.basename(threed_front_results.config["data"]["dataset_directory"])
        ), None)
    assert room_type is not None
    print("Room type:", room_type)
    if not threed_front_results.config["network"].get("room_mask_condition", True):
        args.without_floor = True

    # Default output directory
    if args.output_directory is None:
        args.output_directory = os.path.dirname(args.result_file)
    print("Saving rendered results to: {}.".format(args.output_directory))

    # Output paths
    path_to_image = os.path.join(args.output_directory, "{:04d}_{}.png")

    # Check if output directory exists and if it doesn't create it
    if os.path.exists(args.output_directory) and \
        len([fi for fi in os.listdir(args.output_directory) if fi.endswith(".png")]) > 0:
        input("{} contain png files. Press any key to remove all png files..." \
              .format(args.output_directory))
        for fi in os.listdir(args.output_directory):
            if fi.endswith(".png"):
                os.remove(os.path.join(args.output_directory, fi))
    else:
        os.makedirs(args.output_directory, exist_ok=True)

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_future_model.format(room_type)
    )
    print("Loaded {} 3D-FUTURE models from: {}.".format(
        len(objects_dataset), args.path_to_pickled_3d_future_model.format(room_type)
    ))
    
    # Set floor texture or color
    if args.without_floor:
        args.floor_color = None
        floor_textures = [None]
    elif args.no_texture:
        # set floor to specified color if given, or white
        if args.floor_color is None:
            args.floor_color = (1, 1, 1)
        floor_textures = [None]
    else:
        # set floor to specified color if given, or sampled textures
        if args.floor_color is None:
            floor_textures = \
                [os.path.join(args.path_to_floor_plan_textures, fi)
                    for fi in os.listdir(args.path_to_floor_plan_textures)]
        else:
            floor_textures = [None]
    
    # Set color palette if args.no_texture
    if args.no_texture:
        color_palette = \
            sns.color_palette('hls', threed_front_results.test_dataset.n_object_types)
    else:
        color_palette = None
    
    # Create the scene
    scene_params = ORTHOGRAPHIC_PROJECTION_SCENE
    if args.without_floor:
        scene_params["background"] = (1, 1, 1, 1)
    if args.room_side is None:
        scene_params["room_side"] = 3.1 if room_type in ["bedroom", "library"] \
            else 6.1
    else:
        scene_params["room_side"] = args.room_side
    scene = scene_from_args(scene_params)
    print("Room side:", scene_params["room_side"])
    
    # Render projection images
    for i in tqdm(range(len(threed_front_results))):
        scene_idx = threed_front_results[i][0]
        image_path = path_to_image.format(
            i, threed_front_results.test_dataset[scene_idx].scene_id
        )
        threed_front_results.render_projection(
            i, objects_dataset, image_path, scene, 
            floor_texture=np.random.choice(floor_textures), 
            floor_color=args.floor_color, 
            retrieve_mode="size" if args.retrieve_by_size else None,
            color_palette=color_palette
        )


if __name__ == "__main__":
    main(sys.argv[1:])
