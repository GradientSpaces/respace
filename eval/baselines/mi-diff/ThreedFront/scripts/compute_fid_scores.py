#
# Modified from 
#   https://github.com/nv-tlabs/ATISS.
#
"""Script for computing the FID or KID between real and synthesized layout images.
"""
import argparse
import os
import torch
import numpy as np
import shutil
from cleanfid import fid
import pickle

from threed_front.evaluation import ThreedFrontResults
from utils import PROJ_DIR, create_or_clear_output_dir, update_render_paths


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--compute_kid",
        action="store_true",
        help="Compute KID instead of FID"
    )
    parser.add_argument(
        "--dataset_directory",
        default=None,
        help="Path to dataset directory"
        "(default: use train_dataset and test_dataset stored in result_file)"
    )
    parser.add_argument(
        "--synthesized_directory",
        default=None,
        help="Path to the folder containing the synthesized images"
        "(default: the directory containing result_file)"
    )
    parser.add_argument(
        "--output_directory",
        default=None,
        help="Output directory to store real and fake top-down projection images"
        "(default: output/tmp_fid)"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="Select ground-truth images without texture"
    )

    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        input("PyTorch cannot find CUDA. Press any key to continue with CPU...")

    # Score function
    score_func = fid.compute_kid if args.compute_kid else fid.compute_fid

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    test_dataset = threed_front_results.test_dataset
    real_render_name = "rendered_scene{}_256{}.png".format(
        "_notexture" if args.no_texture else "", 
        "_nofloor" if not threed_front_results.floor_condition else ""
    )
    update_render_paths(test_dataset, args.dataset_directory, real_render_name)
    
    # Default output directory
    if args.output_directory is None:
        args.output_directory = os.path.join(PROJ_DIR, "output/tmp_fid")

    # Set up output directories
    fid_real_dir = os.path.join(args.output_directory, "real")
    fid_fake_dir = os.path.join(args.output_directory, "fake")
    create_or_clear_output_dir(fid_real_dir)
    create_or_clear_output_dir(fid_fake_dir)

    # Copy ground-truth dataset images
    for i in range(len(test_dataset)):
        real_render_path = test_dataset._path_to_render(i)
        shutil.copy(real_render_path, "{}/{:05d}.png".format(fid_real_dir, i))
    num_test_scenes = len(test_dataset)
    print("Copied {} real images '{}' to {}".format(
        num_test_scenes, real_render_name, fid_real_dir)
    )

    # Copy synthesized images
    if args.synthesized_directory is None:
        args.synthesized_directory = os.path.dirname(args.result_file)
    synthesized_images = [os.path.join(args.synthesized_directory, fi)
        for fi in os.listdir(args.synthesized_directory) if fi.endswith(".png")]
    for i, fi in enumerate(synthesized_images):
        shutil.copyfile(fi, "{}/{:05d}.png".format(fid_fake_dir, i))

    # Compute
    score_with_all_images = \
        score_func(fid_real_dir, args.synthesized_directory, device=device)
    
    print("Compared images in '{}' and '{}'.".format(
        threed_front_results.config["data"]["dataset_directory"],
        args.synthesized_directory)
    )
    if args.compute_kid:
        print("KID: {}".format(score_with_all_images))
        print("(Note: KID varies slightly across different runs.)")
    else:
        print("FID: {}".format(score_with_all_images))

if __name__ == "__main__":
    main(None)
