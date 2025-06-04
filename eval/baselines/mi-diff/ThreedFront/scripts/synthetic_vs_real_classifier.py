#
# Modified from 
#   https://github.com/nv-tlabs/ATISS.
#

"""Script used to evaluate the scene classification accuracy between real and
synthesized layout images.
"""
import argparse
import os
import shutil
import torch
import numpy as np
import pickle

from threed_front.evaluation import ThreedFrontResults
from threed_front.evaluation.utils import ImageFolderDataset, SyntheticVRealDataset, AlexNet, compute_loss_acc
from utils import PROJ_DIR, create_or_clear_output_dir, update_render_paths


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Train a classifier to discriminate between real "
                     "and synthetic rooms")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--synthesized_directory",
        default=None,
        help="Path to the folder containing the synthesized images"
        "(default: the directory containing result_file)"
    )
    parser.add_argument(
        "--dataset_directory",
        default=None,
        help="Path to dataset directory"
        "(default: use train_dataset and test_dataset stored in result_file)"
    )
    parser.add_argument(
        "--output_directory",
        default=None,
        help="Output directory to store sampled synthesized images"
        "(default: output/tmp_synthesized)"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="Select ground-truth images without texture"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Set the batch size for training and evaluating (default: 256)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Set the PyTorch data loader workers (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Train for that many epochs (default: 10)"
    )
    parser.add_argument(
        "--n_runs",
        default=10,
        type=int,
        help="Number of runs to average over"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling synthesized projection images"
    )

    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        input("PyTorch cannot find CUDA. Press any key to continue with CPU...")

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    real_render_name = "rendered_scene{}_256{}.png".format(
        "_notexture" if args.no_texture else "", 
        "_nofloor" if not threed_front_results.floor_condition else ""
    )
    print("Real images to compare to: {}".format(real_render_name))
    
    # Default output directory
    if args.output_directory is None:
        args.output_directory = os.path.join(PROJ_DIR, "output/tmp_synthesized")
    output_dir = args.output_directory
    create_or_clear_output_dir(output_dir)

    # Access real images through processed dataset
    train_dataset_real = threed_front_results.train_dataset
    test_dataset_real = threed_front_results.test_dataset
    real_render_name = "rendered_scene{}_256{}.png".format(
        "_notexture" if args.no_texture else "", 
        "_nofloor" if not threed_front_results.floor_condition else ""
    )
    update_render_paths(train_dataset_real, args.dataset_directory, real_render_name)
    update_render_paths(test_dataset_real, args.dataset_directory, real_render_name)

    # Find all synthesized images
    if args.synthesized_directory is None:
        args.synthesized_directory = os.path.dirname(args.result_file)
    synthesized_images = [os.path.join(args.synthesized_directory, fi)
        for fi in os.listdir(args.synthesized_directory) if fi.endswith(".png")]

    scores = []
    for run in range(args.n_runs):

        print("\nRun: {}/{}".format(run + 1, args.n_runs))
        # Sample synthesized images
        np.random.shuffle(synthesized_images)
        synthesized_images_subset = np.random.choice(
            synthesized_images, len(test_dataset_real) * 2
        )

        for i, fi in enumerate(synthesized_images_subset):
            shutil.copyfile(fi, "{}/{:05d}.png".format(output_dir, i))
        
        # Create the synthetic datasets by splitting the synthetic images by half
        train_synthetic = ImageFolderDataset(output_dir, True)
        test_synthetic = ImageFolderDataset(output_dir, False)

        # Join them in useable datasets
        train_dataset = SyntheticVRealDataset(
            train_dataset_real, train_synthetic, real_render_name
        )
        test_dataset = SyntheticVRealDataset(
            test_dataset_real, test_synthetic, real_render_name
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        # Create the model
        model = AlexNet()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train the model
        for epoch in range(args.epochs):
            loss_meter, acc_meter = compute_loss_acc(
                model, train_dataloader, True, optimizer
            )

            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    loss_meter, acc_meter = compute_loss_acc(
                        model, test_dataloader, False
                    )
        
        score = acc_meter.value * 100
        if score < 50: score = 100 - score  # avoid <50% runs pulling the mean towards 50%
        scores.append(score)
        print("Accuracy: {:.4f} %".format(acc_meter.value * 100))
    
    print("Compared images in '{}' and '{}'.".format(
        threed_front_results.config["data"]["dataset_directory"],
        args.synthesized_directory)
    )
    print("Average accuracy: {:.4f} +/- {:.4f} %".format(
        np.mean(scores), np.std(scores)
    ))
    

if __name__ == "__main__":
    main(None)
