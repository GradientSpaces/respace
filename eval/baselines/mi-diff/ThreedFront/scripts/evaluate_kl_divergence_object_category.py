"""Script to compute the KL-divergence between the object categories of real 
and synthetic scenes."""
import argparse
import os
import sys
import pickle
import numpy as np

from threed_front.evaluation.utils import collect_cooccurrence
from threed_front.evaluation import ThreedFrontResults


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the KL-divergence between the object category "
                     "distributions of real and synthesized scenes")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--output_directory",
        default=None,
        help="Output directory to store kl-divergence and other stats (default: None)"
    )

    args = parser.parse_args(argv)

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    print("Received {} synthesized scenes".format(len(threed_front_results)))

    # Compute KL-divergence
    kl_divergence, gt_scenes, syn_scenes, gt_count, syn_count = \
        threed_front_results.evaluate_class_labels()

    # Print results
    classes = np.array(threed_front_results.test_dataset.object_types)
    for c, gt_cp, syn_cp in zip(classes, gt_count, syn_count):
        print("[{:>18}]: target: {:.4f} / synth: {:.4f}" \
              .format(c, gt_cp/gt_count.sum(), syn_cp/syn_count.sum()))
    print("object category kl divergence: {}".format(kl_divergence))

    if args.output_directory is not None:
        # Label co-ocurrences        
        gt_cooccurrences = collect_cooccurrence(gt_scenes, len(classes))
        syn_cooccurrences = collect_cooccurrence(syn_scenes, len(classes))

        path_to_stats = os.path.join(args.output_directory, "stats.npz")
        np.savez(path_to_stats,
                kl_divergence=kl_divergence, classes=classes,
                gt_classes=gt_count, syn_classes=syn_count,
                gt_cooccur=gt_cooccurrences, syn_cooccur=syn_cooccurrences)
        print("Saved stats to {}".format(path_to_stats))


if __name__ == "__main__":
    main(sys.argv[1:])
