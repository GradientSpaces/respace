"""Script to count the number of out of boundary objects and pairwise bounding 
boxes IoU in predicted layouts.
"""
import argparse
import numpy as np
import pickle

from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from threed_front.evaluation.utils import count_out_of_boundary, compute_bbox_iou


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result file (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--erosion",
        default=0.1,
        type=float,
        help="Amount of erosion in meters from predicted sizes (default: 0.1)"
    )
    parser.add_argument(
        "--area_tol",
        default=1e-5,
        type=float,
        help="Maximum out of boundary bbox area to be considered within floor bound (default: 1e-5)"
    )
    args = parser.parse_args(argv)
    assert args.erosion >= 0
    assert args.area_tol >= 0

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    assert threed_front_results.floor_condition

    # Load dataset
    config = threed_front_results.config
    raw_dataset = get_raw_dataset(
        config["data"], 
        split=config["validation"].get("splits", ["test"]),
        include_room_mask=True
    ) 
    
    # Count number of out-of-boundary objects
    oob_objects_total, num_objects_total = 0, 0             # synthesized
    oob_objects_ref_total, num_objects_ref_total = 0, 0     # real

    # Compute pairwise bounding boxes IoU
    bbox_iou_total = []             # synthesized
    inter_pairs_total = []          # synthesized, number of intersected (i.e. positive iou)
    bbox_iou_ref_total = []         # real
    inter_pairs_ref_total = []      # real, number of intersected (i.e. positive iou)

    for scene_idx, scene_layout in threed_front_results:
        gt_scene_layout = raw_dataset.get_room_params(scene_idx)
        
        # Out-of-boundary objects
        # synthesized layout
        oob_objects, oob_mask = count_out_of_boundary(
            gt_scene_layout["fpbpn"][:, :2],
            scene_layout, 
            erosion=args.erosion, 
            area_tol=args.area_tol
        )
        oob_objects_total += oob_objects
        num_objects_total += len(oob_mask)
        # check ground-truth layout
        oob_objects, oob_mask = count_out_of_boundary(
            gt_scene_layout["fpbpn"][:, :2],
            gt_scene_layout, 
            erosion=args.erosion, 
            area_tol=args.area_tol
        )
        oob_objects_ref_total += oob_objects
        num_objects_ref_total += len(oob_mask)

        # Bounding boxes IoU
        # synthesized layout
        bbox_iou = np.array(compute_bbox_iou(scene_layout))
        if len(bbox_iou) == 0:
            bbox_iou_total.append(0)
            inter_pairs_total.append(0)
        else:   
            bbox_iou_total.append(bbox_iou.mean()) 
            inter_pairs_total.append((bbox_iou>0).sum())
        # ground-truth layout
        bbox_iou = np.array(compute_bbox_iou(gt_scene_layout))
        bbox_iou_ref_total.append((bbox_iou).mean()) 
        inter_pairs_ref_total.append((bbox_iou>0).sum())

    print("(1) Found {} out-of-boundary objects from {} total ({:.4f} %) in {} synthesized scenes."\
          .format(
              oob_objects_total, num_objects_total, 
              oob_objects_total/num_objects_total * 100, len(threed_front_results)
            ))
    print("    For reference, there are {} out-of-boundary objects from {} total ({:.4f} %) in "
          "corresponding ground-turth scenes."\
          .format(
              oob_objects_ref_total, num_objects_ref_total,
              oob_objects_ref_total/num_objects_ref_total * 100
            ))
    print("(2) Average number of intersected bbox paris is {}, average IoU is {:.4f} % over {} synthesized scenes."\
          .format(
              np.mean(inter_pairs_total), np.mean(bbox_iou_total) * 100, 
              len(threed_front_results)
            ))
    print("    For reference, these are {} and {:.4f} % in ground-truth scenes."\
            .format(
                np.mean(inter_pairs_ref_total), np.mean(bbox_iou_ref_total) * 100
            ))


if __name__ == "__main__":
    main(None)
