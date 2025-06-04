# 
# Modified from: 
#   https://github.com/QiuhongAnnaWei/LEGO-Net/blob/main/data/preprocess_TDFront.py
# 

import numpy as np
import cv2 as cv
from scipy.interpolate import interp1d


def process_floorplan_iterative_closest_point(scene_data, room_side):
    """ Returns ordered floorplan corners w.r.t. "floor_plan_centroid": numpy array of shape [numpt, 2]
        scene_data: pre-processed scene data (loaded from boxes.npz)
        room_side: room_side parameter used to render room_layout in scene_data (can be an approximate)
    """    
    ## Source: contour points found on room_layout mask
    room_layout = np.squeeze(scene_data["room_layout"])
    all_contours, _ = cv.findContours(room_layout, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # all_contours: tuple of contours in image. Each contour = numpy array of (x,y) coordinates of boundary points [numpt, 1, 2]
    contour = all_contours[0] if len(all_contours)==1 else max(all_contours, key = cv.contourArea) # a few edge cases have > 1 enclosed region
    contour = np.squeeze(contour) # (numcontourpt,1,2) -> (numcontourpt,2)
    contour = contour / (room_layout.shape[0]) * (room_side*2) - room_side #[-room_side, room_side]

    ## Target: ATISS's generated points, extracted from 3DFRONT mesh objects in json
    center_2d = np.array([scene_data["floor_plan_centroid"][0], scene_data["floor_plan_centroid"][2]])
    corners = np.unique(scene_data["floor_plan_vertices"],axis=0) # num_pt, 2 -> num_uniquept, 2
    corners = corners[:,[0,2]] - center_2d

    ## Iterative closest point
    max_iter = 3
    dist_to_discard = 0.15
    for _ in range(max_iter):
        scale_sum, new_contour, ordered_corners = 0, [], []
        for conpt in contour:
            distance = np.array([np.linalg.norm(conpt - c, ord=2) for c in corners]) # 1d array
            min_index = np.argmin(distance)
            if distance[min_index] > dist_to_discard: continue # no matching mesh corner points, discard
            
            new_contour.append(conpt) # keep it in next iteration
            ordered_corners.append(corners[min_index]) # for if we break
            scale_sum += np.linalg.norm(corners[min_index]) / np.linalg.norm(conpt) # we take its average

        new_contour = np.array(new_contour)
        transform_scale = scale_sum/new_contour.shape[0]
        if abs(transform_scale-1) < 0.01: break
        contour = new_contour*transform_scale # transform

    ordered_unique_idx = sorted(np.unique(ordered_corners, axis=0, return_index=True)[1]) # mapped_corner retains order of contour
    ordered_corners = np.array([ordered_corners[i] for i in ordered_unique_idx])

    if ordered_corners.shape[0] > corners.shape[0]:
        print("Received {} floor plan vertices but found {} ordered corners."\
              .format(corners.shape[0], ordered_corners.shape[0]))
    
    return ordered_corners


def fp_line_normal(fpoc):
    """ fpoc: [numpt, 2] np array, scene_data's floor_plan_ordered_corners.
        Return normalized floor plan line normals.
    """
    fp_line_n = np.zeros((fpoc.shape[0], 2)) # for each line, get its normal. fpbp_normal[0] for fpoc[0] to fpoc[1]
    for i in range(fpoc.shape[0]):
        line_vec = fpoc[(i+1)%fpoc.shape[0]] - fpoc[i] # clockwise starting from bottom left
        line_len = np.linalg.norm(line_vec)   # possible for a line to have almost 0 len
        if line_len == 0: 
            print("! fp_line_normal: line_len==0!")
            continue
        fp_line_n[i, 0] = line_vec[1]/line_len   # dy for x axis
        fp_line_n[i, 1] = -line_vec[0]/line_len  # -dx for y axis (points inwards towards room center)
    return fp_line_n # normalized


def scene_sample_fpbp(fpoc, num_sampled_points=256):
    """ fpoc: [numpt, 2] np array, scene_data's floor_plan_ordered_corners.
        Return sample floor plan boundary pt + normal.
    """
    nfpbp = num_sampled_points

    x = np.append(fpoc[:,0], [fpoc[0,0]]) # (nline+1,), in [-3,3]
    y = np.append(fpoc[:,1], [fpoc[0,1]]) # append one extra to close the loop, 
    fp_line_n = fp_line_normal(fpoc) # numpt, 2

    # sample nfpbp points randomly from the contour outline:
    # Linear length on the line
    dist_bins = np.cumsum( np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ) ) # (nfpoc+1,) cumulative line seg len from bottom left pt
    dist_bins = dist_bins/dist_bins[-1] # (nfpoc+1,), [0, ..., 1] (values normalized to 0 to 1)

    fx, fy = interp1d(dist_bins, x), interp1d(dist_bins, y) # [0, 1] -> [-3/6, 3/6]

    seg_len = float(1)/nfpbp # total perimeter normalized to 1 (distance above)
    seg_starts = np.linspace(0, 1, nfpbp+1)[:-1] # (nfpbp,), starting point of each segment # [0.   0.25 0.5  0.75 1.  ][:-1]
    per_seg_displacement = np.random.uniform(low=0.0, high=seg_len, size=(nfpbp)) # one for each line segment
    sampled_distance = seg_starts + per_seg_displacement # (nfpbp=250, 1)
    sampled_x, sampled_y = fx(sampled_distance), fy(sampled_distance) # (nfpbp=250,), in [-3,3] (convert from 1d sampling to xy coord)

    fpbp = np.concatenate([np.expand_dims(sampled_x, axis=1), np.expand_dims(sampled_y, axis=1)], axis=1) #(nfpbp, 1+1=2)

    try:
        bin_idx = np.digitize(sampled_distance, dist_bins) # bins[inds[n]-1] <= x[n] < bins[inds[n]]
        bin_idx -= 1 # (nfpbp=250,) in range [0, nline-1] # example: [ 0  7 10 12 14 17 18 21 22 24] 
        fpbp_normal = fp_line_n[bin_idx, :] # fp_line_n: [nline, 2] -> fpbp_normal: [nfpbp, 2]
    except Exception as e:
        # add this fix for certain samples since the algorithm can fail
        print("! scene_sample_fpbp: Exception: {}!".format(e))
        return np.zeros((nfpbp, 4), dtype=np.float32) # [nfpbp, 2+2=4]

    return np.concatenate([fpbp, fpbp_normal], axis=1) # (nfpbp, 2+2=4)
