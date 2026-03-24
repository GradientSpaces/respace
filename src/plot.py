from matplotlib import pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
from pathlib import Path
import copy
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import glob
from collections import defaultdict
import random

from src.sample import AssetRetrievalModule
from src.utils import create_floor_plan_polygon, remove_and_recreate_folder, get_pths_dataset_split, get_test_instrs_all
from src.dataset import create_full_scene_from_before_and_added
from src.viz import render_full_scene_and_export_with_gif, create_360_video_instr, create_360_video_full, create_360_video_voxelization, create_360_videos_assets
from src.eval import eval_scene

def plot_ablation_fid_kid_pbl_pms(title, x_name, x_values, fid_scores, kid_scores, delta_pbl, pms_score):
		
	# Create a figure with 1x3 subplots
	fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

	fig.suptitle(f'Ablation Study: {title}', fontsize=16)
	
	# Plot 1: FID and KID Scores
	ax1 = axes[0]
	ax1_twin = ax1.twinx()
	ax1.plot(x_values, fid_scores, 'g-', marker='o')
	ax1_twin.plot(x_values, kid_scores, 'b-', marker='o')
	ax1.set_xlabel(x_name)
	ax1.set_ylabel('FID Score', color='g')
	ax1_twin.set_ylabel('KID Score', color='b')
	ax1.set_title(f'FID and KID Scores vs {x_name}')
	ax1.set_xticks(x_values)
	ax1.set_xticklabels(x_values, rotation=45)
	ax1.tick_params(axis='y', labelcolor='g')
	ax1_twin.tick_params(axis='y', labelcolor='b')
	#ax1.set_ylim(39.5, 40.5)
	#ax1_twin.set_ylim(4.5, 5.0)
	ax1.grid(alpha=0.3)
	
	# Plot 2: Delta PBL
	ax2 = axes[1]
	ax2.plot(x_values, delta_pbl, 'g-', marker='o')
	ax2.set_xlabel(x_name)
	ax2.set_ylabel('Delta PBL')
	ax2.set_title(f'Delta PBL vs {x_name}')
	ax2.set_xticks(x_values)
	ax2.set_xticklabels(x_values, rotation=45)
	#ax2.set_ylim(0, 0.03)
	ax2.grid(alpha=0.3)
	
	# Plot 3: PMS Score
	ax3 = axes[2]
	ax3.plot(x_values, pms_score, 'g-', marker='o')
	ax3.set_xlabel(x_name)
	ax3.set_ylabel('PMS Score')
	ax3.set_title(f'PMS Score vs {x_name}')
	ax3.set_xticks(x_values)
	ax3.set_xticklabels(x_values, rotation=45)
	#ax3.set_ylim(0.7, 0.8)
	ax3.grid(alpha=0.3)
	
	# Save the combined figure
	plt.savefig(f'plots/{title}.svg')

def get_stats_per_n_object_from_file(filename, n_aggregate_per=2):
	metrics = json.load(open(f"./eval/metrics-raw/{filename}", "r"))
	stats = {}
	floor_areas = {}  # Dictionary to track floor areas per bin
	
	for seed in tqdm(range(3)):
		metrics_seed = metrics[seed]
		for sample in metrics_seed:
			delta_pbl = sample.get("delta_pbl_loss") * 1000
			n_objects = sample.get("scene").get("objects")
			if isinstance(n_objects, list):
				n_objects = len(n_objects)
			
			# Aggregate by grouping objects
			aggregated_bin = (n_objects - 1) // n_aggregate_per * n_aggregate_per + 1

			# Get floor area
			floor_area = create_floor_plan_polygon(sample.get("scene").get("bounds_bottom")).area
			
			# Store delta_pbl values
			if aggregated_bin not in stats:
				stats[aggregated_bin] = {}
				floor_areas[aggregated_bin] = []  # Initialize floor area list for this bin
			if seed not in stats[aggregated_bin]:
				stats[aggregated_bin][seed] = [delta_pbl]
			else:
				stats[aggregated_bin][seed].append(delta_pbl)
				
			# Store floor area for this sample
			floor_areas[aggregated_bin].append(floor_area)

	n_objects_sorted = sorted(stats.keys())
	delta_pbl_mean = []
	delta_pbl_std = []
	mean_floor_areas = []  # List to store mean floor area for each bin
	std_floor_areas = []   # List to store std deviation of floor area for each bin

	for n_obj in n_objects_sorted:
		# Calculate delta_pbl statistics
		seed_means = [np.mean(stats[n_obj][seed]) for seed in range(3) if seed in stats[n_obj]]
		delta_pbl_mean.append(np.mean(seed_means))
		delta_pbl_std.append(np.std(seed_means))
		
		# Calculate mean and std deviation of floor area for this bin
		mean_floor_areas.append(np.mean(floor_areas[n_obj]))
		std_floor_areas.append(np.std(floor_areas[n_obj]))
	
	return n_objects_sorted, delta_pbl_mean, delta_pbl_std, mean_floor_areas, std_floor_areas

def plot_stats_per_n_objects_instr(room_type, postfix, n_aggregate_per=2):

	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.size'] = 12
	plt.rcParams['text.usetex'] = False
	plt.rcParams['axes.unicode_minus'] = True

	times_new_roman_size = 36
	label_font_size = 28
	tick_font_size = 28

	blue_colors = ['#78a5cc', "#3d79b4", "#29619A", "#13417D"]

	fig, ax1 = plt.subplots(figsize=(10, 8))

	n_objects_sorted1, delta_pbl_mean1, delta_pbl_std1, _, _ = get_stats_per_n_object_from_file(
		f"eval_samples_baseline-atiss_instr_{room_type}_raw.json",
		n_aggregate_per=n_aggregate_per
	)
	n_objects_sorted2, delta_pbl_mean2, delta_pbl_std2, _, _ = get_stats_per_n_object_from_file(
		f"eval_samples_baseline-midiff_instr_{room_type}_raw.json",
		n_aggregate_per=n_aggregate_per
	)
	n_objects_sorted4, delta_pbl_mean4, delta_pbl_std4, _, _ = get_stats_per_n_object_from_file(
		f"eval_samples_respace_instr_{postfix}_raw.json",
		n_aggregate_per=n_aggregate_per
	)
	_, _, _, floor_areas, floor_std = get_stats_per_n_object_from_file(
		f"eval_samples_baseline-atiss_instr_{room_type}_raw.json",
		n_aggregate_per=n_aggregate_per
	)

	ax1.plot(n_objects_sorted1, delta_pbl_mean1, 'o-',
			 markersize=6, linewidth=2,
			 color=blue_colors[0], label="ATISS")
	ax1.fill_between(n_objects_sorted1,
					 [m - s for m, s in zip(delta_pbl_mean1, delta_pbl_std1)],
					 [m + s for m, s in zip(delta_pbl_mean1, delta_pbl_std1)],
					 color=blue_colors[0], alpha=0.1)

	ax1.plot(n_objects_sorted2, delta_pbl_mean2, 'o-',
			 markersize=6, linewidth=2,
			 color=blue_colors[1], label="Mi-Diff")
	ax1.fill_between(n_objects_sorted2,
					 [m - s for m, s in zip(delta_pbl_mean2, delta_pbl_std2)],
					 [m + s for m, s in zip(delta_pbl_mean2, delta_pbl_std2)],
					 color=blue_colors[1], alpha=0.1)

	ax1.plot(n_objects_sorted4, delta_pbl_mean4, 'o-',
			 markersize=6, linewidth=2,
			 color=blue_colors[3], label="$\\text{ReSpace/A}^{\\dagger}$")
	ax1.fill_between(n_objects_sorted4,
					 [m - s for m, s in zip(delta_pbl_mean4, delta_pbl_std4)],
					 [m + s for m, s in zip(delta_pbl_mean4, delta_pbl_std4)],
					 color=blue_colors[3], alpha=0.1)

	ax2 = ax1.twinx()

	n_objects_array = np.array(n_objects_sorted1)
	floor_array = np.array(floor_areas)
	x_poly = np.concatenate([n_objects_array, np.flip(n_objects_array)])
	y_poly = np.concatenate([floor_array, np.zeros_like(floor_array)])

	ax2.plot(n_objects_sorted1, floor_areas, linewidth=1.5, color='#9a9a9a', linestyle='--')
	ax2.fill(x_poly, y_poly, alpha=0.1, color='#7a7a7a', label="Floor Area")

	max_n_objects = max(
		max(n_objects_sorted1 or [0]),
		max(n_objects_sorted2 or [0]),
		max(n_objects_sorted4 or [0])
	)
	x_ticks = list(range(1, max_n_objects + 1, n_aggregate_per))
	x_tick_labels = [f"{x_ticks[i]}-{x_ticks[i] + n_aggregate_per - 1}" for i in range(len(x_ticks))]

	ax1.set_title(f"Delta VBL — '{room_type.split('-')[0]}' dataset", fontsize=times_new_roman_size)
	ax1.set_xlabel("# of objects", fontsize=label_font_size)
	ax1.set_ylabel("Δ VBL", fontsize=label_font_size)
	ax2.set_ylabel("Mean Floor Area (m²)", fontsize=label_font_size)

	ax1.tick_params(axis='both', labelsize=tick_font_size)
	ax2.tick_params(axis='both', labelsize=tick_font_size)

	ax1.set_xticks(x_ticks)
	ax1.set_xticklabels(x_tick_labels, fontsize=tick_font_size)
	ax1.set_xlim(left=x_ticks[0], right=x_ticks[-1])
	ax2.set_xlim(left=x_ticks[0], right=x_ticks[-1])

	ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
	ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

	ax1_ylim_min = min(min(delta_pbl_mean1), min(delta_pbl_mean2), min(delta_pbl_mean4))
	ax1.set_ylim(bottom=ax1_ylim_min)
	ax2.set_ylim(bottom=min(floor_areas))

	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=26)
	legend.get_frame().set_alpha(0.99)

	ax1.grid(True, linestyle='--', alpha=0.3)

	plt.tight_layout()
	plt.subplots_adjust(left=0.14, right=0.91, top=0.93, bottom=0.12)
	plt.savefig(f"./plots/delta_pbl_vs_n_objects_{postfix}_with_area.jpg")
	plt.savefig(f"./plots/delta_pbl_vs_n_objects_{postfix}_with_area.pdf")

def render_comparison(mode, row_type, pth_root, pth_folder_fig_prefix, seed_and_idx, camera_height=None, is_supp=False, asset_sampling=False, num_asset_samples=0, sampling_engine=None):
	bg_color = np.array([240, 240, 240]) / 255.0
	room_type = row_type.split("_")[0]
	pth_folder_fig = Path(f"{pth_folder_fig_prefix}-{row_type}")
	remove_and_recreate_folder(pth_folder_fig)
	
	# Load and render the 'before' scene
	scene = json.load(open(f"{pth_root}/baseline-atiss/{mode}/{room_type}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
	scene_before = copy.deepcopy(scene)
	if mode == "instr":
		scene_before["objects"] = scene_before["objects"][:-1]
	else:
		if not is_supp:
			scene_before["objects"] = []
		else:
			# if supp and full scenes, take GT as scene before
			all_pths = get_pths_dataset_split(room_type, "test")
			all_test_instrs = get_test_instrs_all(room_type)
			pth_scene = all_pths[seed_and_idx[1]]
			scene_before = json.load(open(os.getenv("PTH_STAGE_2_DEDUP") + f"/{pth_scene}", "r"))

	# Render the 'before' scene
	render_full_scene_and_export_with_gif(scene_before, filename="0-0", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
	
	# If doing asset sampling, handle it differently
	if asset_sampling:
		# Load our reference scene
		reference_scene = json.load(open(f"{pth_root}/respace/{mode}/{room_type}{'-with-qwen1.5b-all-grpo-bon-1'}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
		
		# Get the last object from reference scene
		target_obj = reference_scene["objects"][-1]
		
		# Remove any existing sampling results
		for key in list(target_obj.keys()):
			if key.startswith("sampled_"):
				del target_obj[key]
		
		# Metrics storage
		metrics_list = []
		
		# Generate and render multiple samples
		for i in range(num_asset_samples):
			# Create a copy of the scene
			scene_with_asset = copy.deepcopy(scene_before)
			
			# Add target object without sampled assets
			scene_with_asset["objects"].append(copy.deepcopy(target_obj))
			
			# Sample a new asset
			scene_with_asset = sampling_engine.sample_last_asset(scene_with_asset, is_greedy_sampling=False)

			# import pdb
			# pdb.set_trace()
			
			# Evaluate the scene
			metrics = eval_scene(scene_with_asset)
			metrics_list.append(metrics)
			
			# Render the scene
			render_full_scene_and_export_with_gif(scene_with_asset, filename=f"0-{i+1}", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		
		# For instruction mode, get and render GT
		if mode == "instr":
			all_pths = get_pths_dataset_split(room_type, "test")
			all_test_instrs = get_test_instrs_all(room_type)
			pth_scene = all_pths[seed_and_idx[1]]
			instr_sample = all_test_instrs.get(pth_scene)[seed_and_idx[0]]
			scene_query = json.loads(instr_sample["sg_input"])
			obj_to_add = json.loads(instr_sample["sg_output_add"])
			gt_scene = create_full_scene_from_before_and_added(scene_query, obj_to_add)
			render_full_scene_and_export_with_gif(gt_scene, filename=f"0-{num_asset_samples+1}", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		
		return metrics_list
	
	# Regular comparison logic (original implementation)
	else:
		# render ATISS
		render_full_scene_and_export_with_gif(scene, filename="0-1", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		
		# render mi-diff
		scene = json.load(open(f"{pth_root}/baseline-midiff/{mode}/{room_type}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
		render_full_scene_and_export_with_gif(scene, filename="0-2", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		
		# render ours
		# scene = json.load(open(f"{pth_root}/respace/{mode}/{room_type}{'-with-qwen1.5b-all-grpo-bon-1'}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
		scene = json.load(open(f"{pth_root}/respace/{mode}/{room_type}{'-rej-n512fixed-1e5-bon-1'}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
		render_full_scene_and_export_with_gif(scene, filename="0-3", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		
		# render GT or special version
		if mode == "instr":
			all_pths = get_pths_dataset_split(room_type, "test")
			all_test_instrs = get_test_instrs_all(room_type)
			pth_scene = all_pths[seed_and_idx[1]]
			instr_sample = all_test_instrs.get(pth_scene)[seed_and_idx[0]]
			scene_query = json.loads(instr_sample["sg_input"])
			obj_to_add = json.loads(instr_sample["sg_output_add"])
			scene = create_full_scene_from_before_and_added(scene_query, obj_to_add)
			render_full_scene_and_export_with_gif(scene, filename="0-4", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		else:
			if not is_supp:
				all_pths = get_pths_dataset_split(room_type, "test")
				all_test_instrs = get_test_instrs_all(room_type)
				pth_scene = all_pths[seed_and_idx[1]]
				scene = json.load(open(os.getenv("PTH_STAGE_2_DEDUP") + f"/{pth_scene}", "r"))
				render_full_scene_and_export_with_gif(scene, filename="0-4", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
			else:
				# take BON8 sample as last column
				# scene = json.load(open(f"{pth_root}/respace/{mode}/{room_type}{'-with-qwen1.5b-all-grpo-bon-8'}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
				scene = json.load(open(f"{pth_root}/respace/{mode}/{room_type}{'-rej-n512fixed-1e5-bon-1-shuffling-8-bonrot-fast-v3'}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
				render_full_scene_and_export_with_gif(scene, filename="0-4", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		
		return None

def plot_qualitative_figure_comparison(mode, num_rows=4, sample_data=None, camera_heights=None, is_supp=False, asset_sampling=False, num_asset_samples=0, skip_scene_before=False):
	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.rcParams['mathtext.fontset'] = 'stix' 
	plt.rcParams['font.size'] = 12
	plt.rcParams['text.usetex'] = False
	plt.rcParams['axes.unicode_minus'] = True

	title_font_size = 32
	label_font_size = 28
	tick_font_size = 28

	seed_idx_lookup = {
		1234: 0,
		3456: 1,
		5678: 2,
	}

	# Path for figures
	pth_root = "./eval/samples"
	pth_folder_fig_prefix = f"./eval/viz/fig-ours-vs-baselines"
	if asset_sampling:
		pth_folder_fig_prefix += f"-assets-{mode}"
	else:
		pth_folder_fig_prefix += f"-{mode}"
	
	# Dictionary to store metrics for each row
	all_metrics = {}
	
	# Render images for each row
	sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=1234, do_print=False)
	for row_type, sample in sample_data.items():
		metrics = render_comparison(mode, row_type, pth_root, pth_folder_fig_prefix, sample, camera_height=camera_heights[row_type], is_supp=is_supp, asset_sampling=asset_sampling, num_asset_samples=num_asset_samples, sampling_engine=sampling_engine)
		all_metrics[row_type] = metrics
	
	# Determine plot size based on mode and settings
	if asset_sampling:
		if mode == "instr":
			plot_figsize = (5*(num_asset_samples+2), 3.2*num_rows)
			num_cols = num_asset_samples + 2  # before, samples, GT
		else:
			plot_figsize = (5*(num_asset_samples+1), (3.6*num_rows))
			num_cols = num_asset_samples + 1  # before, samples
	elif skip_scene_before:
		num_cols = 4  # ATISS, Mi-Diff, Ours, GT (no "Scene Before")
		plot_figsize = (4.5*num_cols, 3.4*num_rows)
	else:
		if mode == "instr":
			plot_figsize = (5*5, 2.5*5)
			num_cols = 5  # Standard 5 columns
		else:
			plot_figsize = (5*5, (3.6*num_rows))
			num_cols = 5  # Standard 5 columns
	
	# Create the figure and axes
	fig, axs = plt.subplots(num_rows, num_cols, figsize=plot_figsize)
	
	# Load and plot images for each row
	row_idx = 0
	for row_type, sample in sample_data.items():
		row_prefix = f"{pth_folder_fig_prefix}-{row_type}"
		
		col_offset = 1 if skip_scene_before else 0  # skip 0-0.jpg (empty floor plan)
		for col_idx in range(num_cols):
			img_col = col_idx + col_offset  # map to rendered image index
			img_path = Path(f"{row_prefix}") / "diag" / f"0-{img_col}.jpg"
			if os.path.exists(img_path):
				img = Image.open(img_path)
				width, height = img.size

				# Apply appropriate cropping based on mode and row
				if mode == "instr":
					if (row_idx == 1):
						crop_top = int(height * 0.1)
						crop_bottom = int(height * 0.7)
					else:
						crop_top = int(height * 0.2)
						crop_bottom = int(height * 0.8)
				else:
					if (row_idx == 0):
						crop_top = int(height * 0.1)
						crop_bottom = int(height * 0.8)
					else:
						crop_top = int(height * 0.15)
						crop_bottom = int(height * 0.85)

				# Crop the image (left, top, right, bottom)
				cropped_img = img.crop((0, crop_top, width, crop_bottom))

				# Display the cropped image
				axs[row_idx, col_idx].imshow(cropped_img)

			axs[row_idx, col_idx].axis('off')
		
		# Load metrics for this row if not asset sampling
		if not asset_sampling:
			row_metrics = load_metrics_for_row(row_type, mode, sample, seed_idx_lookup)
		else:
			row_metrics = all_metrics[row_type]  # Already computed on-the-fly
		
		# Add metric text to plot
		add_metrics_to_plot(axs, row_idx, row_metrics, mode=mode, is_supp=is_supp, asset_sampling=asset_sampling, num_asset_samples=num_asset_samples, skip_scene_before=skip_scene_before)

		row_idx += 1

	# Set titles for columns
	set_column_titles(mode, axs, is_supp=is_supp, asset_sampling=asset_sampling, num_asset_samples=num_asset_samples, skip_scene_before=skip_scene_before)
	
	# Adjust layout
	if skip_scene_before:
		fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0, hspace=0.05, wspace=0.02)
	elif mode == "instr":
		# instr
		# fig.subplots_adjust(left=0.015, right=1.0, top=0.95, bottom=0.0, hspace=0.05, wspace=0.0)
		fig.subplots_adjust(left=0.015, right=1.0, top=0.93, bottom=0.0, hspace=0.05, wspace=0.0)
	else:
		# full
		if is_supp:
			fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.0, hspace=0.1, wspace=0.0)
		else:
			fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0, hspace=0.1, wspace=0.0)

	
	# Determine filename based on settings
	if asset_sampling:
		filename = f"ours_vs_baselines_assets_{mode}"
	else:
		filename = f"ours_vs_baselines_{mode}"
		if is_supp:
			filename += "_supp"
		else:
			filename += "_2"
	
	# Save figure
	plt.savefig(f"./plots/{filename}.pdf", dpi=100)
	plt.savefig(f"./plots/{filename}.jpg", dpi=300)
	


def load_metrics_for_row(row_type, mode, sample, seed_idx_lookup, asset_sampling=False, asset_metrics=None):
	if asset_sampling:
		if asset_metrics is None:
			return None
		
		# Format the asset metrics into a list of dicts
		metrics_list = []
		for metric in asset_metrics:
			metrics_list.append(metric)
		
		return metrics_list
	
	# Original implementation for method comparison
	seed, idx = sample
	seed_idx = seed_idx_lookup[seed]
	
	base_path = f"./eval/metrics-raw/"
	
	# If all_fail, we need to use the "all" dataset
	dataset_type = row_type.split("_")[0] if "_" in row_type else row_type
	
	# Load metrics for ATISS
	atiss_file = f"{base_path}eval_samples_baseline-atiss_{mode}_{dataset_type}_raw.json"
	metrics_atiss = json.load(open(atiss_file, "r"))[seed_idx][idx]
	
	# Load metrics for Mi-Diff
	midiff_file = f"{base_path}eval_samples_baseline-midiff_{mode}_{dataset_type}_raw.json"
	metrics_midiff = json.load(open(midiff_file, "r"))[seed_idx][idx]
	
	# Load metrics for Ours

	# ours_file = f"{base_path}eval_samples_respace_{mode}_{dataset_type}-with-qwen1.5b-all-grpo-bon-1_qwen1.5b-all-grpo-bon-1_raw.json"
	ours_file = f"{base_path}eval_samples_respace_{mode}_{dataset_type}-rej-n512fixed-1e5-bon-1-shuffling-1_raw.json"
	
	# ours_file_bon8 = f"{base_path}eval_samples_respace_{mode}_{dataset_type}-with-qwen1.5b-all-grpo-bon-8_qwen1.5b-all-grpo-bon-8_raw.json"
	ours_file_bon8 = f"{base_path}eval_samples_respace_{mode}_{dataset_type}-rej-n512fixed-1e5-bon-1-shuffling-8-bonrot-fast-v3_raw.json"
	
	metrics_ours = json.load(open(ours_file, "r"))[seed_idx][idx]
	metrics_ours_bon8 = json.load(open(ours_file_bon8, "r"))[seed_idx][idx] if os.path.exists(ours_file_bon8) else None
	
	return {
		"atiss": metrics_atiss,
		"midiff": metrics_midiff,
		"ours": metrics_ours,
		"ours-bon8": metrics_ours_bon8 if mode == "full" else None
	}


def add_metrics_to_plot(axs, row_idx, metrics, mode="instr", is_supp=False, asset_sampling=False, num_asset_samples=0, skip_scene_before=False):
	metric_font_size = 16

	# Handle asset sampling differently
	if asset_sampling:
		textbox_contents = [""]  # First column has no metrics

		for i in range(num_asset_samples):
			if mode == "instr":
				# For "instr" mode, use delta metrics with delta symbol
				textbox_contents.append(
					f"Δ OOB: {round(metrics[i].get('delta_oob_loss', 0), 2)} / Δ MBL: {round(metrics[i].get('delta_mbl_loss', 0), 2)}"
				)
			else:
				# For "full" mode, use total metrics without delta symbol
				textbox_contents.append(
					f"OOB: {round(metrics[i].get('total_oob_loss', 0), 2)} / MBL: {round(metrics[i].get('total_mbl_loss', 0), 2)}"
				)

		# Add empty string for the GT column if in instr mode
		if mode == "instr":
			textbox_contents.append("")

	# Original implementation for method comparison
	else:
		if mode == "instr":
			# For "instr" mode, use delta metrics with delta symbol
			textbox_contents = [
				"",  # First column has no metrics
				f"Δ OOB: {round(metrics['atiss']['delta_oob_loss'], 2)} / Δ MBL: {round(metrics['atiss'].get('delta_mbl_loss'), 2)}",
				f"Δ OOB: {round(metrics['midiff']['delta_oob_loss'], 2)} / Δ MBL: {round(metrics['midiff'].get('delta_mbl_loss'), 2)}",
				f"Δ OOB: {round(metrics['ours']['delta_oob_loss'], 2)} / Δ MBL: {round(metrics['ours'].get('delta_mbl_loss'), 2)}",
				""  # Last column has no metrics
			]
		elif skip_scene_before:
			# 4-column layout: ATISS, Mi-Diff, Ours, GT (no "Scene Before")
			textbox_contents = [
				f"OOB: {round(metrics['atiss']['total_oob_loss'], 2)} / MBL: {round(metrics['atiss'].get('total_mbl_loss'), 2)}",
				f"OOB: {round(metrics['midiff']['total_oob_loss'], 2)} / MBL: {round(metrics['midiff'].get('total_mbl_loss'), 2)}",
				f"OOB: {round(metrics['ours']['total_oob_loss'], 2)} / MBL: {round(metrics['ours'].get('total_mbl_loss'), 2)}",
				"",  # GT column has no metrics
			]
		else:
			# For "full" mode, use total metrics without delta symbol
			textbox_contents = [
				"",  # First column has no metrics
				f"OOB: {round(metrics['atiss']['total_oob_loss'], 2)} / MBL: {round(metrics['atiss'].get('total_mbl_loss'), 2)}",
				f"OOB: {round(metrics['midiff']['total_oob_loss'], 2)} / MBL: {round(metrics['midiff'].get('total_mbl_loss'), 2)}",
				f"OOB: {round(metrics['ours']['total_oob_loss'], 2)} / MBL: {round(metrics['ours'].get('total_mbl_loss'), 2)}",
				f"OOB: {round(metrics['ours-bon8']['total_oob_loss'], 2)} / MBL: {round(metrics['ours-bon8'].get('total_mbl_loss'), 2)}" if is_supp==True else "",
			]
	
	# Add textboxes to the plot
	for col_idx, text in enumerate(textbox_contents):
		if text:  # Only add textbox if there's content
			textbox = dict(boxstyle="square,pad=0.3", alpha=0.8, facecolor='white')
			axs[row_idx, col_idx].text(0.98, 0.02, text, 
									  transform=axs[row_idx, col_idx].transAxes, 
									  fontsize=metric_font_size,
									  horizontalalignment='right', 
									  verticalalignment='bottom', 
									  bbox=textbox)

def set_column_titles(mode, axs, is_supp=False, asset_sampling=False, num_asset_samples=0, skip_scene_before=False):
	title_font_size = 32

	# Handle asset sampling differently
	if asset_sampling:
		column_titles = ["Scene Before"]

		# Add sample titles
		for i in range(num_asset_samples):
			column_titles.append(f"Asset #{i+1}")

		# Add GT title if in instr mode
		if mode == "instr":
			column_titles.append("Scene After (GT)")

	elif skip_scene_before:
		# 4-column layout: no "Scene Before"
		column_titles = [
			"ATISS",
			"Mi-Diff",
			"ReSpace (ours)",
			"GT",
		]

	# Original implementation for method comparison
	else:
		column_titles = [
			"Scene Before" if is_supp==False else "GT",
			"ATISS",
			"Mi-Diff",
			"ReSpace (ours)" if mode == "instr" else ("ReSpace" if is_supp==False else r'$\mathrm{ReSpace/A}^{\dagger}$'),
			"Scene After (GT)" if mode == "instr" else ("GT" if is_supp==False else r'$\mathrm{ReSpace/A}^{\dagger}_{S8{+}R}$')
		]
	
	# Set the titles
	for col_idx, title in enumerate(column_titles):
		axs[0, col_idx].set_title(title, fontsize=title_font_size, pad=16)


def plot_qualitative_figure_ours_vs_baselines_instr():
	sample_data = {
		"bedroom": (1234, 0),
		"livingroom": (1234, 452),
		"all": (3456, 348),
		"all_fail": (1234, 180)
	}
	camera_heights = {
		"bedroom": 4.0,
		"livingroom": 6.0,
		"all": 6.5,
		"all_fail": 6.0
	}
	plot_qualitative_figure_comparison("instr", num_rows=4, sample_data=sample_data, camera_heights=camera_heights)


def plot_qualitative_figure_ours_vs_baselines_full():
	sample_data = {
		# "all": (3456, 348),
		# "all_fail": (1234, 180)
		"bedroom": (1234, 148), # 1234/78, 1234/203, 1234/221
		"livingroom": (1234, 9), #1234/70, 1234/89
	}
	camera_heights = {
		# "all": 9.5,
		# "all_fail": 9.0
		"bedroom": 5.0,
		"livingroom": 5.0,
	}
	plot_qualitative_figure_comparison("full", num_rows=2, sample_data=sample_data, camera_heights=camera_heights, skip_scene_before=True)

def plot_qualitative_figure_ours_vs_baselines_full_supp():
	sample_data = {
		"all_1": (1234, 203),
		"all_2": (1234, 221),
		"all_3": (1234, 403),
		"all_4": (3456, 19),
		"all_5": (3456, 119),
		"all_6": (5678, 120),
		"all_7": (5678, 461),
		"all_8": (5678, 391),
		"all_9": (1234, 72),
		"all_10": (3456, 93),
		"all_11": (5678, 114),
	}
	camera_heights = {
		"all_1": 5.0,
		"all_2": 6.0,
		"all_3": 4.0,
		"all_4": 5.0,
		"all_5": 6.0,
		"all_6": 5.0,
		"all_7": 7.0,
		"all_8": 7.0,
		"all_9": 5.0,
		"all_10": 7.0,
		"all_11": 6.0,
	}
	plot_qualitative_figure_comparison("full", num_rows=11, sample_data=sample_data, camera_heights=camera_heights, is_supp=True)

def render_teaser_figures():

	# bg_color = np.array([240, 240, 240]) / 255.0
	bg_color = [0, 0, 0, 0]

	# scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": []}'
	# scene_before_teaser = json.loads(scene_before_teaser)
	# render_full_scene_and_export_with_gif(scene_before_teaser, filename="teaser-0-0", pth_output=Path("./eval/viz/teaser"), create_gif=False, bg_color=bg_color, camera_height=6.0)

	scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"} ]}'
	scene_before_teaser = json.loads(scene_before_teaser)
	render_full_scene_and_export_with_gif(scene_before_teaser, filename="teaser-0-0", pth_output=Path("./eval/viz/teaser"), create_gif=False, bg_color=bg_color, camera_height=6.0)

	scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"}, {"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "pos": [-0.5, 0.0, -1.96], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.84, 1.78, 0.93], "prompt": "large artificial green plant", "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6", "sampled_asset_desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "sampled_asset_size": [0.8378599882125854, 1.7756899947231837, 0.9323999881744385], "uuid": "02a55e98-2eb7-4a36-b03b-ea063c22b9f7"}]}'
	scene_before_teaser = json.loads(scene_before_teaser)
	render_full_scene_and_export_with_gif(scene_before_teaser, filename="teaser-0-1", pth_output=Path("./eval/viz/teaser"), create_gif=False, bg_color=bg_color, camera_height=6.0)

	scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"} ]}'
	scene_before_teaser = json.loads(scene_before_teaser)
	render_full_scene_and_export_with_gif(scene_before_teaser, filename="teaser-0-2", pth_output=Path("./eval/viz/teaser"), create_gif=False, bg_color=bg_color, camera_height=6.0)

	scene_before_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"}, {"desc": "Classic wooden wardrobe with glass sliding doors and intricately carved floral details, blending traditional design elements.", "pos": [-0.13, 0.0, -2.08], "rot": [0.0, 0.0, 0.0, 1.0], "size": [1.65, 2.33, 0.6], "prompt": "a wooden wardrobe", "sampled_asset_jid": "12c73c31-4b45-42c9-ab98-268efb9768af-(0.66)-(1.0)-(0.8)", "sampled_asset_desc": "Classic wooden wardrobe with glass sliding doors and intricately carved floral details, blending traditional design elements.", "sampled_asset_size": [1.65, 2.33, 0.6], "uuid": "07902af2-ae3b-453b-8814-dfd71a7f9e09"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}]}'
	scene_before_teaser = json.loads(scene_before_teaser)
	render_full_scene_and_export_with_gif(scene_before_teaser, filename="teaser-0-3", pth_output=Path("./eval/viz/teaser"), create_gif=False, bg_color=bg_color, camera_height=6.0)

def render_instr_sample(room_type="bedroom"):
	pth_folder_fig = Path(f"./eval/viz/misc")

	# seed_and_idx = (5678, 63)
	# seed_and_idx = (5678, 264)
	# seed_and_idx = (1234, 203)
	
	# seed_and_idx = (1234, 186)
	
	# all_pths = get_pths_dataset_split(room_type, "test")
	# all_test_instrs = get_test_instrs_all(room_type)
	# pth_scene = all_pths[seed_and_idx[1]]
	# instr_sample = all_test_instrs.get(pth_scene)[seed_and_idx[0]]
	# scene_query = json.loads(instr_sample["sg_input"])
	# obj_to_add = json.loads(instr_sample["sg_output_add"])
	# scene = create_full_scene_from_before_and_added(scene_query, obj_to_add)

	# fig voxelization
	# scene = json.loads('{"room_type": "bedroom", "bounds_top": [[-1.45, 2.6, 2.45], [0.45, 2.6, 2.45], [0.45, 2.6, 1.45], [1.45, 2.6, 1.45], [1.45, 2.6, -2.45], [-1.45, 2.6, -2.45]], "bounds_bottom": [[-1.45, 0.0, 2.45], [0.45, 0.0, 2.45], [0.45, 0.0, 1.45], [1.45, 0.0, 1.45], [1.45, 0.0, -2.45], [-1.45, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "size": [0.57, 1.21, 0.63], "pos": [1.25, 0.0, 1.25], "rot": [0, 0, 0, 1], "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6-(0.68)-(0.68)-(0.68)"}, {"desc": "Elegant wooden wardrobe with three geometric-patterned glass doors, two drawers, and modern metal handles.", "size": [1.45, 2.28, 0.62], "pos": [0.87, 0.0, -2.1], "rot": [0, 0, 0, 0], "sampled_asset_jid": "a0b67c64-15a4-4969-91a6-89e365d87d12"}, {"desc": "Modern contemporary pendant lamp featuring white fabric conical shades on a geometric gold metal frame with multiple light sources.", "size": [1.06, 1.03, 0.47], "pos": [0.02, 2.08, -0.44], "rot": [0, -0.71254, 0, 0.70164], "sampled_asset_jid": "5a72093d-b9e5-4823-906b-331ced5e08d7"}, {"desc": "Modern beige upholstered king-size bed with minimalist design and neatly tailored edges.", "size": [1.9, 1.11, 2.23], "pos": [-0.29, 0.0, -0.3], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "6c7bf8e0-37a2-4661-a554-3af2b1e242d6"}, {"desc": "A modern-traditional nightstand in dark brown wood with a gold geometric patterned front, featuring two drawers and sleek elevated legs.", "size": [0.58, 0.59, 0.46], "pos": [-1.31, 0.0, -1.71], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "8b8cdbde-57e3-432a-a46a-89a77f8e6294"}, {"desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "pos": [-1.1, 0.0, 1.38], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [1.1, 1.36, 0.81], "prompt": "modern dark wooden desk", "sampled_asset_jid": "ec9190d1-cc42-4a85-bb1e-730ed7642f51", "sampled_asset_desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "sampled_asset_size": [1.1008340120315552, 1.3596680217888206, 0.8073000013828278], "uuid": "51b03ac6-941c-4beb-a8c1-84d69f8a41c1"}, {"desc": "A modern, ergonomic office chair with a mesh back, leather seat, metal frame, 360-degree swivel base, and rolling casters.", "pos": [-0.64, 0.0, 1.56], "rot": [0.0, -0.80486, 0.0, 0.59347], "size": [0.66, 0.95, 0.65], "prompt": "office chair", "sampled_asset_jid": "284277da-b2ed-4dea-bc97-498596443294", "sampled_asset_desc": "A modern, ergonomic office chair with a mesh back, leather seat, metal frame, 360-degree swivel base, and rolling casters.", "sampled_asset_size": [0.663752019405365, 0.9482090100936098, 0.6519539952278137], "uuid": "f2259272-7d9d-4015-8353-d8a5d46f1b33"}]}')

	# load from stage_2_dedup
	pth_root = os.getenv("PTH_STAGE_2_DEDUP")
	pth_scene = "0dd9e55c-dac2-4727-b8a1-f266fd11c987-a3ce1ab1-57fa-487d-8c3c-d6f1f66e984f.json"
	scene = json.load(open(os.path.join(pth_root, pth_scene), "r"))

	# order such that bed in the last
	# scene["objects"] = sorted(scene["objects"], key=lambda x: "bed" in x.get("desc").lower())
	# sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=1234, do_print=True)
	# sampling_engine.sample_last_asset(scene)
	# exit()

	# remove object with "bed" in it
	scene_before = copy.deepcopy(scene)
	scene_before["objects"] = [obj for obj in scene["objects"] if "bed" not in obj.get("desc").lower()]

	# print prompt for that object
	for obj in scene["objects"]:
		if "bed" in obj.get("desc").lower():
			jid = obj.get("jid")

	# prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS"), "r"))
	# print(prompts[jid])
	
	bg_color = [0, 0, 0, 0]
	# bg_color = np.array([240, 240, 240]) / 255.0
	camera_height = 4.5
	# camera_height = 5.0
	
	render_full_scene_and_export_with_gif(scene_before, filename="scene_before_2", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
	render_full_scene_and_export_with_gif(scene, filename="scene_after_2", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
	
	# render_full_scene_and_export_with_gif(scene, filename="scene_assets", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
	# render_full_scene_and_export_with_gif(scene, filename="scene_assets_voxels", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height, show_assets_voxelized=True)
	# render_full_scene_and_export_with_gif(scene, filename="scene_bboxes", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height, show_assets=False, show_bboxes=True)

def plot_figures_voxelization():
	obj_plant_oob = 0.010
	obj_wardrobe_oob = 0.144
	obj_lamp_oob = 0.003
	obj_bed_mbl = 0.015

	metrics = [
		{ "OOB": obj_wardrobe_oob },
		{ "OOB": obj_lamp_oob }, 
		{ "OOB": obj_plant_oob },
		{ "MBL": obj_bed_mbl },
	]

	image_paths = [
		'/Users/mnbucher/Downloads/fig-voxelization/voxel_asset_1.png',
		'/Users/mnbucher/Downloads/fig-voxelization/voxel_asset_2.png',
		'/Users/mnbucher/Downloads/fig-voxelization/voxel_asset_5.png',
		'/Users/mnbucher/Downloads/fig-voxelization/voxel_asset_3.png',
	]

	times_new_roman = fm.FontProperties(family='Times New Roman')

	fig, axs = plt.subplots(2, 2, figsize=(10, 10))
	plt.subplots_adjust(wspace=0.01, hspace=0.01)  # Minimal spacing between subplots

	# Flatten the axs array for easier iteration
	axs = axs.flatten()

	# Load and display each image with its metrics text
	for i in range(4):
		try:
			img = Image.open(image_paths[i])
			# Crop if needed
			# img = img.crop((left, top, right, bottom))
		except Exception as e:
			print(f"Error loading image {image_paths[i]}: {e}")
			# Create a placeholder if image can't be loaded
			img = np.zeros((300, 300, 3), dtype=np.uint8)
		
		# Display the image
		axs[i].imshow(np.array(img))
		
		# Create the text label with OOB/MBL values
		key, val = list(metrics[i].items())[0]
		text = f"Δ {key}: {val}"
		
		# Create textbox with semi-transparent black background
		textbox = dict(boxstyle="square,pad=0.3", alpha=0.1, facecolor='black')
		
		# Add the text at the top-right corner
		axs[i].text(
			0.98, 0.98, text, 
			transform=axs[i].transAxes, 
			fontsize=16, 
			fontproperties=times_new_roman,
			color='black',
			horizontalalignment='right', 
			verticalalignment='top', 
			bbox=textbox
		)
		
		# Turn off axis
		axs[i].axis('off')

	# Save the final figure
	plt.savefig('/Users/mnbucher/Downloads/fig-voxelization/visualization_grid.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
	plt.close(fig)

def compute_pms_score(prompt, new_obj_desc):
	if prompt is None:
		return float("inf")

	prompt_words = prompt.split(" ")
	correct_words = 0
	for word in prompt_words:
		if word in new_obj_desc.lower():
			correct_words += 1

	# Recall: how many words from the prompt are in the generated description
	score = correct_words / len(prompt_words)
	return score

# blue_colors = ['#78a5cc', '#286bad', '#0D3A66', '#011733']
blue_colors = ['#78a5cc', "#3d79b4", "#29619A", "#13417D"]
orange_colors = ['#FFC09F', '#FF9A6C', '#FF7F3F', '#FF5C00']  # From lighter to darker

def count_words(text):
	"""Count the number of words in a text string."""
	if not text:
		return 0
	return len(text.split())

def process_full_scenes_data(base_paths, seeds):
	"""Process data from full scene JSON files."""
	all_prompt_word_counts = []
	all_pms_scores = []
	all_object_counts = []
	
	total_objects = 0
	processed_files = 0
	
	for base_path in base_paths:
		for seed in seeds:
			# Construct the path for this seed
			seed_path = os.path.join(base_path, str(seed))
			
			# Get all JSON files in this directory
			json_files = glob.glob(os.path.join(seed_path, "*.json"))
			
			print(f"Found {len(json_files)} JSON files in {seed_path}")
			
			for json_file in tqdm(json_files, desc=f"Processing seed {seed} in {os.path.basename(base_path)}"):
				try:
					with open(json_file, 'r') as f:
						scene_data = json.load(f)
					
					# Check if the file has the expected structure
					if "objects" not in scene_data:
						print(f"Warning: 'objects' not found in {json_file}, skipping...")
						continue
					
					# Process each object in the scene
					for i, obj in enumerate(scene_data["objects"]):
						if "prompt" in obj and "sampled_asset_desc" in obj:
							prompt = obj["prompt"]
							desc = obj["sampled_asset_desc"]
							
							# Skip if prompt or desc is empty
							if not prompt or not desc:
								continue
							
							# Count words in prompt (not characters)
							prompt_word_count = count_words(prompt)
							
							# Calculate PMS score
							pms_score = compute_pms_score(prompt, desc)
							
							# Skip invalid scores
							if pms_score == float("inf") or np.isnan(pms_score):
								continue
							
							# Count objects up to and including this one
							object_count = i + 1
								
							all_prompt_word_counts.append(prompt_word_count)
							all_pms_scores.append(pms_score)
							all_object_counts.append(object_count)
							total_objects += 1
					
					processed_files += 1
					
				except Exception as e:
					print(f"Error processing {json_file}: {str(e)}")
	
	print(f"Processed {processed_files} files with {total_objects} valid objects")
	
	# Create a DataFrame
	df = pd.DataFrame({
		'prompt_word_count': all_prompt_word_counts,
		'pms_score': all_pms_scores,
		'object_count': all_object_counts
	})
	
	return df

def plot_pms_analysis():
	seeds = ["1234", "3456", "5678"]

	# Process data for each room split separately
	room_splits = {
		# "Bedroom": ["eval/samples/respace/full/bedroom-with-qwen1.5b-all-grpo-bon-1/json"],
		# "Livingroom": ["eval/samples/respace/full/livingroom-with-qwen1.5b-all-grpo-bon-1/json"],
		# "All": ["eval/samples/respace/full/all-with-qwen1.5b-all-grpo-bon-1/json"],

		"Bedroom": ["eval/samples/respace/full/bedroom-rej-n512fixed-1e5-bon-1-shuffling-8-bonrot-fast-v3/json"],
		"Livingroom": ["eval/samples/respace/full/livingroom-rej-n512fixed-1e5-bon-1-shuffling-8-bonrot-fast-v3/json"],
		"All": ["eval/samples/respace/full/all-rej-n512fixed-1e5-bon-1-shuffling-8-bonrot-fast-v3/json"],
	}

	split_dfs = {}
	for split_name, base_paths in room_splits.items():
		print(f"Processing {split_name} data...")
		df = process_full_scenes_data(base_paths, seeds)
		if len(df) > 0:
			split_dfs[split_name] = df
			print(f"  {split_name}: {len(df)} data points")

	if not split_dfs:
		print("No valid data found for analysis")
		return

	# Configure font styles
	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.size'] = 12
	plt.rcParams['text.usetex'] = False
	plt.rcParams['axes.unicode_minus'] = True

	# Define font sizes
	times_new_roman_size = 36
	label_font_size = 28
	tick_font_size = 28

	min_samples = 5

	# Create figure
	fig, ax1 = plt.subplots(figsize=(10, 8))

	# Plot each room split as a separate blue line
	global_word_min = float('inf')
	global_word_max = float('-inf')

	for i, (split_name, df) in enumerate(split_dfs.items()):
		color = blue_colors[i]

		# Bin and aggregate
		df['word_count_bin'] = df['prompt_word_count']
		word_agg = df.groupby('word_count_bin')['pms_score'].agg(['mean', 'std', 'count']).reset_index()
		word_agg = word_agg[word_agg['count'] >= min_samples]
		word_agg = word_agg.sort_values('word_count_bin')

		if len(word_agg) == 0:
			continue

		global_word_min = min(global_word_min, word_agg['word_count_bin'].min())
		global_word_max = max(global_word_max, word_agg['word_count_bin'].max())

		ax1.plot(
			word_agg['word_count_bin'],
			word_agg['mean'],
			'o-',
			markersize=6,
			linewidth=2,
			color=color,
			label=split_name,
		)
		ax1.fill_between(
			word_agg['word_count_bin'],
			[m - s for m, s in zip(word_agg['mean'], word_agg['std'])],
			[m + s for m, s in zip(word_agg['mean'], word_agg['std'])],
			color=color,
			alpha=0.05,
		)

	ax1.set_xlabel("Prompt Word Count", fontsize=label_font_size)
	ax1.set_ylabel("PMS", fontsize=label_font_size)
	ax1.tick_params(axis='both', labelsize=tick_font_size)
	ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

	word_ticks = list(range(int(global_word_min), int(global_word_max) + 1, 1))
	ax1.set_xticks(word_ticks)
	ax1.set_xlim(global_word_min, global_word_max - 1)
	ax1.set_ylim(0.0, 1.0)
	ax1.grid(True, linestyle='--', alpha=0.5)

	legend = ax1.legend(loc='lower right', fontsize=26)
	legend.get_frame().set_alpha(0.99)

	plt.title("PMS Variability / Full", fontsize=times_new_roman_size)
	plt.tight_layout()
	plt.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.11)
	plt.savefig("./plots/pms_relationships_all.jpg")
	plt.savefig("./plots/pms_relationships_all.pdf")

	print("Plot saved as ./plots/pms_relationships_all.svg and .pdf")

def plot_bon_full():
# Configure font styles
	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols
	plt.rcParams['font.size'] = 12  # Default size, will override where needed
	plt.rcParams['text.usetex'] = False  # Using built-in math rendering
	plt.rcParams['axes.unicode_minus'] = True  # Proper minus signs
	
	# Define font sizes
	times_new_roman_size = 36
	label_font_size = 28
	tick_font_size = 28
	
	# Create figure
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Hardcoded values for BoN scaling
	bon_samples = [1, 2, 4, 8]  # Best-of-N values
	
	# ours / OOB
	oob_values = [160.2, 133.57, 71.22, 38.28]
	oob_std = [16.0, 6.47, 2.79, 4.02]
	
	# ours / MBL
	mbl_values = [181.6, 137.05, 72.36, 78.26]
	mbl_std = [26.0, 21.90, 5.20, 5.26]
	
	# Hardcoded baseline values
	# ATISS baselines
	atiss_oob = 631.4
	atiss_oob_std = 12.9
	atiss_mbl = 108.5
	atiss_mbl_std = 6.9
	
	# Mi-Diff baselines
	midiff_oob = 327.4
	midiff_oob_std = 41.3
	midiff_mbl = 87.1
	midiff_mbl_std = 2.7
	
	# Plot the baseline horizontal dashed lines for OOB in orange
	# ax.axhline(y=atiss_oob, color=orange_colors[0], linestyle='--', linewidth=2, 
			   # label="ATISS OOB")
	# ax.axhline(y=midiff_oob, color=orange_colors[1], linestyle='--', linewidth=2,
			   # label="Mi-Diff OOB")
	
	# Plot the baseline horizontal dashed lines for MBL in blue
	ax.axhline(y=atiss_mbl, color=orange_colors[1], linestyle='--', linewidth=2, 
			   label="ATISS MBL")
	ax.axhline(y=midiff_mbl, color=orange_colors[2], linestyle='--', linewidth=2,
			   label="Mi-Diff MBL")
	
	# Plot the BoN scaling curve for OOB in orange
	ax.plot(bon_samples, oob_values, 'd-', markersize=16, linewidth=2.5, 
			color=blue_colors[2], label="$\\text{ReSpace/A}^{\\dagger}$ OOB")
	
	# Add shaded area for OOB standard deviation
	ax.fill_between(
		bon_samples,
		[v-s for v,s in zip(oob_values, oob_std)],
		[v+s for v,s in zip(oob_values, oob_std)],
		color=blue_colors[2],
		alpha=0.1
	)
	
	# Plot the BoN scaling curve for MBL in blue
	ax.plot(bon_samples, mbl_values, '*-', markersize=18, linewidth=2.5, 
			color=orange_colors[3], label="$\\text{ReSpace/A}^{\\dagger}$ MBL")
	
	# Add shaded area for MBL standard deviation
	ax.fill_between(
		bon_samples,
		[v-s for v,s in zip(mbl_values, mbl_std)],
		[v+s for v,s in zip(mbl_values, mbl_std)],
		color=orange_colors[3],
		alpha=0.1
	)
	
	# Set axis labels
	ax.set_xlabel("Best-of-N (BoN)", fontsize=label_font_size)
	ax.set_ylabel("Layout Violations (OOB / MBL) × 10³", fontsize=label_font_size)
	
	# Set title
	ax.set_title("BoN Scaling / Full — 'all' split", fontsize=times_new_roman_size)
	
	# Set x-axis to log scale to better show BoN scaling
	ax.set_xscale('log', base=2)
	
	# Format x-ticks to show actual BoN values
	ax.set_xticks(bon_samples)
	ax.set_xticklabels([str(n) for n in bon_samples], fontsize=tick_font_size)
	
	# Format y-ticks
	ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
	
	# Style tick parameters
	ax.tick_params(axis='both', labelsize=tick_font_size)
	
	# Add grid
	ax.grid(True, linestyle='--', alpha=0.5)
	
	# Set y-axis limits with some padding
	all_y_values = oob_values + mbl_values + [atiss_oob, midiff_oob, atiss_mbl, midiff_mbl]
	all_std_values = oob_std + mbl_std + [atiss_oob_std, midiff_oob_std, atiss_mbl_std, midiff_mbl_std]
	
	min_y = min([y-s for y, s in zip(all_y_values, all_std_values)]) * 0.9
	max_y = max([y+s for y, s in zip(all_y_values, all_std_values)]) * 1.1
	
	# Adjust max_y to make sure ATISS OOB is visible
	max_y = max(max_y, atiss_oob * 1.1)
	
	ax.set_ylim(min_y, 200)
	
	# Add legend with styled font
	legend = ax.legend(loc='upper right', fontsize=22, ncol=2)  # Using 2 columns for the legend
	legend.get_frame().set_alpha(0.99)
	
	# Adjust layout
	plt.tight_layout()
	plt.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.11)
	
	# Save plot
	plt.savefig("./plots/bon_scaling_oob_mbl.svg")

# plt.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.11)

def plot_qualitative_figure_ours_vs_baseline_instr_assets():
	# we want a plot with 3x5 renderings
	# first colunn is scene before, next 4 columns are 4 different assets with title ”Sample #1", "Sample #2", "Sample #3", "Sample #4"
	# for this, we will disable greedy sampling for the sampling engine, we will pick the same instrs as in the main paper for inst
	sample_data = {
		"bedroom": (1234, 0),
		"livingroom": (1234, 452),
		"all": (3456, 348),
	}
	camera_heights = {
		"bedroom": 4.0,
		"livingroom": 6.0,
		"all": 6.5,
	}
	plot_qualitative_figure_comparison("instr", num_rows=3, sample_data=sample_data, camera_heights=camera_heights, asset_sampling=True, num_asset_samples=3)

def plot_360_videos_instr():
	sample_data = {
		"bedroom": (1234, 0),
		"livingroom": (1234, 452),
		"all": (3456, 348),
	}
	camera_heights = {
		"bedroom": 5.0,
		"livingroom": 6.0,
		"all": 6.5,
	}

	pth_root = "./eval/samples"
	
	pth_folder_fig = Path(f"./eval/viz/360videos-instr")
	remove_and_recreate_folder(pth_folder_fig)
	
	# Process each room type
	for room_type, sample_info in sample_data.items():
		seed, idx = sample_info
		print(f"Creating 360° videos for {room_type} (seed={seed}, idx={idx})...")

		
		# Background color - match existing rendering
		bg_color = np.array([240, 240, 240]) / 255.0
		
		# Process instruction mode
		instr_scene_path = f"{pth_root}/respace/instr/{room_type}-with-qwen1.5b-all-grpo-bon-1/json/{seed}/{idx}_{seed}.json"
		if os.path.exists(instr_scene_path):
			print(f"Processing instruction scene at: {instr_scene_path}")
			scene = json.load(open(instr_scene_path, "r"))
			
			# Create instruction mode 360° video
			create_360_video_instr(
				scene, 
				filename=f"instr_{room_type}_{idx}_{seed}",
				room_type=room_type,
				pth_output=pth_folder_fig,
				camera_height=camera_heights[room_type],
				fps=30,
				video_duration=8.0,
				visibility_time=0.8,
				bg_color=bg_color
			)
		else:
			print(f"Warning: Instruction scene not found at {instr_scene_path}")
		
		print(f"Completed video creation for {room_type}")

def plot_360_videos_full():

	# sample_data = {
	# 	"bedroom": (1234, 148),
	# 	"livingroom": (1234, 9),
	# }
	# camera_heights = {
	# 	"bedroom": 5.0,
	# 	"livingroom": 5.0,
	# }

	sample_data = {
		# "all_1": (1234, 203),
		# "all_4": (3456, 19),
		# "all_5": (3456, 119),
		# "all_8": (5678, 391),
		# "all_9": (1234, 72),
		"all_10": (5678, 394),
	}
	camera_heights = {
		# "all_1": 5.0,
		# "all_4": 5.0,
		# "all_5": 6.0,
		# "all_8": 7.0,
		# "all_9": 5.0,
		"all_10": 6.0,
	}

	pth_root = "./eval/samples"
	
	pth_folder_fig = Path(f"./eval/viz/360videos-full")
	remove_and_recreate_folder(pth_folder_fig)
	
	# Process each room type
	for sample_key, sample_info in sample_data.items():
		# print("=====")
		room_type = sample_key.split("_")[0]
		seed, idx = sample_info
		print(f"Creating 360° full scene video for {room_type} (seed={seed}, idx={idx})...")
		
		# Background color - match existing rendering
		bg_color = np.array([240, 240, 240]) / 255.0
		
		# Process full scene mode
		# full_scene_path = f"{pth_root}/respace/full/{room_type}-with-qwen1.5b-all-grpo-bon-1/json/{seed}/{idx}_{seed}.json"
		full_scene_path = f"{pth_root}/respace/full/{room_type}-with-qwen1.5b-all-grpo-bon-8/json/{seed}/{idx}_{seed}.json"
		if os.path.exists(full_scene_path):
			print(f"Processing full scene at: {full_scene_path}")
			scene = json.load(open(full_scene_path, "r"))

			# # print prompt for each object in existing order, then skip video gen
			# for obj in scene["objects"]:
			# 	print(obj["prompt"])
			# continue
			
			# Create full scene 360° video
			create_360_video_full(
				scene,
				filename=f"full_{room_type}_{idx}_{seed}",
				room_type=room_type,
				pth_output=pth_folder_fig,
				camera_height=camera_heights[sample_key],
				fps=30,
				video_duration=8.0,
				step_time=0.8,
				bg_color=bg_color
			)
		else:
			print(f"Warning: Full scene not found at {full_scene_path}")
		
		print(f"Completed video creation for {room_type}")
	
	print("All 360° full scene videos completed!")

def plot_teaser_sample_360_video():
	scene_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"}, {"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "pos": [-0.5, 0.0, -1.96], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.84, 1.78, 0.93], "prompt": "large artificial green plant", "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6", "sampled_asset_desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "sampled_asset_size": [0.8378599882125854, 1.7756899947231837, 0.9323999881744385], "uuid": "02a55e98-2eb7-4a36-b03b-ea063c22b9f7"}]}'
	scene_teaser = json.loads(scene_teaser)
	# if object is plant or lamp, then shift down by 1cm
	for obj in scene_teaser["objects"]:
		if "plant" in obj["desc"] or "lamp" in obj["desc"]:
			obj["pos"][1] -= 0.01
	# create video in FULL style
	pth_folder_fig = Path(f"./eval/viz/360videos-teaser")
	remove_and_recreate_folder(pth_folder_fig)
	create_360_video_full(
		scene_teaser,
		filename=f"teaser_360_video",
		room_type="teaser",
		pth_output=pth_folder_fig,
		camera_height=5.5,
		fps=30,
		video_duration=8.0,
		step_time=0.8,
		bg_color=np.array([240, 240, 240]) / 255.0
	)

def plot_voxelization_360_video():

	# scene_teaser = '{"room_type": "livingroom", "bounds_top": [[-1.95, 2.6, 2.45], [-1.95, 2.6, 3.45], [-0.45, 2.6, 3.45], [-0.45, 2.6, 2.45], [1.95, 2.6, 2.45], [1.95, 2.6, -2.45], [1.95, 2.6, -3.05], [1.05, 2.6, -3.05], [1.05, 2.6, -2.45], [-1.95, 2.6, -2.45]], "bounds_bottom": [[-1.95, 0.0, 2.45], [-1.95, 0.0, 3.45], [-0.45, 0.0, 3.45], [-0.45, 0.0, 2.45], [1.95, 0.0, 2.45], [1.95, 0.0, -2.45], [1.95, 0.0, -3.05], [1.05, 0.0, -3.05], [1.05, 0.0, -2.45], [-1.95, 0.0, -2.45]], "objects": [{"desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "size": [0.75, 0.75, 0.75], "pos": [0.1, 0.0, 1.62], "rot": [0, 0.98113, 0, 0.19333], "jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_jid": "4d5a0347-ad0b-4296-990d-06b4fa622ba2", "sampled_asset_desc": "Modern pink fabric armchair with a cushioned seat, ribbed side details, and a metal swivel base.", "sampled_asset_size": [0.7486140131950378, 0.7531509538074275, 0.7511670291423798], "uuid": "dfca7d6b-55c0-4037-8dde-d02e8d000763"}, {"desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "size": [1.11, 2.06, 0.86], "pos": [-1.55, 0.0, 1.8], "rot": [0, 0, 0, 1], "jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_jid": "0f1d9021-594f-4413-ba81-092ae228b4d8-(1.0)-(1.0)-(0.75)", "sampled_asset_desc": "Artificial plant with detailed green foliage and white floral accents in a yellow square pot, ideal for contemporary interiors.", "sampled_asset_size": [1.11, 2.06, 0.86], "uuid": "4f4dc289-996e-4869-8694-633f78e9a8f8"}, {"desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "size": [1.61, 0.54, 0.45], "pos": [-1.71, 0.0, 0.38], "rot": [0, 0.70711, 0, 0.70711], "jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_jid": "43ba505f-1e4e-41ce-aabe-b45823c6b350", "sampled_asset_desc": "Modern eclectic wooden TV stand with vibrant geometric drawers in brown, red, and yellow.", "sampled_asset_size": [1.6092499494552612, 0.5361420105615906, 0.45029403269290924], "uuid": "b1c1f1ae-c502-4476-9ce1-1c66bde8b906"}, {"desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "size": [0.79, 1.68, 0.33], "pos": [1.47, 0.0, -2.46], "rot": [0, 0.92388, 0, -0.38268], "jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_jid": "c376c778-fab9-4f26-b494-fe0abdc17751-(0.88)-(1.0)-(1.0)", "sampled_asset_desc": "Modern floor lamp with a gold metal frame, arc design, and white glass spherical shade for minimalist elegance.", "sampled_asset_size": [0.79, 1.68, 0.33], "uuid": "ed9d7915-21b2-43f4-998c-75650321f05f"}, {"desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "pos": [0.01, 0.0, 0.41], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [0.77, 0.39, 0.77], "prompt": "large wooden coffee table", "sampled_asset_jid": "3bfeed24-ef65-45ec-b93f-3d1815947b02", "sampled_asset_desc": "Mid-century modern minimalist coffee table with a circular top, raised edge, and angular legs made of solid wood.", "sampled_asset_size": [0.7718539834022522, 0.39424204601546897, 0.7718579769134521], "uuid": "fc646ea9-d3e3-4bc2-8fae-a13982afa43d"}, {"desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "pos": [1.48, 0.0, 0.41], "rot": [0.0, -0.70711, 0.0, 0.70711], "size": [2.53, 0.96, 0.93], "prompt": "dark mid century sofa", "sampled_asset_jid": "12331863-e353-4926-9b40-e1b0d32e3342", "sampled_asset_desc": "This modern mid-century dark gray fabric three-seat sofa features a tufted seat cushion, square arms, cylindrical bolster pillows, and tapered wooden legs.", "sampled_asset_size": [2.531214952468872, 0.9562000231380807, 0.9328610002994537], "uuid": "75e37102-64d5-4480-b3a7-5509cec39764"}, {"desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "size": [0.8, 1.85, 0.32], "pos": [-1.77, 0.0, -1.17], "rot": [0, 0.70711, 0, 0.70711], "jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_jid": "c97bf2e1-1fa0-4267-9795-b53b19655601", "sampled_asset_desc": "A modern minimalist wood bookcase with five open shelves and a single drawer, featuring a sleek and rectangular design ideal for contemporary settings.", "sampled_asset_size": [0.8001269996166229, 1.8525430085380865, 0.32494688034057617], "uuid": "626c5ca7-2f07-4559-947e-828304dc09ae"}, {"desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "pos": [0.1, 1.75, 0.41], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.76, 0.87, 0.79], "prompt": "large white pendant lamp", "sampled_asset_jid": "01fdf241-67bb-482c-844c-61e261b8d484-(2.61)-(1.0)-(3.17)", "sampled_asset_desc": "A modern minimalist pendant lamp with a spherical design, featuring concentric rings in white and gold and suspended elegantly from a thin cable.", "sampled_asset_size": [0.76, 0.87, 0.79], "uuid": "957ed0af-da4d-490d-b4cf-6ad91e5cb90f"}, {"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "pos": [-0.5, 0.0, -1.96], "rot": [0.0, 0.0, 0.0, 1.0], "size": [0.84, 1.78, 0.93], "prompt": "large artificial green plant", "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6", "sampled_asset_desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "sampled_asset_size": [0.8378599882125854, 1.7756899947231837, 0.9323999881744385], "uuid": "02a55e98-2eb7-4a36-b03b-ea063c22b9f7"}]}'
	scene_voxelization_example = json.loads('{"room_type": "bedroom", "bounds_top": [[-1.45, 2.6, 2.45], [0.45, 2.6, 2.45], [0.45, 2.6, 1.45], [1.45, 2.6, 1.45], [1.45, 2.6, -2.45], [-1.45, 2.6, -2.45]], "bounds_bottom": [[-1.45, 0.0, 2.45], [0.45, 0.0, 2.45], [0.45, 0.0, 1.45], [1.45, 0.0, 1.45], [1.45, 0.0, -2.45], [-1.45, 0.0, -2.45]], "objects": [{"desc": "A modern minimalist artificial plant featuring a black ceramic planter, twisted trunk, and lush green foliage, ideal for contemporary spaces.", "size": [0.57, 1.21, 0.63], "pos": [1.25, 0.0, 1.25], "rot": [0, 0, 0, 1], "sampled_asset_jid": "ef223247-429e-43b4-bd72-ba6f0ae3c1f6-(0.68)-(0.68)-(0.68)"}, {"desc": "Elegant wooden wardrobe with three geometric-patterned glass doors, two drawers, and modern metal handles.", "size": [1.45, 2.28, 0.62], "pos": [0.87, 0.0, -2.1], "rot": [0, 0, 0, 0], "sampled_asset_jid": "a0b67c64-15a4-4969-91a6-89e365d87d12"}, {"desc": "Modern contemporary pendant lamp featuring white fabric conical shades on a geometric gold metal frame with multiple light sources.", "size": [1.06, 1.03, 0.47], "pos": [0.02, 2.08, -0.44], "rot": [0, -0.71254, 0, 0.70164], "sampled_asset_jid": "5a72093d-b9e5-4823-906b-331ced5e08d7"}, {"desc": "Modern beige upholstered king-size bed with minimalist design and neatly tailored edges.", "size": [1.9, 1.11, 2.23], "pos": [-0.29, 0.0, -0.3], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "6c7bf8e0-37a2-4661-a554-3af2b1e242d6"}, {"desc": "A modern-traditional nightstand in dark brown wood with a gold geometric patterned front, featuring two drawers and sleek elevated legs.", "size": [0.58, 0.59, 0.46], "pos": [-1.31, 0.0, -1.71], "rot": [0, 0.70711, 0, 0.70711], "sampled_asset_jid": "8b8cdbde-57e3-432a-a46a-89a77f8e6294"}, {"desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "pos": [-1.1, 0.0, 1.38], "rot": [0.0, 0.70711, 0.0, 0.70711], "size": [1.1, 1.36, 0.81], "prompt": "modern dark wooden desk", "sampled_asset_jid": "ec9190d1-cc42-4a85-bb1e-730ed7642f51", "sampled_asset_desc": "This modern mid-century desk features a dark brown wooden frame with an elevated shelf, clean lines, and tapered legs supported by crossbars, blending functionality with aesthetic appeal.", "sampled_asset_size": [1.1008340120315552, 1.3596680217888206, 0.8073000013828278], "uuid": "51b03ac6-941c-4beb-a8c1-84d69f8a41c1"}, {"desc": "A modern, ergonomic office chair with a mesh back, leather seat, metal frame, 360-degree swivel base, and rolling casters.", "pos": [-0.64, 0.0, 1.56], "rot": [0.0, -0.80486, 0.0, 0.59347], "size": [0.66, 0.95, 0.65], "prompt": "office chair", "sampled_asset_jid": "284277da-b2ed-4dea-bc97-498596443294", "sampled_asset_desc": "A modern, ergonomic office chair with a mesh back, leather seat, metal frame, 360-degree swivel base, and rolling casters.", "sampled_asset_size": [0.663752019405365, 0.9482090100936098, 0.6519539952278137], "uuid": "f2259272-7d9d-4015-8353-d8a5d46f1b33"}]}')

	# scene_voxelization_example = json.loads(scene_voxelization_example)
	
	# Fix flickering issues (same as in other functions)
	for obj in scene_voxelization_example["objects"]:
		if "plant" in obj["desc"] or "lamp" in obj["desc"]:
			obj["pos"][1] -= 0.01
	
	# Create output directory
	pth_folder_fig = Path("./eval/viz/360videos-voxelization")
	remove_and_recreate_folder(pth_folder_fig)

	create_360_video_voxelization(scene_voxelization_example, pth_folder_fig)
	
def plot_assets_360_video():
	
	# scene = json.load(open(f"{pth_root}/baseline-atiss/instr/all/json/3456/348_3456.json", "r"))
	scene_example = json.load(open(f"./eval/samples/respace/instr/all{'-with-qwen1.5b-all-grpo-bon-1'}/json/3456/348_3456.json", "r"))
	camera_height = 6.5,

	# Create output directory
	pth_folder_fig = Path("./eval/viz/360videos-assets")
	remove_and_recreate_folder(pth_folder_fig)

	create_360_videos_assets(scene_example, camera_height, pth_folder_fig)

def plot_histogram_corner_count_for_roomtype():
	
	# room_types = ["bedroom", "livingroom", "office", "all"]
	room_types = ["bedroom"]
	for room_type in room_types:
		print(f"Processing room type: {room_type}")
		create_histogram_corner_count_for_roomtype(
			room_type=room_type,
			filename=f"histogram_corner_count_{room_type}",
			pth_output="./plots"
		)
	
	print("All histograms completed!")	

def create_histogram_corner_count_for_roomtype(room_type, filename, pth_output):
	
	# iterate over all rooms for the given split and count corners
	counts = []
	
	# combine train/val/test splits
	for split in ["train", "val", "test"]:
		all_pths = get_pths_dataset_split(room_type, split)
		for pth_json in tqdm(all_pths):
			scene = json.load(open(os.getenv("PTH_STAGE_2_DEDUP") + f"/{pth_json}", "r"))
			bounds_top = scene["bounds_top"]
			num_corners = len(bounds_top)
			counts.append(num_corners)
	
	# Plot histogram
	plt.figure(figsize=(8, 6))
	plt.hist(counts, bins=range(min(counts)-1, max(counts) + 2), align='left', color='skyblue', edgecolor='black')
	plt.title(f'Histogram of Corner Counts for {room_type.capitalize()}')
	plt.xlabel('Number of Corners')
	plt.ylabel('Frequency')
	plt.xticks(range(min(counts), max(counts) + 1))
	plt.grid(axis='y', alpha=0.75)
	
	# Save figure
	pth_fig = pth_output + f"/{filename}.png"
	plt.savefig(pth_fig)
	plt.close()
	print(f"Saved histogram to {pth_fig}")
def render_gt_test_all(room_types=None, pth_output_base="./eval/viz/gt-renders", camera_height=5.0):
	if room_types is None:
		room_types = ["bedroom", "livingroom", "all"]
	
	bg_color = np.array([240, 240, 240]) / 255.0  # Match your existing renders
	
	for room_type in room_types:
		print(f"Processing GT scenes for room type: {room_type}")
		
		# Create output directory for this room type
		pth_output = Path(pth_output_base) / room_type / "diag"
		pth_output.mkdir(parents=True, exist_ok=True)
		
		# Get all test scene paths
		all_pths = get_pths_dataset_split(room_type, "test")
		
		print(f"Found {len(all_pths)} test scenes for {room_type}")
		
		# Process each scene
		for idx, pth_scene in enumerate(tqdm(all_pths, desc=f"Rendering {room_type} GT scenes")):
			try:
				# Load the GT scene
				scene_path = os.path.join(os.getenv("PTH_STAGE_2_DEDUP"), pth_scene)
				scene = json.load(open(scene_path, "r"))
				
				# Create filename: gt_{room_type}_{idx}.jpg
				filename = f"{idx}"

				# render_full_scene_and_export_with_gif(scene, idx, pth_output=pth_viz_output, create_gif=args.create_gifs)
				
				# Render the scene
				render_full_scene_and_export_with_gif(scene, filename=filename, pth_output=pth_output.parent, create_gif=False) #, bg_color=bg_color, camera_height=camera_height)
				
			except Exception as e:
				print(f"Error processing scene {idx} ({pth_scene}): {str(e)}")
				continue
		
		print(f"Completed rendering {len(all_pths)} GT scenes for {room_type}")

def aggregate_removal_data_across_seeds(all_analysis_data):
	"""
	Aggregate removal analysis data across seeds to compute mean and std.
	
	Args:
		all_analysis_data: List of lists, where each sublist contains data for one seed
	
	Returns:
		Dictionary with aggregated metrics for each plot type
	"""
	# Flatten all data across seeds
	all_data_flat = []
	for seed_data in all_analysis_data:
		all_data_flat.extend(seed_data)
	
	# Group by prompt_length for plot 2
	prompt_length_groups = defaultdict(list)
	for item in all_data_flat:
		prompt_length_groups[item['prompt_length']].append(item['is_success'])
	
	prompt_length_stats = {}
	for length, successes in prompt_length_groups.items():
		prompt_length_stats[length] = {
			'mean': np.mean(successes) * 100,
			'std': np.std(successes) * 100,
			'count': len(successes)
		}
	
	# Group by scene_length bins for plot 1
	scene_lengths = [item['scene_length'] for item in all_data_flat]
	min_scene_len, max_scene_len = min(scene_lengths), max(scene_lengths)
	n_bins = 10
	bin_size = (max_scene_len - min_scene_len) / n_bins
	
	scene_length_groups = defaultdict(list)
	for item in all_data_flat:
		bin_idx = int((item['scene_length'] - min_scene_len) / bin_size)
		bin_idx = min(bin_idx, n_bins - 1)  # Ensure last bin captures max value
		bin_center = min_scene_len + (bin_idx + 0.5) * bin_size
		scene_length_groups[bin_center].append(item['is_success'])
	scene_length_groups.pop(max(scene_length_groups.keys()))
	
	scene_length_stats = {}
	for bin_center, successes in scene_length_groups.items():
		scene_length_stats[bin_center] = {
			'mean': np.mean(successes) * 100,
			'std': np.std(successes) * 100,
			'count': len(successes)
		}
	
	# Group by confusion_type for plot 3 (filter out None values)
	# Track both success stats and failure counts
	confusion_groups = defaultdict(list)
	for item in all_data_flat:
		if item['confusion_type'] is not None:
			confusion_groups[item['confusion_type']].append(item['is_success'])
	
	confusion_stats = {}
	for conf_type, successes in confusion_groups.items():
		failure_count = sum(1 for s in successes if not s)  # Count failures
		confusion_stats[conf_type] = {
			'mean': np.mean(successes) * 100,
			'std': np.std(successes) * 100,
			'count': len(successes),
			'failure_count': failure_count  # Add failure count
		}
	
	return {
		'prompt_length': prompt_length_stats,
		'scene_length': scene_length_stats,
		'confusion_type': confusion_stats
	}


def plot_removal_analysis(room_type, json_path):
	"""
	Create a 1x3 plot showing removal analysis results.
	
	Args:
		room_type: The room type being analyzed (e.g., "bedroom", "livingroom", "all")
		json_path: Path to the JSON file with removal analysis data
	"""
	# Load the data
	with open(json_path, 'r') as f:
		all_analysis_data = json.load(f)
	
	# Aggregate data across seeds
	stats = aggregate_removal_data_across_seeds(all_analysis_data)
	
	# Configure font styles to match existing plots
	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.size'] = 12
	plt.rcParams['text.usetex'] = False
	plt.rcParams['axes.unicode_minus'] = True
	
	# Define font sizes
	times_new_roman_size = 36
	label_font_size = 28
	tick_font_size = 28
	
	# Blue color scheme matching existing plots
	blue_colors = ['#78a5cc', '#286bad', '#0D3A66', '#011733']
	
	# Create figure with 1x3 subplots
	fig, axes = plt.subplots(1, 3, figsize=(24, 6))
	
	# ============== PLOT 1: Scene Length vs Success Rate ==============
	ax1 = axes[0]
	scene_data = stats['scene_length']
	scene_bins = sorted(scene_data.keys())
	scene_means = [scene_data[bin]['mean'] for bin in scene_bins]
	scene_stds = [scene_data[bin]['std'] for bin in scene_bins]
	
	# Plot with dots and error bars
	ax1.plot(scene_bins, scene_means, 'o-', 
			 markersize=8, linewidth=2.5, 
			 color=blue_colors[2], label="Success Rate")
	
	# Add shaded area for standard deviation
	ax1.fill_between(
		scene_bins,
		[m-s for m,s in zip(scene_means, scene_stds)],
		[m+s for m,s in zip(scene_means, scene_stds)],
		color=blue_colors[2],
		alpha=0.1
	)
	
	ax1.set_xlabel("Scene Length (words)", fontsize=label_font_size)
	ax1.set_ylabel("Success Rate (%)", fontsize=label_font_size)
	ax1.set_title("Success Rate vs Scene Length", fontsize=times_new_roman_size, pad=16)
	ax1.tick_params(axis='both', labelsize=tick_font_size)
	ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
	ax1.grid(True, linestyle='--', alpha=0.3)
	ax1.set_ylim(0, 105)
	
	# ============== PLOT 2: Prompt Length vs Success Rate ==============
	ax2 = axes[1]
	prompt_data = stats['prompt_length']
	prompt_lengths = sorted(prompt_data.keys())
	prompt_means = [prompt_data[length]['mean'] for length in prompt_lengths]
	prompt_stds = [prompt_data[length]['std'] for length in prompt_lengths]
	
	# Plot with dots and error bars
	ax2.plot(prompt_lengths, prompt_means, 'o-', 
			 markersize=8, linewidth=2.5, 
			 color=blue_colors[2], label="Success Rate")
	
	# Add shaded area for standard deviation
	ax2.fill_between(
		prompt_lengths,
		[m-s for m,s in zip(prompt_means, prompt_stds)],
		[m+s for m,s in zip(prompt_means, prompt_stds)],
		color=blue_colors[2],
		alpha=0.1
	)
	
	ax2.set_xlabel("Prompt Length (words)", fontsize=label_font_size)
	ax2.set_ylabel("Success Rate (%)", fontsize=label_font_size)
	ax2.set_title("Success Rate vs Prompt Length", fontsize=times_new_roman_size, pad=16)
	ax2.tick_params(axis='both', labelsize=tick_font_size)
	ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
	ax2.grid(True, linestyle='--', alpha=0.3)
	ax2.set_ylim(0, 105)
	
	# ============== PLOT 3: Confusion Type Distribution ==============
	ax3 = axes[2]
	confusion_data = stats['confusion_type']
	
	# Define order and labels for confusion types
	confusion_order = ['only_same', 'only_different', 'mixed']
	confusion_labels = ['Same Class', 'Different Class', 'Mixed']
	
	# Filter to only include types that exist in the data
	existing_types = [ct for ct in confusion_order if ct in confusion_data]
	existing_labels = [confusion_labels[confusion_order.index(ct)] for ct in existing_types]
	
	failure_counts = [confusion_data[ct]['failure_count'] for ct in existing_types]
	
	# Create bar positions
	x_pos = np.arange(len(existing_types))
	
	# Plot bars without error bars
	bars = ax3.bar(x_pos, failure_counts, 
				   color=blue_colors[2], alpha=0.7, width=0.6)
	
	ax3.set_xlabel("Failure Type", fontsize=label_font_size)
	ax3.set_ylabel("Number of Failures", fontsize=label_font_size)
	ax3.set_title("Failure Count by Confusion Type", fontsize=times_new_roman_size, pad=16)
	ax3.set_xticks(x_pos)
	ax3.set_xticklabels(existing_labels, fontsize=tick_font_size, rotation=0)
	ax3.tick_params(axis='y', labelsize=tick_font_size)
	ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
	ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

	print(failure_counts)
	
	# Set ylim based on data (add some headroom)
	max_count = max(failure_counts) if failure_counts else 1
	ax3.set_ylim(0, max_count * 1.1)
	
	# Adjust layout
	plt.tight_layout()
	plt.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.15, wspace=0.25)
	
	# Save plot
	output_path = f"./plots/removal_analysis_{room_type}.pdf"
	plt.savefig(output_path, dpi=300)
	print(f"Saved removal analysis plot to {output_path}")
	
	# Also save as SVG
	output_path_svg = f"./plots/removal_analysis_{room_type}.svg"
	plt.savefig(output_path_svg)
	print(f"Saved removal analysis plot to {output_path_svg}")
	
	plt.close()

def get_image_path(folder: str, seed: int, idx: int) -> Path:
	base = Path("/home/martinbucher/git/stan-24-sgllm/eval/samples/respace/full")
	return base / folder / "viz" / str(seed) / "diag" / f"{idx}.jpg"


def crop_image(img: Image.Image) -> Image.Image:
	w, h = img.size
	return img.crop((0, int(h * 0.15), w, int(h * 0.85)))


def plot_qualitative_figure_ours_sft_old_new_dpo_rej(MODELS, HARDCODED_INDICES=None):
	seed = 1234
	num_rows = 5
	rand_seed = 44

	# --- Choose indices ---
	if HARDCODED_INDICES is not None:
		chosen_indices = HARDCODED_INDICES[:num_rows]
	else:
		rng = random.Random(rand_seed)
		chosen_indices = sorted(rng.sample(range(500), num_rows))
	print(f"Scene indices: {chosen_indices}")

	# --- Style ---
	plt.rcParams["font.family"] = "STIXGeneral"
	plt.rcParams["mathtext.fontset"] = "stix"
	plt.rcParams["font.size"] = 12
	plt.rcParams["text.usetex"] = False
	plt.rcParams["axes.unicode_minus"] = True

	title_font_size = 28

	num_cols = len(MODELS)
	fig, axs = plt.subplots(num_rows, num_cols, figsize=(4.2 * num_cols, 3.4 * num_rows))

	# --- Fill grid ---
	for row_idx, scene_idx in enumerate(chosen_indices):
		for col_idx, model_cfg in enumerate(MODELS):
			ax = axs[row_idx, col_idx]
			img_path = get_image_path(model_cfg["folder"], seed, scene_idx)

			if img_path.exists():
				img = crop_image(Image.open(img_path))
				ax.imshow(img)
			else:
				ax.set_facecolor("#d0d0d0")
				ax.text(0.5, 0.5, f"missing\n{img_path.name}",
						transform=ax.transAxes, ha="center", va="center",
						fontsize=9, color="#555")

			ax.axis("off")

			if col_idx == 0:
				ax.annotate(f"Scene #{scene_idx}", xy=(0, 0.5), xycoords="axes fraction",
							xytext=(-6, 0), textcoords="offset points",
							ha="right", va="center", fontsize=18, rotation=90)

	# --- Column titles ---
	for col_idx, model_cfg in enumerate(MODELS):
		axs[0, col_idx].set_title(model_cfg["label"], fontsize=title_font_size, pad=12)

	plt.tight_layout()
	plt.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.01, hspace=0.04, wspace=0.02)

	# os.makedirs("./plots", exist_ok=True)
	plt.savefig("./plots/ours_sft_old_new_dpo_rej.svg", bbox_inches="tight")
	#plt.savefig("./plots/ours_sft_old_new_dpo_rej.pdf", bbox_inches="tight")
	#plt.savefig("./plots/ours_sft_old_new_dpo_rej.jpg", dpi=200, bbox_inches="tight")
	print("Saved to ./plots/ours_sft_old_new_dpo_rej.{svg,pdf,jpg}")
	plt.close(fig)

def plot_seq_accuracy_vs_instr_length(room_type: str, bon_values: list = [1, 8], include_rot: bool = False):
	"""
	Plot Sequence Accuracy vs Instruction Length for one or more BoN settings.

	Reads:  ./plots/seq_eval_raw_{room_type}_bon_{bon}.json
			(+ ./plots/seq_eval_raw_{room_type}_bon_{bon}_rot.json if include_rot)
	Saves:  ./plots/seq_accuracy_vs_instr_length_{room_type}.pdf

	JSON structure (from print_seq_eval_results):
		List[           # seeds
			List[       # samples
				{"acc_seq": float, "seq_length": int, ...}
			]
		]
	"""
	plt.rcParams['font.family']        = 'STIXGeneral'
	plt.rcParams['mathtext.fontset']   = 'stix'
	plt.rcParams['font.size']          = 12
	plt.rcParams['text.usetex']        = False
	plt.rcParams['axes.unicode_minus'] = True

	times_new_roman_size = 36
	label_font_size      = 32
	tick_font_size       = 32
	legend_font_size     = 30

	# ── Build list of configs to plot ─────────────────────────────────────────
	# Each config: (label, json_path, color, linestyle)
	configs = []
	color_idx = 0
	for bon in sorted(bon_values):
		pth = f"./plots/seq_eval_raw_{room_type}_bon_{bon}.json"
		configs.append((f"BoN={bon}", pth, blue_colors[color_idx % len(blue_colors)], '-'))
		color_idx += 1
		if include_rot:
			pth_rot = f"./plots/seq_eval_raw_{room_type}_bon_{bon}_rot.json"
			configs.append((f"BoN={bon} + ROT", pth_rot, blue_colors[color_idx % len(blue_colors)], '--'))
			color_idx += 1

	# ── Load & aggregate per seq_length ───────────────────────────────────────
	series = {}  # label -> {seq_length: [acc_seq values]}

	for label, pth, _, _ in configs:
		if not os.path.exists(pth):
			print(f"  Skipping {label}: {pth} not found")
			continue
		with open(pth, "r") as f:
			all_seq_metrics = json.load(f)

		length_to_seed_means = defaultdict(list)
		for seed_metrics in all_seq_metrics:
			length_to_accs_this_seed = defaultdict(list)
			for m in seed_metrics:
				if m.get("acc_seq") is not None:
					length_to_accs_this_seed[m["seq_length"]].append(m["acc_seq"])
			for length, accs in length_to_accs_this_seed.items():
				length_to_seed_means[length].append(np.mean(accs))

		series[label] = length_to_seed_means

	# ── Figure ────────────────────────────────────────────────────────────────
	fig, ax = plt.subplots(figsize=(10, 8))

	all_lengths = sorted(set(l for label in series for l in series[label]))

	for label, _, color, linestyle in configs:
		if label not in series:
			continue
		lengths = sorted(series[label].keys())
		means   = [np.mean(series[label][l]) for l in lengths]
		stds    = [np.std(series[label][l])  for l in lengths]

		ax.plot(lengths, means, 'o' + linestyle,
				markersize=6, linewidth=2,
				color=color, label=label)
		ax.fill_between(lengths,
						[m - s for m, s in zip(means, stds)],
						[m + s for m, s in zip(means, stds)],
						color=color, alpha=0.1)
		
	# set horizontal dashed line for GT accuracy (hardcoded)
	# ax.axhline(0.95, color=blue_colors[0], linestyle='--', linewidth=1.5, label="GT (Upper Bound)")

	# ── Print GT accuracy per seq_length (recomputing acc_seq on the fly) ────
	pth_gt = f"./plots/seq_eval_raw_{room_type}_gt.json"
	if os.path.exists(pth_gt):
		with open(pth_gt, "r") as f:
			gt_metrics = json.load(f)

		# Per seed, compute mean acc per length bin
		gt_length_to_seed_means = defaultdict(list)
		for seed_metrics in gt_metrics:
			length_to_accs = defaultdict(list)
			for m in seed_metrics:
				if m.get("seq_length", 0) > 0:
					acc = (m["n_add_passed"] + m["n_remove_total"]) / m["seq_length"]
					length_to_accs[m["seq_length"]].append(acc)
			for length, accs in length_to_accs.items():
				gt_length_to_seed_means[length].append(np.mean(accs))

		print("GT upper bound (per seq_length):")
		gt_lengths = sorted(gt_length_to_seed_means)
		gt_means, gt_stds = [], []
		for L in gt_lengths:
			seed_means = gt_length_to_seed_means[L]
			m, s = np.mean(seed_means), np.std(seed_means)
			gt_means.append(m)
			gt_stds.append(s)
			print(f"  L={L:2d} | mean={m:.3f} ± {s:.3f}")

		# Plot as a dashed gray line with shaded std
		ax.plot(gt_lengths, gt_means, '--', color='gray', linewidth=2, label="Ground Truth")
		ax.fill_between(gt_lengths,
						[m - s for m, s in zip(gt_means, gt_stds)],
						[m + s for m, s in zip(gt_means, gt_stds)],
						color='gray', alpha=0.1)
	else:
		print(f"GT json not found at {pth_gt}")

	# ── Labels & title ────────────────────────────────────────────────────────
	room_label = room_type.split('-')[0]
	ax.set_title(f"Seq. Accuracy — '{room_label}' dataset",
				 fontsize=times_new_roman_size)
	ax.set_xlabel("Instruction Length", fontsize=label_font_size)
	ax.set_ylabel("Sequence Accuracy",  fontsize=label_font_size)

	# ── Ticks ─────────────────────────────────────────────────────────────────
	ax.tick_params(axis='both', labelsize=tick_font_size)
	ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
	ax.set_xticks(all_lengths)
	ax.set_xticklabels([str(l) for l in all_lengths], fontsize=tick_font_size)
	ax.set_xlim(left=all_lengths[0], right=all_lengths[-1])
	ax.set_ylim(bottom=0.0, top=1.05)

	# ── Legend ────────────────────────────────────────────────────────────────
	try:
		legend_font = fm.FontProperties(
			fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
		legend_font.set_size(legend_font_size)
		legend = ax.legend(loc='lower right', prop=legend_font)
	except Exception:
		legend = ax.legend(loc='lower right', fontsize=legend_font_size)
	legend.get_frame().set_alpha(0.99)

	# ── Grid & layout ─────────────────────────────────────────────────────────
	ax.grid(True, linestyle='--', alpha=0.3)
	plt.tight_layout()
	plt.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.12)

	suffix = f"_{room_type}_rot" if include_rot else f"_{room_type}"
	plt.savefig(f"./plots/seq_accuracy_vs_instr_length{suffix}.svg")
	plt.savefig(f"./plots/seq_accuracy_vs_instr_length{suffix}.pdf")
	plt.close()
	print(f"Saved to ./plots/seq_accuracy_vs_instr_length{suffix}.{{svg,pdf}}")

def plot_ttc_scaling():
	"""
	Test-time compute scaling plot for full scene synthesis (bedroom split).
	X-axis: per-scene runtime in seconds (linear).
	Primary Y-axis (left, blue):  Total VBL × 10³ (lower is better).
	Secondary Y-axis (right, black): PMS (higher is better).
	Win-rate annotated as a badge where available.
	"""

	# ── Hardcoded data ────────────────────────────────────────────────────────
	# (label, runtime_s, vbl_mean, vbl_std, pms_mean, winrate_or_None)
	# Runtime from LaTeX table comments (mean scene gen time, seed 5678).
	# VBL and PMS from the bedroom ablation table.
	TTC_CONFIGS = [
		# label                 rt(s)   	VBL   		   PMS    WR%
		("$B_1$", 				6.11,		134.8,	5.3,  0.69,  None),
		
		("$B_8$", 				9.88,		68.5,   2.8,  0.80,  None),
		
		("$B_1{+}R$", 			9.16,		68.9,   1.3,  0.79,  None),
		
		("$B_8{+}R$", 			16.75,		39.0,   4.7,  0.83,  None),

		("$B_1{+}S_8$", 		15.90, 		27.4,   0.5,  0.80,  70.0),
		
		("$B_8{+}S_8$", 		32.46,		15.1,   2.0,  0.90,  None),
		
		("$B_1{+}R{+}S_8$", 	16.12,		14.6,   1.7,  0.90,  None),
		
		("$B_8{+}R{+}S_8$",		139.29,		8.8,   	2.9,  0.94,  None),
	]

	# ── Font / style ──────────────────────────────────────────────────────────
	plt.rcParams['font.family']        = 'STIXGeneral'
	plt.rcParams['mathtext.fontset']   = 'stix'
	plt.rcParams['font.size']          = 12
	plt.rcParams['text.usetex']        = False
	plt.rcParams['axes.unicode_minus'] = True

	times_new_roman_size = 36
	label_font_size      = 28
	tick_font_size       = 28
	legend_font_size     = 22
	annot_font_size      = 26

	# ── Unpack ────────────────────────────────────────────────────────────────
	labels   = [c[0] for c in TTC_CONFIGS]
	runtimes = np.array([c[1] for c in TTC_CONFIGS])
	vbl_mean = np.array([c[2] for c in TTC_CONFIGS])
	vbl_std  = np.array([c[3] for c in TTC_CONFIGS])
	pms_mean = np.array([c[4] for c in TTC_CONFIGS])
	winrates = [c[5] for c in TTC_CONFIGS]

	# ── Figure with twin y-axis ───────────────────────────────────────────────
	fig, ax1 = plt.subplots(figsize=(10, 8))
	ax2 = ax1.twinx()

	x_lo = runtimes.min() * 0.75
	x_hi = runtimes.max() * 1.4          # extra right margin for labels

	# ── Faint connecting lines (sorted by runtime) ────────────────────────────
	order = np.argsort(runtimes)
	ax1.plot(runtimes[order], vbl_mean[order],
			 '-', linewidth=2.5, color=blue_colors[1], alpha=0.65, zorder=3)
	ax2.plot(runtimes[order], pms_mean[order],
			 '-', linewidth=2.5, color=orange_colors[2], alpha=0.65, zorder=3)

	# ── VBL markers + error bars (ax1, blue) ─────────────────────────────────
	marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
	for i in range(len(TTC_CONFIGS)):
		ax1.errorbar(
			runtimes[i], vbl_mean[i],
			yerr=vbl_std[i],
			fmt=marker_styles[i % len(marker_styles)],
			markersize=11,
			color=blue_colors[2],
			ecolor=blue_colors[1],
			elinewidth=1.5,
			capsize=4,
			zorder=5,
		)

	# ── PMS markers (ax2, orange) ─────────────────────────────────────────────
	for i in range(len(TTC_CONFIGS)):
		ax2.scatter(runtimes[i], pms_mean[i],
					marker=marker_styles[i % len(marker_styles)],
					s=90, color=orange_colors[3], zorder=4)

	# ── Method name labels (alternating above / below) ────────────────────────
	label_dy = [16, -12, 8, 10, -8, 8, -8, 8]
	for i, (lbl, xp, ym) in enumerate(zip(labels, runtimes, vbl_mean)):
		dy = label_dy[i]
		va = 'bottom' if dy > 0 else 'top'
		ax1.annotate(
			lbl,
			xy=(xp, ym),
			xytext=(0, dy),
			textcoords="offset points",
			fontsize=annot_font_size,
			ha='center', va=va,
			color='black',
			zorder=7,
		)

	# # ── Win-rate badges ───────────────────────────────────────────────────────
	# for i, (xp, ym, wr) in enumerate(zip(runtimes, vbl_mean, winrates)):
	#     if wr is None:
	#         continue
	#     ax1.annotate(
	#         f"WR: {wr:.0f}\\%",
	#         xy=(xp, ym),
	#         xytext=(34, 0),
	#         textcoords="offset points",
	#         fontsize=annot_font_size,
	#         color='black',
	#         fontweight='bold',
	#         va='center',
	#         bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
	#                   edgecolor='#888888', alpha=0.92),
	#         arrowprops=dict(arrowstyle="-", color='#888888', lw=1.0),
	#         zorder=8,
	#     )

	# ── X-axis (log scale, sparse clean ticks) ────────────────────────────────
	# One tick per data point causes overlap in the 6/8/9 and 16/24 clusters.
	# Instead show a small set of round reference values that don't collide.
	ax1.set_xscale('log')
	ax1.xaxis.set_minor_locator(plt.NullLocator())
	sparse_ticks = [6, 10, 25, 140]
	ax1.set_xticks(sparse_ticks)
	ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}s"))
	ax1.tick_params(axis='x', which='major', labelsize=tick_font_size - 2)
	ax1.set_xlim(x_lo, x_hi)

	# ── Y-axis limits based on our data only ──────────────────────────────────
	vbl_lo = max(0, (vbl_mean - vbl_std).min() * 0.85)
	vbl_hi = (vbl_mean + vbl_std).max() * 1.1
	ax1.set_ylim(0, vbl_hi)
	ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
	ax1.tick_params(axis='both', labelsize=tick_font_size, labelcolor='black')
	ax1.yaxis.label.set_color('black')
	ax1.spines['left'].set_color('black')

	pms_lo = pms_mean.min()
	pms_hi = min(1.0, pms_mean.max() * 1.04)
	ax2.set_ylim(0, pms_hi)
	ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
	ax2.tick_params(axis='y', labelcolor='black', labelsize=tick_font_size)
	ax2.yaxis.label.set_color('black')
	ax2.spines['right'].set_color('black')

	# ── Legend ────────────────────────────────────────────────────────────────
	from matplotlib.lines import Line2D
	legend_handles = [
		Line2D([0], [0], marker='o', color='w',
			   markerfacecolor=blue_colors[2], markersize=10,
			   label="VBL $\\times 10^3$ $\\downarrow$"),
		Line2D([0], [0], marker='o', color='w',
			   markerfacecolor=orange_colors[3], markersize=10,
			   label="PMS $\\uparrow$"),
	]
	try:
		legend_font = fm.FontProperties(
			fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
		legend_font.set_size(legend_font_size)
		legend = ax1.legend(handles=legend_handles, loc='upper right', prop=legend_font, bbox_to_anchor=(1.0, 0.92))
	except Exception:
		legend = ax1.legend(handles=legend_handles, loc='upper right',
							fontsize=legend_font_size)
	legend.get_frame().set_alpha(0.99)

	# ── Labels / title ────────────────────────────────────────────────────────
	ax1.set_title("Test-Time Compute Scaling / Full",
				  fontsize=times_new_roman_size)
	ax1.set_xlabel("Runtime Per Scene (log scale, seconds)", fontsize=label_font_size)
	ax1.set_ylabel("Total VBL $\\times 10^3$ $\\downarrow$", fontsize=label_font_size)
	ax2.set_ylabel("PMS $\\uparrow$", fontsize=label_font_size)

	ax1.grid(True, linestyle='--', alpha=0.3)

	plt.tight_layout()
	plt.subplots_adjust(left=0.12, right=0.87, top=0.93, bottom=0.11)

	plt.savefig("./plots/ttc_scaling.svg")
	plt.savefig("./plots/ttc_scaling.pdf")
	print("Saved ./plots/ttc_scaling.{svg,pdf}")
	plt.close()

if __name__ == '__main__':
	
	# load_dotenv(".env.local")
	load_dotenv(".env.stanley")

	# MODELS = [
	# 	{"label": "SFT$_{\\mathrm{old}}$",  "folder": "all-with-qwen1.5b-all-bon-1"},
	# 	{"label": "GRPO$_{\\mathrm{old}}$", "folder": "all-with-qwen1.5b-all-grpo-bon-1"},
	# 	{"label": "SFT$_{\\mathrm{new}}$",  "folder": "all-sft-feb20-bon-1"},
	# 	{"label": "DPO (2e-5)",             "folder": "all-dpo-2e5-neq-bon-1"},
	# 	{"label": "Rej. Sampling",          "folder": "all-rejn500-feb18-bon-1"},
	# ]
	# plot_qualitative_figure_ours_sft_old_new_dpo_rej(MODELS)

	# plot_seq_accuracy_vs_instr_length(room_type="all", bon_values=[1, 8], include_rot=True)
	# plot_seq_accuracy_vs_instr_length(room_type="all", bon_values=[1])

	# plot_histogram_corner_count_for_roomtype()
		
	# bon_values = [ 1, 2, 4, 8, 16 ]
	# fid_scores = [39.917, 39.823, 39.893, 40.017, 39.700 ]
	# kid_scores = [ 4.657, 4.787, 4.643, 4.760, 4.763 ]
	# delta_pbl = [ 0.029, 0.018, 0.009, 0.006, 0.004 ]
	# pms_score = [ 0.739, 0.740, 0.740, 0.739, 0.735 ]
	# plot_ablation_fid_kid_pbl_pms("ablation_instr_bon", "BON", x_values=bon_values, fid_scores=fid_scores, kid_scores=kid_scores, delta_pbl=delta_pbl, pms_score=pms_score)

	# k_values = [ 1, 2, 4, 8, 16 ]
	# fid_scores = [ 49.653, 48.577, 51.640, 52.033, 51.300 ]
	# kid_scores = [ 4.857, 3.563, 5.387, 6.280, 5.82 ]
	# delta_pbl = [ 0.224, 0.224, 0.226, 0.436, 0.313 ]
	# pms_score = [ 0.488, 0.511, 0.587, 0.608, 0.646 ]
	# plot_ablation_fid_kid_pbl_pms("ablation_full_icl_k", "ICL_K", x_values=k_values, fid_scores=fid_scores, kid_scores=kid_scores, delta_pbl=delta_pbl, pms_score=pms_score)

	# plot_stats_per_n_objects_instr("bedroom", "bedroom-with-qwen1.5b-all_qwen1.5B-all", n_aggregate_per=2)
	# plot_stats_per_n_objects_instr("livingroom", "livingroom-with-qwen1.5b-all_qwen1.5B-all", n_aggregate_per=4)
	# plot_stats_per_n_objects_instr("all", "all_qwen1.5B", n_aggregate_per=4)
	# plot_stats_per_n_objects_instr("bedroom", "bedroom-with-qwen1.5b-all-grpo-bon-1_qwen1.5b-all-grpo-bon-1", n_aggregate_per=2)
	# plot_stats_per_n_objects_instr("livingroom", "livingroom-with-qwen1.5b-all-grpo-bon-1_qwen1.5b-all-grpo-bon-1", n_aggregate_per=4)
	
	# plot_stats_per_n_objects_instr("all", "all-with-qwen1.5b-all-grpo-bon-1_qwen1.5b-all-grpo-bon-1", n_aggregate_per=4)
	plot_stats_per_n_objects_instr("all", "all-rej-n512fixed-1e5-bon-1-shuffling-1", n_aggregate_per=4)

	# plot_qualitative_figure_ours_vs_baselines_instr()
	# plot_qualitative_figure_ours_vs_baselines_full()
	
	# plot_qualitative_figure_ours_vs_baselines_full_supp()

	# plot_qualitative_figure_ours_vs_baseline_instr_assets()

	# plot_pms_analysis()

	# plot_ttc_scaling()

	# plot_bon_full()

	# plot_figures_voxelization()
	# render_instr_sample()
	# render_teaser_figures()
	# plot_teaser_sample_360_video()
	# plot_360_videos_instr()
	# plot_360_videos_full()
	# plot_assets_360_video()
	# plot_voxelization_360_video()

	# render_gt_test_all(["all"])

	# plot_removal_analysis("livingroom", "./plots/removal_analysis_data_livingroom_test.json")
	
	# test_coordinate_system()