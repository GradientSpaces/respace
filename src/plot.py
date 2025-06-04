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

from src.sample import AssetRetrievalModule
from src.utils import get_pth_mesh, create_floor_plan_polygon, remove_and_recreate_folder, precompute_fid_scores_for_caching, get_pths_dataset_split, get_model, get_test_instrs_all
from src.dataset import load_train_val_test_datasets, create_full_scene_from_before_and_added, create_instruction_from_scene, process_scene_sample
from src.viz import render_full_scene_and_export_with_gif, create_360_video_instr, create_360_video_full, create_360_video_voxelization, create_360_videos_assets
from src.eval import eval_scene

def plot_ablation_fid_kid_pbl_pms(title, x_name, x_values, fid_scores, kid_scores, delta_pbl, pms_score):
		
	# Create a figure with 1x3 subplots
	fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

	fig.suptitle(f'Ablation Study: {title}', fontsize=16)
	
	# Plot 1: FID and KID Scores
	ax1 = axes[0]
	ax1_twin = ax1.twinx()
	ax1.plot(bon_values, fid_scores, 'g-', marker='o')
	ax1_twin.plot(bon_values, kid_scores, 'b-', marker='o')
	ax1.set_xlabel(x_name)
	ax1.set_ylabel('FID Score', color='g')
	ax1_twin.set_ylabel('KID Score', color='b')
	ax1.set_title(f'FID and KID Scores vs {x_name}')
	ax1.set_xticks(bon_values)
	ax1.set_xticklabels(bon_values, rotation=45)
	ax1.tick_params(axis='y', labelcolor='g')
	ax1_twin.tick_params(axis='y', labelcolor='b')
	#ax1.set_ylim(39.5, 40.5)
	#ax1_twin.set_ylim(4.5, 5.0)
	ax1.grid(alpha=0.3)
	
	# Plot 2: Delta PBL
	ax2 = axes[1]
	ax2.plot(bon_values, delta_pbl, 'g-', marker='o')
	ax2.set_xlabel(x_name)
	ax2.set_ylabel('Delta PBL')
	ax2.set_title(f'Delta PBL vs {x_name}')
	ax2.set_xticks(bon_values)
	ax2.set_xticklabels(bon_values, rotation=45)
	#ax2.set_ylim(0, 0.03)
	ax2.grid(alpha=0.3)
	
	# Plot 3: PMS Score
	ax3 = axes[2]
	ax3.plot(bon_values, pms_score, 'g-', marker='o')
	ax3.set_xlabel(x_name)
	ax3.set_ylabel('PMS Score')
	ax3.set_title(f'PMS Score vs {x_name}')
	ax3.set_xticks(bon_values)
	ax3.set_xticklabels(bon_values, rotation=45)
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
	import matplotlib.font_manager as fm
	
	# Create figure with shared x-axis
	fig, ax1 = plt.subplots(figsize=(10, 8))

	plt.rcParams['font.family'] = 'STIXGeneral'
	plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols
	plt.rcParams['font.size'] = 12  # Default size, will override where needed
	plt.rcParams['text.usetex'] = False  # Using built-in math rendering
	plt.rcParams['axes.unicode_minus'] = True  # Proper minus signs

	times_new_roman_size = 36
	
	# Better blue color palette - more coherent and professional
	blue_colors = ['#78a5cc', '#286bad', '#0D3A66', '#011733']  # Ordered from lighter to darker
	
	# Get delta_pbl results for each model
	n_objects_sorted1, delta_pbl_mean1, delta_pbl_std1, _, _ = get_stats_per_n_object_from_file(
		f"eval_samples_baseline-atiss_instr_{room_type}_raw.json", 
		n_aggregate_per=n_aggregate_per
	)
	
	n_objects_sorted2, delta_pbl_mean2, delta_pbl_std2, _, _ = get_stats_per_n_object_from_file(
		f"eval_samples_baseline-midiff_instr_{room_type}_raw.json", 
		n_aggregate_per=n_aggregate_per
	)
	
	# n_objects_sorted3, delta_pbl_mean3, delta_pbl_std3, _, _ = get_stats_per_n_object_from_file(
	#     f"eval_samples_respace_instr_{room_type}_raw_llama1b.json", 
	#     n_aggregate_per=n_aggregate_per
	# )
	
	n_objects_sorted4, delta_pbl_mean4, delta_pbl_std4, _, _ = get_stats_per_n_object_from_file(
		f"eval_samples_respace_instr_{postfix}_raw.json", 
		n_aggregate_per=n_aggregate_per
	)
	
	# Get floor areas data from the first file - they're the same for all models
	_, _, _, floor_areas, floor_std = get_stats_per_n_object_from_file(
		f"eval_samples_baseline-atiss_instr_{room_type}_raw.json", 
		n_aggregate_per=n_aggregate_per
	)
	
	# Use less intrusive error visualization - alpha and smaller markers
	# Plot delta_pbl lines with error regions instead of bars
	ax1.plot(n_objects_sorted1, delta_pbl_mean1, 'o-', 
			 markersize=6, linewidth=2, 
			 color=blue_colors[0], label="ATISS")
	ax1.fill_between(n_objects_sorted1, 
					[m-s for m,s in zip(delta_pbl_mean1, delta_pbl_std1)],
					[m+s for m,s in zip(delta_pbl_mean1, delta_pbl_std1)],
					color=blue_colors[0], alpha=0.1)
	
	ax1.plot(n_objects_sorted2, delta_pbl_mean2, 'o-', 
			 markersize=6, linewidth=2, 
			 color=blue_colors[1], label="Mi-Diff")
	ax1.fill_between(n_objects_sorted2, 
					[m-s for m,s in zip(delta_pbl_mean2, delta_pbl_std2)],
					[m+s for m,s in zip(delta_pbl_mean2, delta_pbl_std2)],
					color=blue_colors[1], alpha=0.1)
	
	# Plot the fourth dataset (Ours-1.5B) also using blue scheme
	ax1.plot(n_objects_sorted4, delta_pbl_mean4, 'o-', 
			 markersize=6, linewidth=2, 
			 color=blue_colors[3], label="$\\text{ReSpace/A}^{\\dagger}$")
	ax1.fill_between(n_objects_sorted4, 
					[m-s for m,s in zip(delta_pbl_mean4, delta_pbl_std4)],
					[m+s for m,s in zip(delta_pbl_mean4, delta_pbl_std4)],
					color=blue_colors[3], alpha=0.1)
	
	# Set up secondary y-axis for floor area
	ax2 = ax1.twinx()
	
	# Plot floor area as a shaded polygon
	ax2.plot(n_objects_sorted1, floor_areas, linewidth=1.5, color='#9a9a9a', linestyle='--')
	
	# Create shaded area below the floor area line
	# Convert to arrays if they're not already
	n_objects_array = np.array(n_objects_sorted1)
	floor_array = np.array(floor_areas)
	
	# Create polygon vertices for shaded area
	x_poly = np.concatenate([n_objects_array, np.flip(n_objects_array)])
	y_poly = np.concatenate([floor_array, np.zeros_like(floor_array)])
	
	# Plot the shaded area
	ax2.fill(x_poly, y_poly, alpha=0.1, color='#7a7a7a', label="Floor Area")
	
	# Calculate max range for x-axis
	max_n_objects = max(max(n_objects_sorted1 or [0]), 
					  max(n_objects_sorted2 or [0]), 
					  # max(n_objects_sorted3 or [0]), 
					  max(n_objects_sorted4 or [0]))
	
	# Create appropriate bins for x-ticks based on aggregation parameter
	x_ticks = list(range(1, max_n_objects + 1, n_aggregate_per))
	x_tick_labels = []
	for i in range(0, len(x_ticks), 1):
		start = x_ticks[i]
		end = start + n_aggregate_per - 1
		x_tick_labels.append(f"{start}-{end}")
	
	ax1.set_title(f"Delta VBL / Instr — ‘{room_type.split('-')[0]}’ dataset", fontsize=times_new_roman_size)
	
	# Create smaller font size for labels
	# label_font = fm.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
	label_font_size = 32
	
	ax1.set_xlabel("# of objects", fontsize=label_font_size)
	ax1.set_ylabel("Δ VBL", fontsize=label_font_size)
	ax2.set_ylabel("Mean Floor Area (m²)", fontsize=label_font_size)

	tick_font_size = 32
	
	ax1.tick_params(axis='y', labelcolor='black')
	ax2.tick_params(axis='y', labelcolor='black')
	
	ax1.set_xticks(x_ticks)
	ax1.set_xticklabels(x_tick_labels, fontsize=tick_font_size)

	# round y-axis ticks to 2 decimal places
	ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

	# set ylim_min of both y-axis to the min value
	ax1_ylim_min = min(min(delta_pbl_mean1), min(delta_pbl_mean2), min(delta_pbl_mean4))
	ax1.set_ylim(bottom=ax1_ylim_min)
	ax2_ylim_min = min(floor_areas)
	print(ax1_ylim_min, ax2_ylim_min)
	ax2.set_ylim(bottom=ax2_ylim_min)

	# set to steps of 40
	ax1.set_yticks(np.arange(20, 160, 40))

	# for ax2, set yticks to be values in steps divisible by 5 but do NOT show other values
	yticks = ax2.get_yticks()
	yticks = [tick for tick in yticks if tick % 5 == 0]
	if room_type == "all":
		# remove 10.0 and 15.0
		yticks = [tick for tick in yticks if tick != 10.0 and tick != 15.0]
	elif room_type == "livingroom":
		# remove 25.0
		yticks = [tick for tick in yticks if tick != 25.0]
	ax2.set_yticks(yticks)
	ax2.set_yticklabels([f"{tick:.0f}" for tick in yticks], fontsize=tick_font_size)

	ax1.set_xlim(left=x_ticks[0], right=x_ticks[-1])
	ax2.set_xlim(left=x_ticks[0], right=x_ticks[-1])
	
	# Apply tick font size to y-tick labels
	for label in ax1.get_yticklabels() + ax2.get_yticklabels():
		label.set_size(tick_font_size)
	
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	
	legend_font = fm.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
	legend_font.set_size(30)
	
	legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=legend_font)
	legend.get_frame().set_alpha(0.99)
	# Additional customization of legend frame
	# legend.get_frame().set_edgecolor('black')
	
	# Add grid for better readability
	ax1.grid(True, linestyle='--', alpha=0.3)
	
	plt.tight_layout()

	# make spacing even tighter
	plt.subplots_adjust(left=0.16, right=0.9, top=0.94, bottom=0.12)
	plt.savefig(f"./plots/delta_pbl_vs_n_objects_{postfix}_with_area.pdf")
	# plt.savefig(f"./plots/delta_pbl_vs_n_objects_{room_type}_with_area.png", dpi=300)

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
		scene = json.load(open(f"{pth_root}/respace/{mode}/{room_type}{'-with-qwen1.5b-all-grpo-bon-1'}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
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
				scene = json.load(open(f"{pth_root}/respace/{mode}/{room_type}{'-with-qwen1.5b-all-grpo-bon-8'}/json/{seed_and_idx[0]}/{seed_and_idx[1]}_{seed_and_idx[0]}.json", "r"))
				render_full_scene_and_export_with_gif(scene, filename="0-4", pth_output=pth_folder_fig, create_gif=False, bg_color=bg_color, camera_height=camera_height)
		
		return None

def plot_qualitative_figure_comparison(mode, num_rows=4, sample_data=None, camera_heights=None, is_supp=False, asset_sampling=False, num_asset_samples=0):
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
		
		for col_idx in range(num_cols):
			img_path = Path(f"{row_prefix}") / "diag" / f"0-{col_idx}.jpg"
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
		add_metrics_to_plot(axs, row_idx, row_metrics, mode=mode, is_supp=is_supp, asset_sampling=asset_sampling, num_asset_samples=num_asset_samples)
		
		row_idx += 1
	
	# Set titles for columns
	set_column_titles(mode, axs, is_supp=is_supp, asset_sampling=asset_sampling, num_asset_samples=num_asset_samples)
	
	# Adjust layout
	if mode == "instr":
		# instr
		# fig.subplots_adjust(left=0.015, right=1.0, top=0.95, bottom=0.0, hspace=0.05, wspace=0.0)
		fig.subplots_adjust(left=0.015, right=1.0, top=0.93, bottom=0.0, hspace=0.05, wspace=0.0)
	else:
		# full
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
	ours_file = f"{base_path}eval_samples_respace_{mode}_{dataset_type}-with-qwen1.5b-all-grpo-bon-1_qwen1.5b-all-grpo-bon-1_raw.json"
	ours_file_bon8 = f"{base_path}eval_samples_respace_{mode}_{dataset_type}-with-qwen1.5b-all-grpo-bon-8_qwen1.5b-all-grpo-bon-8_raw.json"
	
	metrics_ours = json.load(open(ours_file, "r"))[seed_idx][idx]
	metrics_ours_bon8 = json.load(open(ours_file_bon8, "r"))[seed_idx][idx] if os.path.exists(ours_file_bon8) else None
	
	return {
		"atiss": metrics_atiss,
		"midiff": metrics_midiff,
		"ours": metrics_ours,
		"ours-bon8": metrics_ours_bon8 if mode == "full" else None
	}


def add_metrics_to_plot(axs, row_idx, metrics, mode="instr", is_supp=False, asset_sampling=False, num_asset_samples=0):
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

def set_column_titles(mode, axs, is_supp=False, asset_sampling=False, num_asset_samples=0):
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
	
	# Original implementation for method comparison
	else:
		column_titles = [
			"Scene Before" if is_supp==False else "GT",
			"ATISS",
			"Mi-Diff",
			"ReSpace (ours)" if mode == "instr" else ("ReSpace (ours)" if is_supp==False else "ReSpace (ours) (BON=1)"),
			"Scene After (GT)" if mode == "instr" else ("GT" if is_supp==False else "ReSpace (ours) (BON=8)")
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
	plot_qualitative_figure_comparison("full", num_rows=2, sample_data=sample_data, camera_heights=camera_heights)

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

blue_colors = ['#78a5cc', '#286bad', '#0D3A66', '#011733']
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
	
	# Process data for all room types
	base_paths = [
		"eval/samples/respace/full/bedroom-with-qwen1.5b-all-grpo-bon-1/json",
		"eval/samples/respace/full/livingroom-with-qwen1.5b-all-grpo-bon-1/json"
		"eval/samples/respace/full/all-with-qwen1.5b-all-grpo-bon-1/json"
	]
	
	print("Processing all room data...")
	df = process_full_scenes_data(base_paths, seeds)
	
	if len(df) > 0:
		print(f"\nTotal data points: {len(df)}")
		
		# Configure font styles to match example
		plt.rcParams['font.family'] = 'STIXGeneral'
		plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols
		plt.rcParams['font.size'] = 12  # Default size, will override where needed
		plt.rcParams['text.usetex'] = False  # Using built-in math rendering
		plt.rcParams['axes.unicode_minus'] = True  # Proper minus signs
		
		# Define font sizes
		times_new_roman_size = 36
		label_font_size = 28
		tick_font_size = 28
		
		# Create bins for prompt word count and object count
		word_bin_size = 1  # Each word gets its own bin
		obj_bin_size = 1   # Each object count gets its own bin
		
		# Bin the data
		df['word_count_bin'] = (df['prompt_word_count'] // word_bin_size) * word_bin_size
		df['object_count_bin'] = (df['object_count'] // obj_bin_size) * obj_bin_size
		
		# Aggregate data for prompt word count
		word_agg = df.groupby('word_count_bin')['pms_score'].agg(['mean', 'std', 'count']).reset_index()
		
		# Aggregate data for object count
		obj_agg = df.groupby('object_count_bin')['pms_score'].agg(['mean', 'std', 'count']).reset_index()
		
		# Filter bins with too few samples for reliability
		min_samples = 5  # Minimum number of samples per bin
		word_agg = word_agg[word_agg['count'] >= min_samples]
		obj_agg = obj_agg[obj_agg['count'] >= min_samples]
		
		# Sort the aggregated data by bin value
		word_agg = word_agg.sort_values('word_count_bin')
		obj_agg = obj_agg.sort_values('object_count_bin')
		
		# Create figure with two x-axes
		fig, ax1 = plt.subplots(figsize=(10, 8))
		
		# Plot prompt word count vs PMS score
		ax1.plot(
			word_agg['word_count_bin'], 
			word_agg['mean'], 
			'o-',  
			markersize=6, 
			linewidth=2,  
			color=blue_colors[0], 
			label="Prompt Word Count"
		)
		ax1.fill_between(
			word_agg['word_count_bin'], 
			[m-s for m,s in zip(word_agg['mean'], word_agg['std'])],
			[m+s for m,s in zip(word_agg['mean'], word_agg['std'])],
			color=blue_colors[0], 
			alpha=0.1
		)
		
		# Create a twin x-axis for object count
		ax2 = ax1.twiny()
		
		# Plot object count vs PMS score on the same y-axis
		ax2.plot(
			obj_agg['object_count_bin'], 
			obj_agg['mean'], 
			'o-',  
			markersize=6, 
			linewidth=2,  
			color=blue_colors[2], 
			label="Object Count"
		)
		ax1.fill_between(
			obj_agg['object_count_bin'], 
			[m-s for m,s in zip(obj_agg['mean'], obj_agg['std'])],
			[m+s for m,s in zip(obj_agg['mean'], obj_agg['std'])],
			color=blue_colors[2], 
			alpha=0.1
		)
		
		# Set up the axes labels with styled fonts
		ax1.set_xlabel("Prompt Word Count", fontsize=label_font_size)
		ax1.set_ylabel("PMS", fontsize=label_font_size)
		ax2.set_xlabel("# of objects", fontsize=label_font_size)
		
		# Style tick parameters
		ax1.tick_params(axis='both', labelsize=tick_font_size)
		ax2.tick_params(axis='x', labelsize=tick_font_size)
		
		# Round y-axis ticks to 1 decimal place
		ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
		
		# Set x-axis limits for both axes
		word_min = min(word_agg['word_count_bin'])
		word_max = max(word_agg['word_count_bin'])
		obj_min = min(obj_agg['object_count_bin'])
		obj_max = max(obj_agg['object_count_bin'])
		
		# Create appropriate bins for x-ticks to match style
		word_ticks = list(range(int(word_min), int(word_max) + 1, 1))
		obj_ticks = list(range(int(obj_min), int(obj_max) + 1, 2))
		
		# Set tick positions and labels
		ax1.set_xticks(word_ticks)
		ax2.set_xticks(obj_ticks)
		
		# Set x-axis limits
		ax1.set_xlim(word_min - 0.5, word_max + 0.5)
		ax2.set_xlim(obj_min - 0.5, obj_max + 0.5)
		
		# Add grid for better readability
		ax1.grid(True, linestyle='--', alpha=0.5)

		# set y axis max to 1.0
		ax1.set_ylim(0.4, 1.05)
		
		# Combine legends from both axes using styled font
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		
		legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=26)
		legend.get_frame().set_alpha(0.99)
		
		# Set title with styled font
		plt.title("PMS Variability / Full — merged datasets", fontsize=times_new_roman_size)
		
		# Adjust layout to match example
		plt.tight_layout()
		plt.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.11)
		
		# Save as both JPG and SVG
		# plt.savefig("./plots/pms_relationships_combined.jpg", dpi=300)
		plt.savefig("./plots/pms_relationships_combined.svg")
		
		print("Plot saved as ./plots/pms_relationships_combined.jpg and .svg")
	else:
		print("No valid data found for analysis")

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
	ax.set_title("BoN Scaling / Full — 'all' dataset", fontsize=times_new_roman_size)
	
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

if __name__ == '__main__':
	
	# load_dotenv(".env.local")
	load_dotenv(".env.stanley")
		
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
	plot_stats_per_n_objects_instr("all", "all-with-qwen1.5b-all-grpo-bon-1_qwen1.5b-all-grpo-bon-1", n_aggregate_per=4)
	# plot_stats_per_n_objects_instr("all", "all-with-qwen1.5b-all-grpo-bon-1_qwen1.5b-all-grpo-bon-1", n_aggregate_per=4)

	# plot_qualitative_figure_ours_vs_baselines_instr()
	# plot_qualitative_figure_ours_vs_baselines_full()
	# plot_qualitative_figure_ours_vs_baselines_full_supp()

	# plot_qualitative_figure_ours_vs_baseline_instr_assets()

	# plot_pms_analysis()

	# render_instr_sample()

	# plot_figures_voxelization()

	# render_teaser_figures()

	# plot_bon_full()

	# plot_teaser_sample_360_video()

	# plot_360_videos_instr()
	# plot_360_videos_full()

	# plot_assets_360_video()

	# plot_voxelization_360_video()