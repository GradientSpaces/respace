import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict

from src.utils import get_room_type_from_id
from src.eval import eval_scene

def collect_room_stats(pth_root, pths_scenes):
	stats = {
		"bedroom": defaultdict(list),
		"livingroom": defaultdict(list),
		"all": defaultdict(list)
	}
	
	for pth in tqdm(pths_scenes):
		pth = os.path.join(pth_root, pth)
		scene = json.load(open(pth))
		room_type = scene.get("room_type")
		
		# Collect metrics
		n_objects = len(scene.get("objects"))
		n_corners = len(scene.get("bounds_bottom"))
		bounds = np.array(scene.get("bounds_bottom"))
		x_span = np.max(bounds[:, 0]) - np.min(bounds[:, 0])
		z_span = np.max(bounds[:, 2]) - np.min(bounds[:, 2])
		
		# Store for "all" category
		stats["all"]["objs"].append(n_objects)
		stats["all"]["corners"].append(n_corners)
		stats["all"]["x_spans"].append(x_span)
		stats["all"]["z_spans"].append(z_span)
		
		# Store for specific room type if matching
		if room_type in ["bedroom", "livingroom"]:
			stats[room_type]["objs"].append(n_objects)
			stats[room_type]["corners"].append(n_corners)
			stats[room_type]["x_spans"].append(x_span)
			stats[room_type]["z_spans"].append(z_span)
	
	return stats

def get_axis_limits(stats, metric):
	all_values = []
	max_count = 0
	for room_type in stats.keys():
		values = stats[room_type][metric]
		all_values.extend(values)
		# Get histogram counts for this room type
		if metric == "objs":
			bins = range(0, int(max(values)) + 10, 2)
		elif metric == "corners":
			bins = range(0, int(max(values)) + 2, 2)
		else:  # spans
			bins = range(0, int(max(values)) + 1, 2)
		counts, _ = np.histogram(values, bins=bins)
		max_count = max(max_count, max(counts))
	return min(all_values), max(all_values), max_count

def show_scene_stats(pth_root, pths_scenes):
	# Collect all stats in one pass
	stats = collect_room_stats(pth_root, pths_scenes)
	
	# Create 3x4 subplot grid
	fig, axs = plt.subplots(3, 4, figsize=(18, 12))
	
	# Get purple colors from tab20b
	colors = plt.cm.tab20b([0, 1, 2])  # First three purples from tab20b
	colors = {
		"bedroom": colors[0],
		"livingroom": colors[1],
		"all": colors[2]
	}
	
	room_types = ["bedroom", "livingroom", "all"]
	metrics = ["objs", "corners", "x_spans", "z_spans"]
	titles = ["Number of Objects", "Number of Corners", "Max Span on x-coord", "Max Span on z-coord"]
	
	# Calculate global min/max for each metric
	y_max_per_column = {metric: 0 for metric in metrics}
	bins_per_metric = {}
	
	# First determine bins and count max for y-axis
	for metric in metrics:
		min_val, max_val, max_count = get_axis_limits(stats, metric)
		if metric == "objs":
			bins = range(0, int(max_val) + 10, 2)
		elif metric == "corners":
			bins = range(0, int(max_val) + 2, 2)
		else:  # spans
			bins = range(0, int(max_val) + 1, 1)
		bins_per_metric[metric] = bins
		y_max_per_column[metric] = max_count
		
		# Calculate max count across all room types
		# for room_type in room_types:
			# counts, _ = np.histogram(stats[room_type][metric], bins=bins)
			# y_max_per_column[metric] = max(y_max_per_column[metric], max(counts))
	
	# Plot with standardized axes
	for row, room_type in enumerate(room_types):
		for col, (metric, title) in enumerate(zip(metrics, titles)):
			ax = axs[row, col]
			values = stats[room_type][metric]
			
			if len(values) == 0:
				ax.text(0.5, 0.5, 'No data', ha='center', va='center')
				continue
			
			bins = bins_per_metric[metric]
			ax.hist(values, bins=bins, color=colors[room_type], alpha=0.8)
			ax.set_title(f'{title}\n({room_type})')
			ax.set_xlabel(title.split()[-1])
			ax.set_ylabel('Count')
			
			# set y axis to log
			ax.set_yscale('log')
			
			# Set consistent y-axis limit for each column
			ax.set_ylim(0, y_max_per_column[metric] * 1.1)  # Add 10% padding
			
			# Set x-axis ticks
			ax.set_xticks(list(bins)[::2])
			ax.tick_params(axis='x', rotation=45)
			
			# Add grid for better readability
			ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.show()

# Main execution
if __name__ == "__main__":
	
	# load_dotenv(".env.local")
	load_dotenv(".env.stanley")
	
	# pth_root = os.getenv("PTH_STAGE_2_DEDUP")
	# pths_scenes = [f for f in os.listdir(pth_root) if f.endswith('.json') and not f.startswith(".")]
	# show_scene_stats(pth_root, pths_scenes)
	
	# **********************************************************************************************************

	# print stats of STAGE 1
	# room_counts = defaultdict(int)
	# all_pths = [f for f in os.listdir(os.getenv("PTH_STAGE_1")) if f.endswith('.json') and not f.startswith(".")]
	# for pth_scene in tqdm(all_pths):
	# 	scene = json.load(open(os.path.join(os.getenv("PTH_STAGE_1"), pth_scene)))
	# 	room_type = get_room_type_from_id(scene.get("room_id"))
	# 	room_counts[room_type] += 1
	# print("STAGE 1 STATS")
	# print(room_counts)

	# # print stats of deduplicated dataset to compare
	# room_counts = defaultdict(int)
	# all_pths = [f for f in os.listdir(os.getenv("PTH_STAGE_2")) if f.endswith('.json') and not f.startswith(".")]
	# for pth_scene in tqdm(all_pths):
	# 	scene = json.load(open(os.path.join(os.getenv("PTH_STAGE_2"), pth_scene)))
	# 	room_counts[scene.get("room_type")] += 1
	# print("STAGE 2 STATS (BEFORE DEDUP)")
	# print(room_counts)

	# print final stats
	# room_counts = defaultdict(int)
	# all_pths = [f for f in os.listdir(os.getenv("PTH_STAGE_2_DEDUP")) if f.endswith('.json') and not f.startswith(".")]
	# for pth_scene in tqdm(all_pths):
	# 	scene = json.load(open(os.path.join(os.getenv("PTH_STAGE_2_DEDUP"), pth_scene)))
	# 	room_counts[scene.get("room_type")] += 1
	# print("STAGE 2 STATS (AFTER DEDUP)")
	# print(room_counts)

	# get PBL stats
	# room_pbls = defaultdict(list)
	all_pths = [f for f in os.listdir(os.getenv("PTH_STAGE_2_DEDUP")) if f.endswith('.json') and not f.startswith(".")]
	# all_pths = all_pths[:10]

	for pth_scene in tqdm(all_pths):
		scene = json.load(open(os.path.join(os.getenv("PTH_STAGE_2_DEDUP"), pth_scene)))

		# for every object, check if product of "size" is bigger than 150 ?
		for obj in scene.get("objects"):
			if obj.get("size") is not None:
				if (obj.get("size")[0] * obj.get("size")[1] * obj.get("size")[2]) > 150.0:
					print(f"{pth_scene} — object {obj.get('jid')} has size {obj.get('size')}")
					# break

		# metrics = eval_scene(scene, is_debug=False)
		# room_type = scene.get("room_type")
		# if room_type == "bedroom" or room_type == "livingroom":
			# room_pbls[room_type].append(metrics["total_pbl_loss"])
		# room_pbls["all"].append(metrics["total_pbl_loss"])
	
	# colors = plt.cm.tab20b([0, 1, 2])
	# bins = np.arange(0, 0.1, 0.005)
	
	# # Get consistent y-axis limit across all histograms
	# y_max_count = -np.inf
	# for room_type in room_pbls.keys():
	# 	counts, _ = np.histogram(room_pbls[room_type], bins=bins)
	# 	y_max_count = max(y_max_count, max(counts))

	# fig, axs = plt.subplots(1, 3, figsize=(18, 6))

	# for i, key in enumerate(["bedroom", "livingroom", "all"]):

	# 	counts, _ = np.histogram(room_pbls[key], bins=bins)

	# 	# axs[i].hist(room_pbls[key], bins=len(bins), color=colors[i])
	# 	axs[i].set_title(f'PBL distribution for {key}')
	# 	axs[i].set_xlabel('PBL')
	# 	axs[i].set_ylabel('Count')

	# 	# make width of histogram bars consistent
	# 	axs[i].bar(bins[:-1], counts, width=bins[1] - bins[0], align='edge', color=colors[i])
		
	# 	# Set consistent y-axis limit for each column
	# 	axs[i].set_ylim(0, y_max_count * 1.1) # Add 10% padding
		
	# 	# Set x-axis ticks
	# 	axs[i].set_xticks(list(bins)[::2])
	# 	axs[i].tick_params(axis='x', rotation=45)
		
	# 	# Add grid for better readability
	# 	axs[i].grid(True, alpha=0.3)

	# # write stats to file
	# with open("pbl_stats.json", "w") as f:
	# 	json.dump(room_pbls, f, indent=4)

	# plt.tight_layout()
	# plt.show()