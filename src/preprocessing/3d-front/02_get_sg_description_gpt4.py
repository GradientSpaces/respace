from openai import OpenAI
import pdb
import json
import base64
import re
import numpy as np
import trimesh
from tqdm import tqdm
import concurrent.futures
import threading
import time

# Function to encode the image
def encode_image(img_pth):
	with open(img_pth, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def compute_bounding_box_sizes(scene):
	if isinstance(scene, trimesh.Scene):
		all_bounds = np.array([geom.bounds for geom in scene.geometry.values()])
		min_bounds = np.min(all_bounds[:, 0, :], axis=0)
		max_bounds = np.max(all_bounds[:, 1, :], axis=0)
	elif isinstance(scene, trimesh.Trimesh):
		min_bounds, max_bounds = scene.bounds
	else:
		raise ValueError("The input scene is neither a trimesh.Scene nor a trimesh.Trimesh.")
	
	bbox_size = (max_bounds - min_bounds).tolist()

	bbox_size = [ round(elem, 2) for elem in bbox_size ]

	return bbox_size


def do_gpt4_request(full_prompt, base64_image, model_id):
	print(f"doing request for {model_id}")
	resp = openai_client.chat.completions.create(
			model="gpt-4o",
			messages=[
				{
					"role": "user",
					"content": [
						{"type": "text", "text": full_prompt},
							{
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{base64_image}"
								},
							},
					],
				}
			],
		)

	return resp


def get_gpt4_resp(full_prompt, base64_image, asset):
	
	answer_extr = None

	while answer_extr is None:

		try:
			response = do_gpt4_request(full_prompt, base64_image, asset.get("model_id"))
			resp = json.loads(response.json())
			answer = resp.get("choices")[0].get("message").get("content")
			answer_extr_tmp = json.loads(answer)
			if isinstance(answer_extr_tmp, dict):
				answer_extr = answer_extr_tmp

		except Exception as exc:
			print(exc)
			if resp is not None: print(resp)
			print("trying again...")

		if answer_extr is None:
			t_sleep = np.random.randint(5, 9)
			print(f"sleeping {t_sleep}s...")
			time.sleep(t_sleep)

	return answer_extr


def get_keywords(asset):

	full_prompt = gpt4v_prompt + gpt4v_postfix

	if asset.get("category") is not None:
		full_prompt = gpt4v_prompt + " Hint: It's a " + asset.get("category") + ". " + gpt4v_postfix

	img_pth = "/Volumes/apollo11/data/3D-FUTURE-assets/" + asset.get("model_id") + "/image.jpg"

	answer_extr = None

	while answer_extr is None:
	
		try:
			base64_image = encode_image(img_pth)
			
			answer_extr = get_gpt4_resp(full_prompt, base64_image, asset)

			jid = asset.get("model_id")

			# overwrite existing size because super noisy
			asset = trimesh.load(f"{PTH_DATASET}/{jid}/raw_model.glb")
			bbox_size = compute_bounding_box_sizes(asset)
			answer_extr["size"] = bbox_size

		except Exception as exc:
			print(exc)
			print(asset)

	return answer_extr


# Thread-safe dictionary and file writing
class ThreadSafeDict:
	def __init__(self):
		self.data = {}
		self.lock = threading.Lock()

	def __setitem__(self, key, value):
		with self.lock:
			self.data[key] = value

	def __getitem__(self, key):
		with self.lock:
			return self.data[key]

	def get(self, key, default=None):
		with self.lock:
			return self.data.get(key, default)


def thread_safe_file_write(file_path, data):
	with threading.Lock():
		with open(file_path, 'w', encoding='utf-8') as f:
			json.dump(data, f, ensure_ascii=False, indent=4)


# Worker function for each thread
def process_asset(index, asset, total):
	model_id = asset.get('model_id')
	print(f"Processing item {index + 1}/{total}: {model_id}")
	
	# Check if the asset has already been processed
	if model_info_martin.get(model_id) is not None:
		print(f"Skipping item {index + 1}/{total}: {model_id} (already processed)")
		return

	asset_metadata = get_keywords(asset)
	if asset_metadata:
		model_info_martin[model_id] = asset_metadata
		thread_safe_file_write(f"{PTH_DATASET}/model_info_martin.json", model_info_martin.data)


# **********************************************************************************************************

gpt4v_prompt = "Please provide a concise JSON object of the furniture item in the image using 'style', 'color', 'material', 'characteristics', and 'summary' as keys. Describe the style, noting any blends of design elements. Specify the materials used for different components (if applicable). List the key characteristics, including the shape, design features, and any distinctive elements or decorative accents. If there are multiple values for a key, use a list of strings. DO NOT build a nested JSON. The summary compactly captures the essence of the furniture’s style, functionality, and aesthetic appeal, emphasizing its unique attributes. This description should clearly differentiate this piece from others while succinctly capturing its essential properties and we will use it for object retrieval, so it should be as accurate as possible, keyword-heavy, but just be one extremely short sentence. You are an interior designer EXPERT."
gpt4v_postfix = "Only output the JSON as a plain string and nothing else."

openai_client = OpenAI(api_key = "sk-proj-n0LSwBDfjFbxT6Xx4KkwT3BlbkFJM1qUnlM8RaaXcO5A6TCz")

PTH_DATASET = "/Volumes/apollo11/data/3D-FUTURE-assets/"

# build dictionary from metadata instead of list for O(1) access
# model_info_martin = {}

try:
	with open(f"{PTH_DATASET}/model_info_martin.json", 'r') as f:
		existing_data = json.load(f)
	model_info_martin = ThreadSafeDict()
	model_info_martin.data = existing_data
	print(f"Loaded {len(existing_data)} existing entries.")
except FileNotFoundError:
	model_info_martin = ThreadSafeDict()
	print("No existing data found. Starting fresh.")

assets = json.load(open(f"{PTH_DATASET}/model_info.json"))
total_assets = len(assets)

# Process assets using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
	futures = [executor.submit(process_asset, i, asset, total_assets) for i, asset in enumerate(assets)]
	for future in tqdm(concurrent.futures.as_completed(futures), total=total_assets, desc="overall progress"):
		future.result()

# for asset in tqdm(json.load(open(f"{PTH_DATASET}/model_info.json"))):

	# asset_metadata = get_keywords(asset)
	# model_info_martin[asset.get("model_id")] = asset_metadata

	# with open(f"{PTH_DATASET}/model_info_martin.json", 'w', encoding='utf-8') as f:
		# json.dump(model_info_martin, f, ensure_ascii=False, indent=4)

	# exit()