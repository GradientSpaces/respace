import os
import json
import pickle
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, upload_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import zipfile
import tempfile
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import pdb

# Import your existing functions
from src.utils import get_pths_dataset_split
from src.dataset import create_dataset_from_files


def create_raw_zip():
	"""Create a zip file with all raw scene JSON files for direct download."""
	print("Creating raw dataset zip file...")
	
	pth_root = os.getenv("PTH_STAGE_2_DEDUP")
	if not pth_root:
		raise ValueError("PTH_STAGE_2_DEDUP environment variable not set")
	
	# Collect all unique scene files first
	all_scene_files = set()
	for room_type in ["bedroom", "livingroom", "all"]:
		for split in ["train", "val", "test"]:
			try:
				pths = get_pths_dataset_split(room_type, split)
				all_scene_files.update(pths)
			except Exception as e:
				print(f"Warning: Could not process {room_type} {split}: {e}")
	
	print(f"Found {len(all_scene_files)} unique scene files")
	
	# Create temporary zip file
	temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
	
	with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
		# Add all unique scene files
		for pth_scene in tqdm(all_scene_files, desc="Adding unique scenes"):
			full_path = os.path.join(pth_root, pth_scene)
			if os.path.exists(full_path):
				arcname = f"scenes/{pth_scene}"
				zipf.write(full_path, arcname)
		
		# Add splits metadata
		splits_file = os.path.join(os.getenv("PTH_STAGE_3"), "all_splits.pkl")
		if os.path.exists(splits_file):
			zipf.write(splits_file, "splits/all_splits.pkl")
		
		# Add CSV splits if they exist
		for room_type in ["bedroom", "livingroom", "all"]:
			splits_pkl = os.path.join(os.getenv("PTH_STAGE_3"), f"{room_type}_splits.pkl")
			if os.path.exists(splits_pkl):
				zipf.write(splits_pkl, f"splits/{room_type}_splits.pkl")
				
			splits_csv = os.path.join(os.getenv("PTH_STAGE_3"), f"{room_type}_splits.csv")
			if os.path.exists(splits_csv):
				zipf.write(splits_csv, f"splits/{room_type}_splits.csv")
	
	print(f"Raw dataset zip created: {temp_zip.name}")
	return temp_zip.name


def create_hf_dataset():
	"""Create HuggingFace Dataset format with proper splits."""
	print("Creating HuggingFace dataset format...")
	
	# Dictionary to collect unique scenes and their split memberships
	unique_scenes = {}
	
	# Process all three room types with their splits
	for room_type in ["bedroom", "livingroom", "all"]:
		print(f"Processing {room_type} dataset...")
		
		for split in ["train", "val", "test"]:
			print(f"  Processing {split} split...")
			
			# Use your existing dataset creation function
			temp_path = f"/tmp/dataset_{room_type}_{split}.pkl"
			dataset = create_dataset_from_files(temp_path, room_type, split)
			
			# Process each sample
			for i in tqdm(range(len(dataset)), desc=f"Processing {room_type} {split} data"):
				sample = dataset[i]
				file_id = sample["pth_orig_file"].replace(".json", "")
				split_name = f"{room_type}_{split}"
				
				if file_id in unique_scenes:
					# Scene already exists, add this split to its list
					unique_scenes[file_id]["splits"].append(split_name)
				else:
					# Clean the scene data to ensure consistent schema
					clean_scene = {
						"room_type": sample["scene"]["room_type"],
						"bounds_top": sample["scene"]["bounds_top"],
						"bounds_bottom": sample["scene"]["bounds_bottom"],
						"objects": []
					}
					
					# Clean objects data
					for obj in sample["scene"].get("objects", []):
						clean_obj = {
							"desc": obj["desc"],
							"size": obj["size"],
							"pos": obj["pos"],
							"rot": obj["rot"],
							"jid": obj["jid"]
						}
						clean_scene["objects"].append(clean_obj)
					
					# Add room_id if it exists
					if "room_id" in sample["scene"]:
						clean_scene["room_id"] = sample["scene"]["room_id"]
					
					# New scene, create entry
					unique_scenes[file_id] = {
						"file_id": file_id,
						"room_type": sample["room_type"],
						"n_objects": sample["n_objects"],
						"scene": clean_scene,
						"splits": [split_name]
					}
			
			# Clean up temp file
			if os.path.exists(temp_path):
				os.remove(temp_path)
	
	# Now organize scenes into train/val/test splits
	split_datasets = {"train": [], "val": [], "test": []}
	
	for scene in unique_scenes.values():
		splits = scene["splits"]
		
		# Determine which HF split this scene should go to
		# Priority: if it's in any "all" split, use that; otherwise use the first split
		target_split = None
		for split_name in splits:
			if split_name.startswith("all_"):
				target_split = split_name.split("_")[1]  # train/val/test
				break
		
		# If no "all" split found, use the first available split
		if target_split is None:
			target_split = splits[0].split("_")[1]  # train/val/test
		
		# Add to the appropriate split
		split_datasets[target_split].append(scene)
	
	print(f"Total unique scenes: {sum(len(scenes) for scenes in split_datasets.values())}")
	
	# Show split statistics
	for split, scenes in split_datasets.items():
		print(f"{split}: {len(scenes)} scenes")
		
		# Show breakdown by original splits
		split_counts = {}
		for scene in scenes:
			for orig_split in scene["splits"]:
				split_counts[orig_split] = split_counts.get(orig_split, 0) + 1
		print(f"  Original split breakdown: {split_counts}")
	
	# Create DatasetDict with proper train/val/test splits
	final_datasets = {}
	for split, data_list in split_datasets.items():
		if data_list:  # Only create split if it has data
			final_datasets[split] = Dataset.from_list(data_list)
	
	return DatasetDict(final_datasets)


def create_readme(dataset_stats=None):
	if dataset_stats is None:
		dataset_stats = {
			"train": {"num_examples": 0, "num_bytes": 0},
			"val": {"num_examples": 0, "num_bytes": 0},
			"test": {"num_examples": 0, "num_bytes": 0},
			"total_size": 0
		}

	readme_content = f"""---
license: cc-by-nc-sa-4.0
task_categories:
- text-generation
- text2text-generation
- robotics
language:
- en
tags:
- 3d-scenes
- indoor-scenes
- furniture
- spatial-reasoning
- text-to-3d
- scene-synthesis
- computer-graphics
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
	path: "data/train-*"
  - split: val
	path: "data/val-*"
  - split: test
	path: "data/test-*"
---

"""
	
	readme_content += """# SSR-3DFRONT: Structured Scene Representation for 3D Indoor Scenes

This dataset provides a processed version of the 3D-FRONT dataset with structured scene representations for text-driven 3D indoor scene synthesis and editing.

## Dataset Description

SSR-3DFRONT contains 13,055 valid indoor scenes with:
- Explicit room boundaries as rectilinear polygons
- Natural language object descriptions via GPT-4o
- Comprehensive object metadata including position, size, and orientation
- Train/val/test splits for training your machine learning models

## Dataset Structure

### Splits
- **Train**: ~12K scenes for training
- **Val**: ~1.5K scenes for validation
- **Test**: ~1.5K scenes for testing

### Room Types
- **`bedroom'**: Bedroom scenes
- **`livingroom'**: Living rooms, dining rooms, and combined spaces
- **`all'**: Combined dataset with all room types

### Features

Each scene contains:
- `file_id`: Unique identifier for the scene
- `room_type`: Type of room (bedroom/livingroom/other)
- `n_objects`: Number of objects in the scene
- `scene`: Complete scene (in SSR with additional props)
- `splits`: Which splits this scene belongs to (e.g., bedroom_train, livingroom_val)

## Usage

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("gradient-spaces/SSR-3DFRONT")

train_data = dataset["train"]
val_data = dataset["val"]
test_data = dataset["test"]

# Filter by specific splits
bedroom_train = train_data.filter(lambda x: "bedroom_train" in x["splits"])
all_val = val_data.filter(lambda x: "all_val" in x["splits"])

# Example: Get a scene
scene = train_data[0]
print(f"Room type: {scene['room_type']}")
print(f"Number of objects: {scene['n_objects']}")
print(f"Scene: {scene['scene']}")
```

### Loading Raw Files (if using our codebase or for other purposes)

```python
import requests
import zipfile

# Download raw dataset
url = "https://huggingface.co/datasets/gradient-spaces/SSR-3DFRONT/resolve/main/raw_scenes.zip"
response = requests.get(url)
with open("ssr_3dfront_raw.zip", "wb") as f:
	f.write(response.content)

# Extract
with zipfile.ZipFile("ssr_3dfront_raw.zip", 'r') as zip_ref:
	zip_ref.extractall("ssr_3dfront_raw/")
```

## Data Format

### Scene JSON Structure
```json
{
  "room_type": "bedroom",
  "bounds_top": [[-1.55, 2.6, 1.9], [1.55, 2.6, 1.9], ...],
  "bounds_bottom": [[-1.55, 0.0, 1.9], [1.55, 0.0, 1.9], ...],
  "objects": [
	{
	  "desc": "Modern minimalist bed with wooden frame and gray headboard",
	  "size": [1.77, 0.99, 1.94],
	  "pos": [0.44, 0.0, -0.44],
	  "rot": [0.0, 0.70711, 0.0, -0.70711],
	  "jid": "asset_id_reference"
	}
  ]
}
```

### Object Properties
- `desc`: Natural language description of the object
- `size`: [width, height, depth] in meters
- `pos`: [x, y, z] position in room coordinates
- `rot`: Quaternion rotation [x, y, z, w]
- `jid`: Reference to 3D asset in 3D-FUTURE catalog

## Preprocessing Details

The dataset was created by:
1. Filtering 3D-FRONT scenes for validity and quality
2. Extracting explicit room boundaries as rectilinear polygons
3. Generating natural language descriptions using GPT-4o
4. Creating train/val/test splits with proper scene distribution
5. Applying data augmentation (rotation, perturbation, boundary shifting)

## Citation

If you use this dataset, please cite:

```bibtex
@article{bucher2025respace,
  title={ReSpace: Text-Driven 3D Scene Synthesis and Editing with Preference Alignment},
  ...
  TODO: add full citation
}

@inproceedings{fu20213d_front,
  title={3d-front: 3d furnished rooms with layouts and semantics},
  author={Fu, Huan and Cai, Bowen and Gao, Lin and Zhang, Ling-Xiao and Wang, Jiaming and Li, Cao and Zeng, Qixun and Sun, Chengyue and Jia, Rongfei and Zhao, Binqiang and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10933--10942},
  year={2021}
}

@article{fu20213d_future,
  title={3d-future: 3d furniture shape with texture},
  author={Fu, Huan and Jia, Rongfei and Gao, Lin and Gong, Mingming and Zhao, Binqiang and Maybank, Steve and Tao, Dacheng},
  journal={International Journal of Computer Vision},
  volume={129},
  pages={3313--3337},
  year={2021},
  publisher={Springer}
}
```

## License

This dataset is released under the same license as the original 3D-FRONT dataset. Please refer to the original dataset terms of use.

## Acknowledgments

- Original 3D-FRONT dataset by Alibaba
- 3D-FUTURE asset catalog
- OpenAI GPT-4o for description generation
"""
	return readme_content


def upload_dataset():
	parser = argparse.ArgumentParser(description="Push SSR-3DFRONT dataset to HuggingFace Hub")
	parser.add_argument("--repo-id", default="gradient-spaces/SSR-3DFRONT", help="HuggingFace repo ID")
	parser.add_argument("--skip-raw", action="store_true", help="Skip creating raw zip file")
	parser.add_argument("--skip-hf", action="store_true", help="Skip creating HF dataset format")
	parser.add_argument("--dry-run", action="store_true", help="Don't actually upload, just prepare files")
	
	args = parser.parse_args()
	
	# Initialize HF API
	api = HfApi()
	
	print(f"Preparing to upload dataset to: {args.repo_id}")
	
	# Check environment variables
	required_envs = ["PTH_STAGE_2_DEDUP", "PTH_STAGE_3", "PTH_ASSETS_METADATA_PROMPTS"]
	for env_var in required_envs:
		if not os.getenv(env_var):
			raise ValueError(f"Environment variable {env_var} not set")
	
	# Create repository if it doesn't exist
	if not args.dry_run:
		try:
			api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
			print(f"Repository {args.repo_id} ready")
		except Exception as e:
			print(f"Error creating repository: {e}")
			return
	
	# Create and upload raw zip file
	if not args.skip_raw:
		print("\n=== Creating Raw Dataset Files ===")
		raw_zip_path = create_raw_zip()
		
		if not args.dry_run:
			print("Uploading raw dataset zip...")
			upload_file(
				path_or_fileobj=raw_zip_path,
				path_in_repo="raw_scenes.zip",
				repo_id=args.repo_id,
				repo_type="dataset",
				commit_message="Add raw scene files for direct download"
			)
			print("Raw dataset zip uploaded successfully!")
		
		# Clean up temp file
		os.unlink(raw_zip_path)
	
	# Create and upload HF dataset format
	if not args.skip_hf:
		print("\n=== Creating HuggingFace Dataset Format ===")
		dataset = create_hf_dataset()
		
		print(f"Dataset created with {len(dataset)} unique scenes")

		# Collect dataset statistics - Fixed to use "val" consistently
		dataset_stats = {
			"train": {"num_examples": len(dataset.get("train", [])), "num_bytes": 0},
			"val": {"num_examples": len(dataset.get("val", [])), "num_bytes": 0},  # Changed from "validation"
			"test": {"num_examples": len(dataset.get("test", [])), "num_bytes": 0},
			"total_size": 0
		}
		
		for split_name, split_data in dataset.items():
			print(f"{split_name}: {len(split_data)} scenes")
			# Fixed the mapping logic
			dataset_stats[split_name]["num_examples"] = len(split_data)
		
		total_examples = sum(stats["num_examples"] for stats in dataset_stats.values() if isinstance(stats, dict))
		dataset_stats["total_size"] = total_examples  # Rough estimate
		
		if not args.dry_run:
			print("Uploading HuggingFace dataset...")
			dataset.push_to_hub(
				args.repo_id,
				commit_message="Add deduplicated dataset with split membership tracking"
			)
			print("HuggingFace dataset uploaded successfully!")
	
	# Create and upload README
	print("\n=== Creating Documentation ===")
	readme_content = create_readme()
	
	if not args.dry_run:
		with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
			f.write(readme_content)
			readme_path = f.name
		
		upload_file(
			path_or_fileobj=readme_path,
			path_in_repo="README.md",
			repo_id=args.repo_id,
			repo_type="dataset",
			commit_message="Add comprehensive dataset documentation"
		)
		
		os.unlink(readme_path)
		print("README uploaded successfully!")
	
	print(f"\n🎉 Dataset preparation complete!")
	print(f"📊 Dataset URL: https://huggingface.co/datasets/{args.repo_id}")
	
	if args.dry_run:
		print("(Dry run mode - nothing was actually uploaded)")

def test_hf_dataset():

	# Load the full dataset
	dataset = load_dataset("gradient-spaces/SSR-3DFRONT")

	train_data = dataset["train"]
	val_data = dataset["val"] 
	test_data = dataset["test"]

	# import pdb
	# pdb.set_trace()

	# for each split, print the number of scenes for each room type in "splits" key
	room_types = ["bedroom", "livingroom", "all"]
	for split_name, split_data in dataset.items():
		print(f"\n{split_name} split:")
		for room_type in room_types:
			count = sum(1 for scene in split_data if f"{room_type}_{split_name}" in scene["splits"])
			print(f"  {room_type}: {count} scenes")

	# get all train scenes from bedroom
	# bedroom_train = train_data.filter(lambda x: "bedroom_train" in x["splits"])
	# all_val = val_data.filter(lambda x: "all_val" in x["splits"])
	pdb.set_trace()

	# Example: Get a scene
	scene = train_data[0]
	print(f"Room type: {scene['room_type']}")
	print(f"Number of objects: {scene['n_objects']}")
	print(f"Scene: {scene['scene']}")

	scene = val_data[0]
	print(f"Room type: {scene['room_type']}")
	print(f"Number of objects: {scene['n_objects']}")
	print(f"Scene: {scene['scene']}")

	scene = test_data[0]
	print(f"Room type: {scene['room_type']}")
	print(f"Number of objects: {scene['n_objects']}")
	print(f"Scene: {scene['scene']}")

def create_model_readme():
    """Create a minimal README for the ReSpace SG-LLM model."""
    
    readme_content = """---
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
language:
- en
pipeline_tag: text-generation
tags:
- 3d-scenes
- indoor-scenes
- furniture
- fine-tuned
- qwen2.5
- respace
- sg-llm
- spatial-reasoning
- text-to-3d
- scene-synthesis
- computer-graphics
---

# respace-sg-llm-1.5b

Fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for 3D indoor scene synthesis.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gradient-spaces/respace-sg-llm-1.5b")
tokenizer = AutoTokenizer.from_pretrained("gradient-spaces/respace-sg-llm-1.5b")
```

## More Information

For detailed usage instructions, training details, and examples, see the associated repository: https://github.com/GradientSpaces/respace
"""
    
    return readme_content

def upload_model():
	parser = argparse.ArgumentParser(description="Push ReSpace SG-LLM model to Hugging Face Hub")
	parser.add_argument("--dry-run", action="store_true", help="Prepare files without uploading")
	
	args = parser.parse_args()

	# Hardcoded configuration
	local_model_path = "./ckpts/64663807/checkpoint-best"
	repo_name = "respace-sg-llm-1.5b"
	organization = "gradient-spaces"
	base_model = "Qwen/Qwen2.5-1.5B-Instruct"
	private = False
	
	# Initialize HF API
	api = HfApi()
	
	repo_id = f"{organization}/{repo_name}"
	print(f"Preparing to push model to: {repo_id}")
	
	# Create repository if it doesn't exist
	if not args.dry_run:
		try:
			api.create_repo(repo_id, private=private, exist_ok=True)
			print(f"✅ Repository {repo_id} ready")
		except Exception as e:
			print(f"❌ Error creating repository: {e}")
			return False
	
	# Create a temporary directory for additional files
	with tempfile.TemporaryDirectory() as temp_dir:
		temp_model_path = os.path.join(temp_dir, "model")
		
		# Copy model files to temp directory
		import shutil
		shutil.copytree(local_model_path, temp_model_path)
		
		# Create README
		readme_content = create_model_readme()
		
		readme_path = os.path.join(temp_model_path, "README.md")
		with open(readme_path, "w", encoding="utf-8") as f:
			f.write(readme_content)
		
		print("✅ Generated README.md")
		
		# Verify we can load the model (optional validation)
		try:
			print("🔍 Validating model can be loaded...")
			tokenizer = AutoTokenizer.from_pretrained(temp_model_path)
			# Just check if model config loads, don't load full model to save memory
			config_path = os.path.join(temp_model_path, "config.json")
			with open(config_path, 'r') as f:
				config = json.load(f)
			print(f"✅ Model validation passed. Model type: {config.get('model_type', 'unknown')}")
		except Exception as e:
			print(f"⚠️  Warning: Model validation failed: {e}")
			if not args.dry_run:
				response = input("Continue with upload anyway? (y/N): ")
				if response.lower() != 'y':
					return False
		
		if args.dry_run:
			print("🏃 Dry run mode - model files prepared but not uploaded")
			print(f"Files ready in: {temp_model_path}")
			print("Contents:")
			for root, dirs, files in os.walk(temp_model_path):
				level = root.replace(temp_model_path, '').count(os.sep)
				indent = ' ' * 2 * level
				print(f"{indent}{os.path.basename(root)}/")
				subindent = ' ' * 2 * (level + 1)
				for file in files:
					file_path = os.path.join(root, file)
					size = os.path.getsize(file_path)
					print(f"{subindent}{file} ({size:,} bytes)")
			return True
		
		# Upload the model
		try:
			print("🚀 Uploading model to Hub...")
			api.upload_folder(
				folder_path=temp_model_path,
				repo_id=repo_id,
				commit_message=f"Upload ReSpace SG-LLM fine-tuned model",
				ignore_patterns=["*.pyc", "__pycache__", ".git"]
			)
			print(f"✅ Model uploaded successfully!")
			print(f"🌐 Model URL: https://huggingface.co/{repo_id}")
			
		except Exception as e:
			print(f"❌ Error uploading model: {e}")
			return False
	
	return True

if __name__ == "__main__":
	load_dotenv(".env.stanley")
	
	test_hf_dataset()
	
	# upload_dataset()

	# upload_model()