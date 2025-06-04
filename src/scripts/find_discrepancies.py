import json
import os
from pathlib import Path

def find_directory_discrepancies(base_path, json_file1, json_file2):
    # Read JSON files
    with open(json_file1, 'r') as f:
        json1 = json.load(f)
    with open(json_file2, 'r') as f:
        json2 = json.load(f)
    
    # Combine keys from both JSONs
    json_ids = set(json1.keys()) | set(json2.keys())
    
    # Get all directories from filesystem
    filesystem_dirs = set()
    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        if os.path.isdir(full_path) and not item.startswith('.'):
            filesystem_dirs.add(item)
    
    # Find differences
    dirs_not_in_json = filesystem_dirs - json_ids
    json_ids_not_in_dirs = json_ids - filesystem_dirs
    
    print(f"Total directories in filesystem: {len(filesystem_dirs)}")
    print(f"Total IDs in JSON files: {len(json_ids)}")
    print(f"\nDirectories that exist but are not in JSONs ({len(dirs_not_in_json)}):")
    for dir_name in sorted(dirs_not_in_json):
        print(f"- {dir_name}")
    
    print(f"\nJSON IDs that don't exist as directories ({len(json_ids_not_in_dirs)}):")
    for id_name in sorted(json_ids_not_in_dirs):
        print(f"- {id_name}")

# Usage:
if __name__ == "__main__":
    base_path = "/home/martinbucher/git/stan-24-sgllm/data/3D-FUTURE-assets"  # or provide the full path to your directory
    json_file1 = "/home/martinbucher/git/stan-24-sgllm/data/3D-FUTURE-assets/model_info_martin.json"   # replace with your JSON paths
    json_file2 = "/home/martinbucher/git/stan-24-sgllm/data/3D-FUTURE-assets/model_info_martin_scaled_assets.json"
    
    find_directory_discrepancies(base_path, json_file1, json_file2)