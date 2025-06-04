import os
import re
import shutil
from tqdm import tqdm

# Define the directory to search
directory_path = "/Volumes/apollo11/data/3D-FUTURE-assets"

# Regular expression to match long folder names
long_folder_pattern = re.compile(r"\([0-9.]+\)-\([0-9.]+\)-\([0-9.]+\)$")

# Collect long-format folder paths
long_folders = []

for root, dirs, files in os.walk(directory_path):
    for dir_name in dirs:
        # Check if the folder name matches the long format pattern
        if long_folder_pattern.search(dir_name):
            folder_path = os.path.join(root, dir_name)
            long_folders.append(folder_path)

# Print total folders to be deleted
print(f"Found {len(long_folders)} long-format folders.")

# Use tqdm for progress tracking
for folder in tqdm(long_folders, desc="Deleting folders"):
    print(f"Deleting folder: {folder}")
    shutil.rmtree(folder)