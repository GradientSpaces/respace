#!/bin/bash

#SBATCH --job-name=create_tar
#SBATCH --output=create_tar_%j.out
#SBATCH --error=create_tar_%j.err
#SBATCH --time=14:00:00
#SBATCH --mem=32G

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /oak/stanford/groups/iarmeni/mnbucher/stan24sgllm/3dfront

tar -czf 3D-FUTURE-assets.tar.gz 3D-FUTURE-assets/