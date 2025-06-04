#!/bin/bash

#SBATCH --job-name=unzip_tar
#SBATCH --output=unzip_tar_%j.out
#SBATCH --error=unzip_tar_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL

cd /scratch/users/mnbucher/stan24sgllm/3dfront

# cp /oak/stanford/groups/iarmeni/mnbucher/stan24sgllm/3dfront/3D-FUTURE-assets.tar.xz.STANLEY ./3D-FUTURE-assets.tar.xz.STANLEY
# mkdir -p 3D-FUTURE-assets-new
# tar -xvf 3D-FUTURE-assets.tar.xz.STANLEY -C 3D-FUTURE-assets-new/

mkdir -p ./3D-FUTURE-assets-new/
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip 3D-FUTURE-assets.zip.v2.v2 -d ./3D-FUTURE-assets-new/