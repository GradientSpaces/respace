# Threed-Front Dataset
<p>
    <img width="180" alt="Example 1" src="demo/example_gif/Bedroom-4098.gif"/>
    <img width="180" alt="Example 2" src="demo/example_gif/LivingDiningRoom-37001.gif"/>
    <img width="180" alt="Example 3" src="demo/example_gif/LivingDiningRoom-54997.gif"/>
    <img width="180" alt="Example 4" src="demo/example_gif/DiningRoom-11628.gif"/>
</p>

This is developed as a part of [MiDiffusion](https://github.com/MIT-SPARK/MiDiffusion), a mixed diffusion model for 3D indoor scene synthesis.
This repository provides a standalone python package for dataset classes related to [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) and [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) datasets. We also include evaluation scripts for the 3D indoor scene synthesis problem.

Most of the code is adapted from [ATISS](https://github.com/nv-tlabs/ATISS/). 
We also incorporate features from [DiffuScene](https://github.com/tangjiapeng/DiffuScene/) so that the evaluation scripts are compatible with their setup. 
The optional floor plan preprocessing script converting images to points is adapted from [LEGO-Net](https://github.com/QiuhongAnnaWei/LEGO-Net/).
Please refer to the <a href="./external_licenses/">external_licenses</a> directory for their licensing information. 

## Installation
We list dependency versions in `requirements.txt`. The following dependencies are required:
- [numpy](https://numpy.org/doc/stable/user/install.html)
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [pyrr](https://pyrr.readthedocs.io/en/latest/index.html)
- [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [PyTorch & Torchvision](https://pytorch.org/get-started/locally/)
- [scipy](https://scipy.org/install/)
- [tqdm](https://github.com/tqdm/tqdm)

These are required to process raw meshes and visualize the scenes:
- [seaborn](https://seaborn.pydata.org/)
- [simple-3dviz==0.7.0](https://simple-3dviz.com/)
- [trimesh](https://github.com/mikedh/trimesh)
- [opencv-python](https://opencv.org/get-started/) (optional, better than pillow in saving textured layouts to images)
- [wxPython==4.1.0](https://wxpython.org/index.html) (optional, for simple-3dviz GUI)

These additional dependencies are required for evaluation:
- [clean-fid](https://www.cs.cmu.edu/~clean-fid/) (optional, for FID and KID)
- [shapely](https://shapely.readthedocs.io/en/stable/installation.html) (optional, for bounding box analysis)

`pip install wxPython` might fail looking for a specific wheel and automatically switch to building from source. 
For Linux, the easiest way to install wxPython is to use the pre-built wheels from https://extras.wxpython.org/wxPython4/extras/linux/. For example,
```
pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.1.0-cp38-cp38-linux_x86_64.whl
```
To use simple-3dviz on a remote ubuntu18 or ubuntu20 server, follow [this link](https://moderngl.readthedocs.io/en/5.6.2/the_guide/headless_ubunut18_server.html).

## Data Preprocessing
We use the same dataset splits and data filtering method as [ATISS](https://github.com/nv-tlabs/ATISS/). The dataset files in `dataset_files/` are directly copied from [here](https://github.com/nv-tlabs/ATISS/tree/e643000de5990c2325653afa86174957f0f0e8de/config). Parsed and preprocessed data will be saved to an `output/` directory. The default output paths can be modified in `scripts/utils.py`.

### Download the datasets
You need to first obtain the 3D-FRONT and the 3D-FUTURE datasets. To download both datasets, please follow the download instructions from [the official website](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset). This repository is developed based on [this release](https://tianchi.aliyun.com/dataset/65347). 

### Parse 3D-FRONT
First, use `pickle_threed_front_dataset.py` to pickle a list of `Room` objects to `threed_front.pkl`:
```
python scripts/pickle_threed_front_dataset.py <dataset_dir>/3D-FRONT/3D-FRONT <dataset_dir>/3D-FUTURE/3D-FUTURE-model <dataset_dir>/3D-FUTURE/3D-FUTURE-model/model_info.json
```
This script will save the parsed rooms to `output/threed_front.pkl` by default.

### Parse 3D-FUTURE by room types
Use `pickle_threed_future_dataset.py` to pickle a `ThreedFutureDataset` object which contains a list of `ThreedFutureModel` (furnitures in the training set of the specified room type) to `threed_future_model_<room_type>.pkl`:
```
python scripts/pickle_threed_future_dataset.py threed_front_<room_type>
```
This script will save the parsed dataset to `output/threed_future_model_<room_type>.pkl` by default. 

### Preprocess 3D-FRONT by room types
Use `preprocess_data.py` to extract and save basic object encoding (`boxes.npz`), floor plan (`room_mask.png`), and top-down projection of the layout (`rendered_scene_256.png`) for each scene to a separate subdirectory: 
```
python scripts/preprocess_data.py threed_front_<room_type>
```
By default, the results are saved to `output/3d_front_processed/<room_type>` with each sub-directory named as the UID of that scene.
Note that texture (`--no_texture`) and floor plan (`--without_floor`) options will result in different rendered image names.
This means you can run this script multiple times and save multiple layout images to each sub-directory by allowing to overwrite when prompted (choose option 1 to save/overwrite rendered images without changing other files).
If you want to use point cloud embeddings as an additional object feature, please refer to the [DiffuScene repository](https://github.com/tangjiapeng/DiffuScene/).

### (Optional) Sample floor plan boundary
Instead of a binary mask, the floor plan can also be represented as sampled 2D points and normals along the boundary. Use `preprocess_floorplan.py` to add these features to each scene data (`boxes.npz`). 
```
python scripts/preprocess_floorplan.py output/3d_front_processed/<room_type> --room_side <room_side>
```
You need to make sure `--room_side` is set close to the value used to generate `room_mask.png` in the data directory. By default, this script samples 256 points per room. You can adjust this using the `--n_sampled_points` argument.

### Render scenes
To render ground-truth scenes, you can run `render_threedfront_scene.py` with a scene ID (e.g., Bedroom-4098):
```
python scripts/render_threedfront_scene.py <scene_id>
```
Please read the script for visualization options.

## Evaluation
You can `pip install` this package as a dependency for your own project. 
For evaluation, you need to save the generated results as a `ThreedFrontResults` object (see `threed_front/evaluation/__init__.py`) and pickle it to a file (e.g., `results.pkl`). Our [MiDiffusion repository](https://github.com/MIT-SPARK/MiDiffusion) contains an example script (`scripts/generate_results.py`). 
Then you can render results to top-down projection images in the same way as it is done for ground-truth layouts in the preprocessing step:
```
python scripts/render_results.py <result_file>
```
By default, the rendered images will be saved to the same directory as the result file. Please read the script for visualization options.

Our evaluation scripts include:
- `evaluate_kl_divergence_object_category.py`: Compute **KL-divergence** between ground-truth and synthesized object category distributions.
- `compute_fid_scores.py`: Compute average **FID** or **KID** (if run with "--compute_kid" flag) between ground-truth and synthesized layout images.
- `synthetic_vs_real_classifier.py`: Train image classifier to distinguish real and synthetic projection images, and compute average **classification accuracy**.
- `bbox_analysis.py`: Count the number of **out-of-boundary** object bounding boxes and compute pairwise bounding boxes **IoU** (this requires sampled floor plan boundary and normal points).

Most evaluation scripts can be run by simply providing the path to the pickled result file. For image comparisons, make sure you set the `--no_texture` flag when comparing textureless synthetic layouts with the ground-truth.

## Code organization
The dependency relationships in `threed_front/datasets`: 
```
__init__.py
├── base.py
├── common.py
├── splits_builder.py
├── threed_front.py
│   └── parse_utils.py
│       └── threed_front_scene.py
└── threed_future_dataset.py
threed_front_encoding_base.py
```
The `__init__.py` contains all dataset classes/functions to read and host preprocessed scene data, 
including `CachedThreedFront` which loads data from `output/3d_front_processed/<room_type>`. 
The `threed_front_encoding_base.py` stores other feature encoding and dataset augmentation classes for training.
