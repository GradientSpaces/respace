import os
import pickle

from threed_front.datasets.threed_front import ThreedFront


PROJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Default input directories
PATH_TO_DATASET_FILES = os.path.join(PROJ_DIR, "dataset_files")
PATH_TO_FLOOR_PLAN_TEXTURES = os.path.join(PROJ_DIR, "demo/floor_plan_texture_images")

# Default parsed/preprocessed data directories (empty string is to be filed with room_type)
PATH_TO_PICKLED_3D_FRONT_DATASET = os.path.join(PROJ_DIR, "output/threed_front.pkl")
PATH_TO_PICKLED_3D_FUTURE_MODEL = os.path.join(PROJ_DIR, "output/threed_future_model_{}.pkl")
PATH_TO_PROCESSED_DATA = os.path.join(PROJ_DIR, "output/3d_front_processed/{}")


def load_pickled_threed_front(file_path, filter_fn=lambda s: s):
    """Load pickled treed-front data to a ThreedFront object"""
    scenes = pickle.load(open(file_path, "rb"))
    threed_front_dataset = ThreedFront([s for s in map(filter_fn, scenes) if s])
    return threed_front_dataset


def create_or_clear_output_dir(output_dir):
    """Clean up output directory, or create an empty directory if does not exist."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        for fi in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, fi))
        print("Removed all files in {}.".format(output_dir))
    else:
        os.makedirs(output_dir, exist_ok=True)


def update_render_paths(dataset, new_base_dir=None, rendered_scene=None):
    """Swich dataset base directory or rendered scene layout name."""
    if new_base_dir is not None:
        dataset._base_dir = new_base_dir
    if rendered_scene is not None:
        dataset._rendered_name = rendered_scene
    assert os.path.exists(dataset._path_to_render(0)), \
        "Path does not exist: {}".format(dataset._path_to_render(0))


def adjust_textured_mesh(renderable, min=0.25, max=0.75):
    """Adjust texture of the mesh to be in the range of [min, max]."""
    renderable.material.ambient = (renderable.material.ambient - 0.5) * (max - min) + 0.5
    renderable.material.diffuse = (renderable.material.diffuse - 0.5) * (max - min) + 0.5
