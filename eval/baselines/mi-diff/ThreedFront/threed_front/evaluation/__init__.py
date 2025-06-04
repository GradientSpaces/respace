import os
from .utils import evaluate_kl_divergence, render_projection_from_layout
from threed_front.datasets import ThreedFront, ThreedFutureDataset


class ThreedFrontResults():
    """Class to save predicted scenes for evaluation.
    Each result has a scene_idx (stored in scene_indices) which is the 
    index of the floor plan in test_dataset, and a layout dictionary 
    {propery: property_matrix} (store in predicted_layouts) for each property 
    (class_label, translations, sizes, angles) and corresponding Nx? property 
    matrix."""
    def __init__(self, train_dataset, test_dataset, config, 
                 scene_indices=[], predicted_layouts=[]):
        # config dict
        self._config = config
        ThreedFrontResults.update_file_paths(config["data"])

        # train dataset
        self._train_dataset = train_dataset
        while not isinstance(self._train_dataset, ThreedFront):
            self._train_dataset = self._train_dataset._dataset
        
        # test dataset
        self._test_dataset = test_dataset
        while not isinstance(self._test_dataset, ThreedFront):
            self._test_dataset = self._test_dataset._dataset
        
        # results - scene_indices stores corresponding room indices in test_dataset
        assert len(scene_indices) == len(predicted_layouts)
        self._scene_indices = scene_indices
        self._predicted_layouts = predicted_layouts
    
    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def n_object_types(self):
        return self._test_dataset.n_object_types

    @property
    def config(self):
        return self._config
    
    @property
    def floor_condition(self):
        return self._config["network"].get("room_mask_condition", True)

    def __getitem__(self, idx):
        return self._scene_indices[idx], self._predicted_layouts[idx]

    def __len__(self):
        return len(self._scene_indices)    
    
    def __str__(self):
        return f"ThreedFrontResults containing {len(self._test_dataset)} real" \
            f" and {len(self._scene_indices)} synthetic scenes."
    
    @staticmethod
    def update_file_paths(data_dict):
        for k, v in data_dict.items():
            if isinstance(v, str) and os.path.exists(v):
                data_dict[k] = os.path.realpath(v)

    def add_result(self, scene_idx, bbox_params):
        self._scene_indices.append(scene_idx)
        self._predicted_layouts.append(bbox_params)

    def kl_divergence(self):
        kl_divergence, *_ = \
        evaluate_kl_divergence(
            self._test_dataset, self._scene_indices, self._predicted_layouts
        )
        return kl_divergence
    
    def evaluate_class_labels(self):
        return evaluate_kl_divergence(
            self._test_dataset, self._scene_indices, self._predicted_layouts
        )
    
    def render_projection(
            self, idx, objects_dataset:ThreedFutureDataset, output_path, 
            scene_viz, floor_texture=None, floor_color=None, retrieve_mode=None,
            color_palette=None, rotate=None
        ):
        scene_idx = self._scene_indices[idx]
        if retrieve_mode is None:
            retrieve_mode = "object" if \
                "objfeats" in self._predicted_layouts[idx].keys() else "size"
        if rotate is not None:
            self._predicted_layouts[idx]["angles"] += rotate
        
        render_projection_from_layout(
            self._test_dataset[scene_idx], self._predicted_layouts[idx], 
            objects_dataset, output_path, self._test_dataset.object_types, 
            scene_viz, floor_texture, floor_color, retrieve_mode=retrieve_mode,
            color_palette=color_palette
        )
