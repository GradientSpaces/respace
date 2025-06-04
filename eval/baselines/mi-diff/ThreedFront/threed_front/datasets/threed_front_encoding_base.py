# 
# modified from: 
#   https://github.com/nv-tlabs/ATISS.
#

import numpy as np
import math

from functools import lru_cache
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset
from .threed_front_scene import rotation_matrix_around_y


class DatasetDecoratorBase(Dataset):
    """A base class that helps us implement decorators for ThreedFront-like
    datasets."""
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):
        return self.bbox_dims + self.n_classes

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims
    
    @property
    def contain_edges(self):
        if hasattr(self._dataset, "contain_edges"):
            return self._dataset.contain_edges
        else:
            return False

    # compute max_length for diffusion models
    @property
    def max_length(self):
        return self._dataset.max_length 
    
    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxOrderedDataset(DatasetDecoratorBase):
    """A base class that helps implement feature encoding."""
    def __init__(self, dataset, box_ordering=None):
        super().__init__(dataset)
        self._box_ordering = box_ordering

    @lru_cache(maxsize=16)
    def _get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        if self._box_ordering is None:
            return scene.bboxes
        elif self._box_ordering == "class_frequencies":
            return scene.ordered_bboxes_with_class_frequencies(
                self.class_frequencies)
        else:
            raise NotImplementedError()


class RoomLayoutEncoder(DatasetDecoratorBase):
    """Implement the encoding for the room layout as images."""
    @property
    def property_type(self):
        return "room_layout"

    def __getitem__(self, idx):
        img = self._dataset[idx].room_mask[:, :, 0:1]
        return np.transpose(img, (2, 0, 1))

    @property
    def bbox_dims(self):
        return 0


class ClassLabelsEncoder(BoxOrderedDataset):
    """Implement the encoding for the class labels."""
    @property
    def property_type(self):
        return "class_labels"

    def __getitem__(self, idx):
        # Make a local copy of the class labels
        classes = self.class_labels

        # Get the scene
        boxes = self._get_boxes(idx)
        class_labels = np.array(
            [bs.one_hot_label(classes) for bs in boxes], dtype=np.float32
        )
        return class_labels

    @property
    def bbox_dims(self):
        return 0


class TranslationEncoder(BoxOrderedDataset):
    """Implement the encoding for translations (i.e. object centroid positions)."""
    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        # Get the scene
        scene = self._dataset[idx]
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        translations = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            translations[i] = bs.centroid(-scene.centroid)
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncoder(BoxOrderedDataset):
    """Implement the encoding for object boundng box sizes."""
    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes

    @property
    def bbox_dims(self):
        return 3


class ObjFeatEncoder(BoxOrderedDataset):
    """Implement the encoding for 64-dimentional object shape features."""
    @property
    def property_type(self):
        return "objfeats"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        latents = np.stack([bs.raw_model_norm_pc_lat() for bs in boxes], axis=0)
        return latents

    @property
    def bbox_dims(self):
        return 64


class ObjFeat32Encoder(BoxOrderedDataset):
    """Implement the encoding for 32-dimentional object shape features."""
    @property
    def property_type(self):
        return "objfeats_32"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        latents = np.stack([bs.raw_model_norm_pc_lat32() for bs in boxes], axis=0)
        return latents

    @property
    def bbox_dims(self):
        return 32


class AngleEncoder(BoxOrderedDataset):
    """Implement the encoding for object vertical rotation angles."""
    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        angles = np.zeros((L, 1), dtype=np.float32)
        for i, bs in enumerate(boxes):
            angles[i] = bs.z_angle
        return angles

    @property
    def bbox_dims(self):
        return 1


class DatasetCollection(DatasetDecoratorBase):
    """Dataset class to combine multiple datasest (containing 'bbox_dims' and 
    'property_type' properties). The __getitem__ function combines outputs from all 
    datasets into a dictionary and collate_fn combines a list of these outputs."""
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

    @staticmethod
    def collate_fn(samples):
        # We assume that all samples have the same set of keys
        key_set = set(samples[0].keys()) - set(["length"])

        # Compute the max length of the sequences in the batch
        max_length = max(sample["length"] for sample in samples)

        # Assume that all inputs that are 3D or 1D do not need padding.
        # Otherwise, pad the first dimension.
        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)
        sample_params = {}
        sample_params.update({
            k: np.stack([sample[k] for sample in samples], axis=0)
            for k in (key_set - padding_keys)
        })

        sample_params.update({k: np.stack(
            [np.vstack([sample[k], np.zeros((max_length-len(sample[k]), sample[k].shape[1]))]
                       ) for sample in samples], axis=0)
            for k in padding_keys})
        sample_params["lengths"] = np.array([
            sample["length"] for sample in samples
        ])
        
        # Make torch tensors from the numpy tensors
        torch_sample = {
            k: torch.from_numpy(sample_params[k]).float() for k in sample_params
        }

        torch_sample.update({
            k: torch_sample[k][:, None] for k in torch_sample.keys() if "_tr" in k
        })
        
        return torch_sample


class CachedDatasetCollection(DatasetCollection):
    """Dataset class inheritated from DatasetCollection. This class is initizlied with 
    a dataset containing get_room_params class function such that __getitem__ returns 
    the output of this function."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class RotationAugmentation(DatasetDecoratorBase):
    """Class to add a random rotation to all object angles."""
    def __init__(self, dataset, min_rad=-math.pi, max_rad=math.pi, fixed=False):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad
        self._fixed   = fixed

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0
    
    @property
    def fixed_rot_angle(self):
        if np.random.rand() < 0.25:
            return np.pi * 1.5
        elif np.random.rand() < 0.50:
            return np.pi
        elif np.random.rand() < 0.75:
            return np.pi * 0.5
        else:
            return 0.0

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        
        # Get the rotation matrix for the current scene
        if self._fixed:
            rot_angle = self.fixed_rot_angle
        else:
            rot_angle = self.rot_angle
        
        if rot_angle == 0:
            return sample_params
        else:
            R = rotation_matrix_around_y(rot_angle).astype(np.float32)
            R_2d = R[:, [0,2]][[0,2], :]    # this is transpose of a 2d rotation matrix
        
        for k, v in sample_params.items():
            if k == "translations":
                sample_params[k] = v @ R.T    # equivalent to (R @ v.T).T 
            elif k == "angles":
                sample_params[k] = \
                    (v - rot_angle + np.pi) % (2 * np.pi) - np.pi
            elif k == "fpbpn":
                sample_params[k][:, :2] = v[:, :2] @ R_2d.T
                sample_params[k][:, 2:] = v[:, 2:] @ R_2d.T
            elif k == "room_layout":
                # Fix the ordering of the channels because it was previously
                # changed
                img = np.transpose(v, (1, 2, 0))
                sample_params[k] = np.transpose(rotate(
                    img, rot_angle * 180 / np.pi, reshape=False
                ), (2, 0, 1))
        return sample_params


class Jitter(DatasetDecoratorBase):
    """Class to jitter translations, sizes, and angles."""
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                sample_params[k] = v + np.random.normal(0, 0.01)
            else:
                sample_params[k] = v
        return sample_params


class Scale(DatasetDecoratorBase):
    """Class to bound all features except objfeats in self.bounds to -1 to +1."""
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in bounds and k not in ["objfeats", "objfeats_32"]:
                sample_params[k] = Scale.scale(v, bounds[k][0], bounds[k][1])
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k in bounds and k not in ["objfeats", "objfeats_32"]:
                sample_params[k] = Scale.descale(v, bounds[k][0], bounds[k][1])
            else:
                sample_params[k] = v
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Scale_CosinAngle(DatasetDecoratorBase):
    """Class to use [cos, sin] representation for angles, 
    bound all other features except objfeats in self.bounds to -1 to +1."""
    @staticmethod
    def scale(x, minimum, maximum):
        return Scale.scale(x, minimum, maximum)

    @staticmethod
    def descale(x, minimum, maximum):
        return Scale.descale(x, minimum, maximum)

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                # [cos, sin]
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)
            elif k in bounds and k not in ["objfeats", "objfeats_32"]:
                sample_params[k] = Scale.scale(v, bounds[k][0], bounds[k][1])
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "angles":
                # theta = arctan2(sin, cos)
                sample_params[k] = np.arctan2(v[:, :, 1:2], v[:, :, 0:1])
            elif k in bounds and k not in ["objfeats", "objfeats_32"]:
                sample_params[k] = Scale.descale(v, bounds[k][0], bounds[k][1])
            else:
                sample_params[k] = v
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2


class Scale_CosinAngle_ObjfeatsNorm(DatasetDecoratorBase):
    """Class to use [cos, sin] representation for angles, 
    bound all other features in self.bounds to -1 to +1."""
    @staticmethod
    def scale(x, minimum, maximum):
        return Scale.scale(x, minimum, maximum)

    @staticmethod
    def descale(x, minimum, maximum):
        return Scale.descale(x, minimum, maximum)

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                # [cos, sin]
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)
            elif k in bounds:
                sample_params[k] = Scale.scale(v, bounds[k][0], bounds[k][1])
        return sample_params

    def post_process(self, s):
        bounds = self.bounds
        sample_params = {}
        for k, v in s.items():
            if k == "angles":
                # theta = arctan2(sin, cos)
                sample_params[k] = np.arctan2(v[:, :, 1:2], v[:, :, 0:1])
            elif k in bounds:
                sample_params[k] = Scale.descale(v, bounds[k][0], bounds[k][1])
            else:
                sample_params[k] = v
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2


class Permutation(DatasetDecoratorBase):
    """Class to permute object ordering in the scene."""
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys
        self._permutation_axis = permutation_axis

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["class_labels"].shape
        ordering = np.random.permutation(shapes[self._permutation_axis])

        for k in self._permutation_keys:
            if k in sample_params:
                sample_params[k] = sample_params[k][ordering]
        
        if self.contain_edges:
            ordering_map = np.arange(len(ordering))[np.argsort(ordering)]
            sample_params["edge_index"] = ordering_map[sample_params["edge_index"]]
            sample_params["adj_matrix"] = \
                sample_params["adj_matrix"][ordering, :][:, ordering]

        return sample_params


class OrderedDataset(DatasetDecoratorBase):
    """Dataset class to re-order a list of selected values based on specified ordering.
    The __getitem__ function of the input dataset class has to return a dictionary."""
    def __init__(self, dataset, ordered_keys, box_ordering):
        super().__init__(dataset)
        self._ordered_keys = ordered_keys
        self._box_ordering = box_ordering

    def __getitem__(self, idx):
        if self._box_ordering == "class_frequencies":
            sample = self._dataset[idx]
            order = self._get_class_frequency_order(sample)
            for k in self._ordered_keys:
                sample[k] = sample[k][order]
            return sample
        else:
            raise NotImplementedError()

    def _get_class_frequency_order(self, sample):
        t = sample["translations"]
        c = sample["class_labels"].argmax(-1)
        class_frequencies = self.class_frequencies
        class_labels = self.class_labels
        f = np.array([[class_frequencies[class_labels[ci]]] for ci in c])
        return np.lexsort(np.hstack([t, f]).T)[::-1]


def get_basic_encoding(dataset, box_ordering=None, add_objfeats=False):
    """Return basic encoding combining ClassLabelsEncoder, TranslationEncoder, 
    SizeEncoder, and AngleEncoder. The __getitem__ member function of the 
    output class returns a dictionary of {property_type: encoding}. """
    box_ordered_dataset = BoxOrderedDataset(dataset, box_ordering)

    class_labels = ClassLabelsEncoder(box_ordered_dataset)
    translations = TranslationEncoder(box_ordered_dataset)
    sizes = SizeEncoder(box_ordered_dataset)
    angles = AngleEncoder(box_ordered_dataset)
    feat_encoders = [class_labels, translations, sizes, angles]

    if add_objfeats:
        if dataset.objfeats is not None:
            feat_encoders.append(ObjFeatEncoder(box_ordered_dataset))
        if dataset.objfeats_32 is not None:
            feat_encoders.append(ObjFeat32Encoder(box_ordered_dataset))

    return DatasetCollection(*feat_encoders)
