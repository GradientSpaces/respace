# 
# Licensed under the NVIDIA Source Code License.
# Modified from https://github.com/nv-tlabs/ATISS.
# 

import os
from typing import Union, Tuple, List
import numpy as np
import torch
from PIL import Image
import trimesh
from pyrr import Matrix44

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.renderables import Renderable, Lines, Spherecloud
from simple_3dviz.utils import save_frame

from threed_front.datasets.threed_front_scene import Room
from threed_front.datasets.threed_front import CachedRoom
from threed_front.datasets.threed_front_scene import rotation_matrix_around_y


""" Helper functions """

def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.get("window_size", (256, 256)), background=args.get("background", (1, 1, 1, 1)))
    scene.up_vector = args.get("up_vector")
    scene.camera_target = args.get("camera_target")
    scene.camera_position = args.get("camera_position")
    scene.light = args.get("camera_position")

    # martin: custom, instead of far_plane=6
    far_plane = 2.0 * args.get("camera_position")[1]

    if "room_side" in args: 
        scene.camera_matrix = Matrix44.orthogonal_projection(
            left=-args.get("room_side"), right=args.get("room_side"),
            bottom=args.get("room_side"), top=-args.get("room_side"),
            near=0.1, far=far_plane)
    return scene


""" Floor plan rednerable """

def get_floor_plan(scene: Union[Room, CachedRoom], texture=None, 
                   color=(0.87, 0.72, 0.53), with_trimesh=False,
                   with_room_mask=False) \
    -> Tuple[TexturedMesh, trimesh.Trimesh, np.ndarray]:
    """Return the floor plan of the scene as a simple-3dviz TexturedMesh, a trimesh mesh, 
    and an optional binary numpy array."""
    vertices, faces = scene.floor_plan
    vertices -= scene.floor_plan_centroid

    if texture is not None:
        uv = np.copy(vertices[:, [0, 2]])
        uv -= uv.min(axis=0)
        uv /= 0.3  # repeat every 30cm
        floor = TexturedMesh.from_faces(
            vertices=np.copy(vertices), uv=np.copy(uv), faces=np.copy(faces),
            material=Material.with_texture_image(texture)
        )
    else:
        floor = Mesh.from_faces(
            vertices=np.copy(vertices), faces=np.copy(faces), colors=color
        )

    if with_trimesh:
        tr_floor = trimesh.Trimesh(
            vertices=np.copy(vertices), faces=np.copy(faces), process=False
        )
        if texture is not None:
            tr_floor.visual = trimesh.visual.TextureVisuals(
                uv=np.copy(uv), image=Image.open(texture)
            )
        else:
            tr_floor.visual.face_colors = np.tile(color, (faces.shape[0], 1))
    else:
        tr_floor = None
    
    if with_room_mask:
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2)))
    else:
        room_mask = None
    
    return floor, tr_floor, room_mask


""" Furniture rednerable """

def get_bbox_points(centroid, size, angle) -> np.ndarray:
    """Return a set of bounding box segments as a 24 by 3 numpy array."""
    R = rotation_matrix_around_y(angle)
    l_x, l_y, l_z = -size / 2
    u_x, u_y, u_z = size / 2
    bbox_points = np.array([
        (l_x, l_y, l_z), (u_x, l_y, l_z), (u_x, l_y, l_z), (u_x, u_y, l_z),
        (u_x, u_y, l_z), (l_x, u_y, l_z), (l_x, u_y, l_z), (l_x, l_y, l_z),
        (l_x, l_y, u_z), (u_x, l_y, u_z), (u_x, l_y, u_z), (u_x, u_y, u_z),
        (u_x, u_y, u_z), (l_x, u_y, u_z), (l_x, u_y, u_z), (l_x, l_y, u_z),
        (l_x, l_y, l_z), (l_x, l_y, u_z), (u_x, l_y, l_z), (u_x, l_y, u_z),
        (u_x, u_y, l_z), (u_x, u_y, u_z), (l_x, u_y, l_z), (l_x, u_y, u_z)
    ])
    return bbox_points @ R.T + centroid


def get_textured_objects_in_scene(scene: Room, colors=None, with_bbox=False, 
                                  box_color=(0.0, 1, 0.4, 1.0), width=0.05) \
    -> List[Renderable]:
    """Return the objects in a scene as a list of simple-3dviz TexturedMesh with an 
    option to add bounding box lines. 
    If "colors" is given, furniture texture will be replaced by the input list of colors."""
    if colors is not None:
        assert len(colors) == len(scene.bboxes)

    renderables = []
    bbox_renderables = []
    for i, furniture in enumerate(scene.bboxes):
        # Load the furniture and scale it as it is given in the dataset
        if colors is None:
            try:
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
            # martin: add this to catch exceptions from assets and just ignore object for rendering
            except Exception as exc:
                print(f"Failed to load TexturedMesh for {furniture.raw_model_path}: {exc}")
                continue  # Skip this furniture item
        else:
            try:
                raw_mesh = Mesh.from_file(furniture.raw_model_path, color=colors[i])
            # martin: add this to catch exceptions from assets
            except Exception as exc:
                print(f"Failed to load Mesh with color, trying TexturedMesh: {exc}")
                try:
                    raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                except Exception as exc2:
                    print(f"Also failed to load TexturedMesh: {exc2}")
                    continue  # Skip this furniture item
        
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
        R = rotation_matrix_around_y(theta)

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R.T, t=translation)
        renderables.append(raw_mesh)

        if with_bbox:
            # Get bounding box segments
            bbox_points = get_bbox_points(translation, bbox[1] - bbox[0], theta)
            bbox_renderables.append(Lines(bbox_points, colors=box_color, width=width))
    
    return renderables + bbox_renderables


def get_textured_objects(
        bbox_params, objects_dataset, classes, retrieve_mode="size", 
        color_palette=None, with_bbox=False, box_color=(0.0, 1.0, 0.4, 1.0), 
        width=0.05, with_trimesh=True
    ) -> Tuple[List[TexturedMesh], List[trimesh.Trimesh]]:
    """Return the predicted objects as a list of simple-3dviz TexturedMesh, 
    and a list of trimesh mesh."""
    # For each one of the boxes replace them with an object
    renderables = []
    trimesh_meshes = []
    bbox_renderables = []
    for j in range(bbox_params["class_labels"].shape[0]):
        # Extract prediction of object j
        class_index = bbox_params["class_labels"][j].argmax(-1)
        query_label = classes[class_index]
        translation = bbox_params["translations"][j]
        query_size = bbox_params["sizes"][j]
        theta = bbox_params["angles"][j, 0]
        R = rotation_matrix_around_y(theta)
        
        if query_label in ["start", "end"]:
            continue

        # Retrieve 3D-FUTURE model
        if retrieve_mode == "size":
            furniture = objects_dataset.get_closest_furniture_to_box(
                query_label, query_size)
        elif retrieve_mode == "objfeat":
            query_objfeat = bbox_params["objfeats"][j]
            furniture = objects_dataset.get_closest_furniture_to_objfeats(
                query_label, query_objfeat
            )
        else:
            return NotImplemented

        # Load the furniture
        if color_palette is None:
            raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
        else:
            raw_mesh = Mesh.from_file(
                furniture.raw_model_path, color=color_palette[class_index]
            )
        # Scale it 
        # as it is given in the dataset
        if retrieve_mode == "size":
            size_scale = furniture.scale
        # by predicted size
        elif retrieve_mode == "objfeat":
            raw_bbox_vertices = \
                np.load(furniture.path_to_bbox_vertices, mmap_mode="r")
            raw_sizes = np.array([  # Note: ThreedFutureModel implements size as half distance between vertices
                np.linalg.norm(raw_bbox_vertices[4] - raw_bbox_vertices[0]) / 2,
                np.linalg.norm(raw_bbox_vertices[2] - raw_bbox_vertices[0]) / 2,
                np.linalg.norm(raw_bbox_vertices[1] - raw_bbox_vertices[0]) / 2
            ])
            size_scale = query_size / raw_sizes
        else:
            return NotImplemented
        raw_mesh.scale(size_scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1]) / 2

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R.T, t=translation)
        renderables.append(raw_mesh)

        if with_bbox:
            # Get bounding box segments
            bbox_points = get_bbox_points(translation, bbox[1] - bbox[0], theta)
            bbox_renderables.append(Lines(bbox_points, colors=box_color, width=width))
        
        if with_trimesh:
            # Create a trimesh object for the same mesh in order to save
            # everything as a single scene
            tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
            if color_palette is None:
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
            else:
                color = color_palette[class_index]
                tr_mesh.visual.face_colors = \
                    (color[None, :].repeat(tr_mesh.faces.shape[0], axis=0) \
                     .reshape(-1, 3) * 255).astype(np.uint8)
                tr_mesh.visual.vertex_colors = \
                    (color[None, :].repeat(tr_mesh.vertices.shape[0], axis=0) \
                     .reshape(-1, 3) * 255).astype(np.uint8)
            tr_mesh.vertices *= size_scale
            tr_mesh.vertices -= centroid
            tr_mesh.vertices[...] = tr_mesh.vertices.dot(R.T) + translation
            trimesh_meshes.append(tr_mesh)

    if with_trimesh:
        return renderables + bbox_renderables, trimesh_meshes
    else:
        return renderables + bbox_renderables, None


def get_edge_renderables(centroids: Union[torch.Tensor, np.ndarray], 
                         edge_index_list: List[np.ndarray], 
                         line_colors=(0.0, 1, 0.4, 1.0), line_widths=0.05, 
                         marker_color=(0.0, 1, 0.4, 1.0), marker_size=0.2):
    
    if isinstance(line_colors[0], float) or isinstance(line_colors[0], int):
        line_colors = [line_colors] * len(edge_index_list)

    if isinstance(line_widths, float) or isinstance(line_widths, int):
        line_widths = [line_widths] * len(edge_index_list)
    
    edge_lines = [
        Lines(
            centroids[edge_index.transpose().reshape(-1)], 
            colors=color, width=width
        ) for edge_index, color, width in 
        zip(edge_index_list, line_colors, line_widths) if edge_index.size != 0
    ]
    centers = Spherecloud(centroids, colors=marker_color, sizes=marker_size)

    return edge_lines + [centers]


""" Rendering """

def render_projection(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        if isinstance(r, TexturedMesh) and r.material.ambient.ndim == 2:
            # take average if the .mtl file provides more than one vector
            r.material.ambient = r.material.ambient.mean(0)
            r.material.diffuse = r.material.diffuse.mean(0)
            r.material.specular = r.material.specular.mean(0)
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def export_scene(output_directory, trimesh_meshes, names=None):
    # Export each object
    if names is None:
        names = ["object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))]
    mtl_names = ["material_{:03d}".format(i) for i in range(len(trimesh_meshes))]

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(m, return_texture=True)

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(tex_out[mtl_key].replace(b"material0", mtl_names[i].encode("ascii")))
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])

    # Export scene (scene.obj, material.mtl, material_0.png)
    trimesh_combined = trimesh.util.concatenate(trimesh_meshes)
    trimesh_combined.export(os.path.join(output_directory, "scene.obj"))
