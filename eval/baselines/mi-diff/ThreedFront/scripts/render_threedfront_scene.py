# 
# Modified from: 
#   https://github.com/nv-tlabs/ATISS.
#
"""Script used for visualizing 3D-FRONT room specified by its scene_id."""
import argparse
import os
import sys
import pickle
import numpy as np

from simple_3dviz.utils import render
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveGif

from threed_front.simple_3dviz_setup import SIDEVIEW_SCENE
from threed_front.rendering import get_floor_plan, export_scene
from utils import PROJ_DIR, PATH_TO_PICKLED_3D_FRONT_DATASET, PATH_TO_FLOOR_PLAN_TEXTURES, adjust_textured_mesh


def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize a 3D-FRONT room from json file"
    )
    parser.add_argument(
        "scene_id",
        help="The scene id of the scene to be visualized"
    )
    parser.add_argument(
        "--path_to_pickled_3d_front_dataset",
        default=PATH_TO_PICKLED_3D_FRONT_DATASET,
        help="Path to pickled 3D-FRONT dataset (default: output/threed_front.pkl)"
    )
    parser.add_argument(
        "--output_directory",
        default=PROJ_DIR + "/output/scene/",
        help="Path to output directory if needed (default: output/scene/)"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering and save result gif to output directory"
    )
    parser.add_argument(
        "--floor_color",
        type=lambda x: tuple(map(float, x.split(","))) if x!= None else None,
        default=None,
        help="Set floor color of generated images, e.g. 0.8,0.8,0.8 "
        "(Note: this overrides path_to_floor_plan_textures)"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default=PATH_TO_FLOOR_PLAN_TEXTURES,
        help="Path to floor texture image directory or a single image file "
        "(default: demo/floor_plan_texture_images)"
    )
    parser.add_argument(
        "--without_texture",
        action="store_true",
        help="Visualize without texture "
        "(object color: (0.5,0.5,0.5), floor color: (0.8,0.8,0.8) or specified)"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="Visualize without the room's floor"
    )
    parser.add_argument(
        "--with_walls",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_door_and_windows",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--export_mesh",
        action="store_true",
        help="Export scene to output_directory/<scene_id> using trimesh "
        "(trimesh rendering style is only affected by floor_color and path_to_floor_plan_textures)"
    )

    args = parser.parse_args(argv)

    # Load dataset
    scenes = pickle.load(open(args.path_to_pickled_3d_front_dataset, "rb"))
    print("Loading full dataset with {} rooms".format(len(scenes)))

    # Find scene in dataset with specified id
    scene_in_dataset = [ss for ss in scenes if ss.scene_id == args.scene_id]
    if len(scene_in_dataset) == 0:
        print("Cannot find scene id:", args.scene_id)
        return
    elif len(scene_in_dataset) > 1:
        print("Multiple scenes found with id:", args.scene_id)
        return
    else:
        ss = scene_in_dataset[0]
        print(ss.json_path, ss.furniture_in_room)

    # Build renderables
    renderables = ss.furniture_renderables(
        with_floor_plan_offset=True, with_texture=not args.without_texture)
    for r in renderables:
        adjust_textured_mesh(r)
    
    if not args.without_floor:
        # use a single floor color for rendering without texture
        if args.without_texture:    
            floor_texture = None
            if args.floor_color is None:
                args.floor_color = (0.8, 0.8, 0.8, 1.0)
        # use input floor color if available
        elif args.floor_color:
            floor_texture = None
        # use floor texture files otherwise
        else:
            if os.path.isdir(args.path_to_floor_plan_textures):
                floor_textures = \
                    [os.path.join(args.path_to_floor_plan_textures, fi)
                        for fi in os.listdir(args.path_to_floor_plan_textures)]
                floor_texture = np.random.choice(floor_textures)
            else:
                floor_texture = args.path_to_floor_plan_textures            
        
        floor_plan, _, _ = get_floor_plan(
            ss, floor_texture, args.floor_color, with_room_mask=False
        )
        renderables.append(floor_plan)

    if args.with_walls:
        for ei in ss.extras:
            if "WallInner" in ei.model_type:
                renderables.append(
                    ei.mesh_renderable(
                        offset=-ss.centroid, colors=(0.8, 0.8, 0.8, 0.6)
                    )
                )

    if args.with_door_and_windows:
        for ei in ss.extras:
            if "Window" in ei.model_type or "Door" in ei.model_type:
                renderables.append(
                    ei.mesh_renderable(
                        offset=-ss.centroid, colors=(0.8, 0.8, 0.8, 0.6)
                    )
                )

    # Visualize scene
    if args.without_screen:
        os.makedirs(args.output_directory, exist_ok=True)
        path_to_gif = "{}/{}.gif".format(
            args.output_directory, 
            ss.scene_id + "_notexture" if args.without_texture else ss.scene_id
        )
        behaviours = [
            LightToCamera(),
            CameraTrajectory(Circle(
                [0, SIDEVIEW_SCENE["camera_position"][1], 0],
                SIDEVIEW_SCENE["camera_position"],
                SIDEVIEW_SCENE["up_vector"]
            ), speed=1/360),
            SaveGif(path_to_gif, 2, duration=32)
        ]
        render(renderables, behaviours, 360, **SIDEVIEW_SCENE)
        print("Saved scene to {}.".format(path_to_gif))
    else:
        behaviours = [LightToCamera(), SnapshotOnKey()]
        show(
            renderables, behaviours=behaviours, **SIDEVIEW_SCENE
        )

    # Create a trimesh scene and export it
    if args.export_mesh:
        path_to_objs = os.path.join(args.output_directory, args.scene_id)
        os.makedirs(path_to_objs, exist_ok=True)

        trimesh_meshes = ss.furniture_meshes()
        _, tr_floor, _ = get_floor_plan(
            ss, floor_texture, args.floor_color, with_room_mask=False,
            with_trimesh=True
        )
        trimesh_meshes.append(tr_floor)
        export_scene(path_to_objs, trimesh_meshes)
        print("Saved meshes to {}.".format(path_to_objs))


if __name__ == "__main__":
    main(sys.argv[1:])
