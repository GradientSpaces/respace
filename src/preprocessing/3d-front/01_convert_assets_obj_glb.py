import bpy
import distutils.dir_util
import os
from tqdm import tqdm
from pathlib import Path
import sys
from contextlib import contextmanager

# distutils.dir_util.copy_tree(obj_id, tgt_pth)
# exit()

def get_folder_names(directory):
	path = Path(directory)
	folder_names = [f.name for f in path.iterdir() if f.is_dir()]
	return folder_names


def clear_blender_data():
	# Deselect all
	bpy.ops.object.select_all(action='DESELECT')
	# Delete all objects
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete(use_global=False)

	# Clear orphan data
	for block in bpy.data.meshes:
		if block.users == 0:
			bpy.data.meshes.remove(block)
	for block in bpy.data.materials:
		if block.users == 0:
			bpy.data.materials.remove(block)
	for block in bpy.data.images:
		if block.users == 0:
			bpy.data.images.remove(block)


def convert_object(pth_src_file, pth_texture):

	try:
		clear_blender_data()

		bpy.ops.wm.obj_import(filepath=pth_src_file, use_split_objects=False, use_split_groups=False)
		bpy.ops.object.select_all(action='SELECT')

		# Apply transformations and set material shading to use nodes (important for exports)
		for obj in bpy.context.selected_objects:
			bpy.context.view_layer.objects.active = obj
			bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
			
			if obj.type == 'MESH':

				# Cleanup operations with more detailed logging
				bpy.ops.object.mode_set(mode='EDIT')
				bpy.ops.mesh.select_all(action='SELECT')
				
				# Flip normals if needed
				bpy.ops.mesh.normals_make_consistent(inside=False)
				
				# Remove doubles (only if necessary)
				# Set a threshold to avoid over-aggressive removal
				# bpy.ops.mesh.remove_doubles(threshold=0.0001)
				
				bpy.ops.object.mode_set(mode='OBJECT')
		
				if not obj.data.materials:
					mat = bpy.data.materials.new(name="DefaultMaterial")
					obj.data.materials.append(mat)

				# Ensure each material uses nodes and apply texture
				for mat in obj.data.materials:
					mat.use_nodes = True
					nodes = mat.node_tree.nodes
					nodes.clear()
					
					# Create a Principled BSDF Shader and Texture Node
					node_texture = nodes.new(type='ShaderNodeTexImage')
					node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
					node_output = nodes.new('ShaderNodeOutputMaterial')
					
					# Load texture image
					if not bpy.data.images.get(os.path.basename(pth_texture)):
						img = bpy.data.images.load(pth_texture)
					else:
						img = bpy.data.images[os.path.basename(pth_texture)]
					node_texture.image = img
					
					# Link texture node to BSDF
					node_tree = mat.node_tree
					node_tree.links.new(node_texture.outputs['Color'], node_bsdf.inputs['Base Color'])
					
					# Link BSDF to Material Output
					node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

		# Export to GLB format
		bpy.ops.export_scene.gltf(filepath=pth_src_file.replace(".obj", ".glb"), export_format='GLB', export_materials='EXPORT')

	except Exception as exc:
		print(exc)

	finally:
		bpy.ops.wm.read_factory_settings(use_empty=True)


obj_ids = [ 
	# "/Users/mnbucher/Downloads/3D-FUTURE-model-part1/3996590b-84d5-4d8a-bf02-29b81cf02a3a" 
	# "0e743639-2b66-49e3-bb44-6e9fe1a7960f",
	"3f012ebe-d552-40be-b18a-b9450b107996",
	"7f414777-fced-4e4f-b770-22a74a75a137",
	"85813056-b5ba-47e8-a940-9c3540ea9cb1",
]

pth_root = "/Volumes/apollo11/data/3D-FUTURE-assets/"
obj_ids = get_folder_names(pth_root)

# print(len(obj_ids))
# exit()

for obj_id in tqdm(obj_ids):

	pth_tgt = pth_root + obj_id

	# tgt_pth = "/Users/mnbucher/git/stan-24-sgllm/src/frontend/sgllm-frontend-vite/public/sample-obj-v3"
	pth_src_file = pth_tgt + "/raw_model.obj"
	pth_texture = pth_tgt + '/texture.png'

	convert_object(pth_src_file, pth_texture)

	# break

	# exit()
