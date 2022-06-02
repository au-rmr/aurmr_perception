import argparse
import os
import numpy as np
import random
import trimesh
import yaml
from collections import defaultdict
from munch import DefaultMunch
import nvisii as visii
import cv2
import h5py
from scripts.visii_renderer import render_scene
from skimage.color import label2rgb
import json

from visii_tools.loader.dict_loader import DictLoader
from visii_tools.scene import scene
from visii_tools.dataset import h5_utils
from visii_tools.render import render_utils
import visii_tools
from visii_utils import create_floor, sample_camera_position, set_support_materials, set_assigned_materials, create_bin

TABLE_DIMS = [0.3 , 0.25, 0.1 ]

manager = trimesh.collision.CollisionManager()

ORIENTATIONS = {
    'FRONT': 0,
    'BACK': 0,
    'LEFT': 0,
    'RIGHT': 0,
}

MAX_HEIGHT = 0.23

# TODO:
# - Setup a scene with just the bin
# - Add one item in its default pose

def get_zoom_K(in_size, out_size, bbox, K, expand_ratio=1.0):
    a = get_zoom_factor(in_size, out_size, bbox, expand_ratio)
    K_new = np.zeros([3,3])
    K_new[:2, :] = a @ K
    K_new[2, 2] = 1
    return K_new

def get_zoom_factor(in_size, out_size, bbox, expand_ratio):
    in_height, in_width = in_size
    out_height, out_width = out_size
    ratio = out_height / out_width
    obj_start_x, obj_start_y, obj_end_x, obj_end_y = bbox
    zoom_c_x = 0.5*(obj_start_x+obj_end_x)
    zoom_c_y = 0.5*(obj_start_y+obj_end_y)

    left_dist = zoom_c_x - obj_start_x
    right_dist = obj_end_x - zoom_c_x
    up_dist = zoom_c_y - obj_start_y
    down_dist = obj_end_y - zoom_c_y

    crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * expand_ratio * 2
    crop_width = crop_height / ratio

    x1 = (zoom_c_x - crop_width / 2)
    x2 = (zoom_c_x + crop_width / 2)
    y1 = (zoom_c_y - crop_height / 2)
    y2 = (zoom_c_y + crop_height / 2)

    pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
    pts2 = np.float32([[0, 0], [0, out_size[0]], [out_size[1], 0]])
    affine_matrix = cv2.getAffineTransform(pts1, pts2)
    return affine_matrix

def project_point_to_image(p, camera, w, h):
    cam_matrix = camera.get_transform().get_world_to_local_matrix()
    cam_proj_matrix = camera.get_camera().get_projection()
    p_image = cam_proj_matrix * (cam_matrix * p)
    p_image = visii.vec2(p_image) / p_image.w
    p_image = p_image * visii.vec2(1,-1)
    p_image = (p_image + visii.vec2(1,1)) * 0.5
    return [p_image[0]*w, p_image[1]*h]

def setup_default_scene(scene_data):
    mesh_root_path = ''
    loader = DictLoader(scene_data, mesh_root_path)
    entity_list = loader.entity_list  # list of entities to export
    create_bin(scene_data['scene']['table_dims'])
    scene.add_default_camera()
    set_assigned_materials(scene_data)
    scene.add_default_lights()

def setup_camera(camera, pos_camera_random):

    entity = visii.entity.get('floor')
    obj_center = visii_tools.utils.vec_to_numpy(entity.get_aabb_center())
    camera.get_transform().look_at(
        at=visii_tools.utils.to_visii_vec(obj_center),
        up=visii.vec3(0, 0, 1),
        eye=visii_tools.utils.to_visii_vec(obj_center + pos_camera_random),
    )
        
    K = np.array([[921.5787, 0, 637.0372], [0, 919.6680, 362.2722], [0, 0, 1]])
    width=1280
    height=720
    top_left = project_point_to_image(visii.vec4(-0.185,0.15,0.275,1), camera, width, height)
    top_right = project_point_to_image(visii.vec4(-0.185,-0.15,0.275,1), camera, width, height)
    bot_right = project_point_to_image(visii.vec4(-0.185,-0.15,0.05,1), camera, width, height)
    bot_left = project_point_to_image(visii.vec4(-0.185,0.15,0.05,1), camera, width, height)

    x_min = min(top_left[0], bot_left[0])
    x_max = max(top_right[0], bot_right[0])

    y_min = min(top_left[1], top_right[1])
    y_max = max(bot_left[1], bot_right[1])

    K = get_zoom_K((1280,720), (256,256), (x_min, y_min, x_max, y_max), K)
    width=256
    height=256
    camera.get_camera().set_intrinsics(K[0,0], K[1,1], K[0,2], K[1,2], width, height)

def append_to_h5(h5_file, data0, mask0, data1, mask1):
    if 'frame0_data' not in h5_file:
        h5_file.create_dataset('frame0_data', data=data0, compression="gzip", chunks=True, maxshape=(None,256,256,4))
        h5_file.create_dataset('frame1_data', data=data1, compression="gzip", chunks=True, maxshape=(None,256,256,4))
        h5_file.create_dataset('frame0_mask', data=mask0, compression="gzip", chunks=True, maxshape=(None,256,256))
        h5_file.create_dataset('frame1_mask', data=mask1, compression="gzip", chunks=True, maxshape=(None,256,256))
        # h5_file.create_dataset('frame0_metadata', compression="gzip", chunks=True, maxshape=(None,), dtype=object)
        # h5_file.create_dataset('frame1_metadata', compression="gzip", chunks=True, maxshape=(None,), dtype=object)
    else:
        h5_file['frame0_data'].resize(h5_file['frame0_data'].shape[0] + 1, axis=0)
        h5_file['frame0_data'][-h5_file['frame0_data'].shape[0]:] = data0
        h5_file['frame0_mask'].resize(h5_file['frame0_mask'].shape[0] + 1, axis=0)
        h5_file['frame0_mask'][-h5_file['frame0_mask'].shape[0]:] = mask0
        h5_file['frame1_data'].resize(h5_file['frame1_data'].shape[0] + 1, axis=0)
        h5_file['frame1_data'][-h5_file['frame1_data'].shape[0]:] = data1
        h5_file['frame1_mask'].resize(h5_file['frame1_mask'].shape[0] + 1, axis=0)
        h5_file['frame1_mask'][-h5_file['frame1_mask'].shape[0]:] = mask1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input dataset directory')
    # parser.add_argument('-n', '--num_items', default=1, type=int, help='Max number of items to include in the layout')
    args = parser.parse_args()

    h5_file = h5py.File(args.input, 'r')
    
    num_samples = h5_file['frame0_data'].shape[0]
    print(f"num_samples: {num_samples}")

    for i in range(num_samples):
        # if i < 3:
        #     continue
        print(f"i: {i}")
        im0 = cv2.cvtColor(h5_file['frame0_data'][i], cv2.COLOR_BGR2RGB)
        im1 = cv2.cvtColor(h5_file['frame1_data'][i], cv2.COLOR_BGR2RGB)
        label0 = label2rgb(h5_file['frame0_mask'][i])
        label1 = label2rgb(h5_file['frame1_mask'][i])
        depth0 = h5_file['frame0_depth'][i]
        depth1 = h5_file['frame1_depth'][i]
        md0 = json.loads(h5_file['frame0_metadata'][i])
        md1 = json.loads(h5_file['frame1_metadata'][i])

        # relevant_labels0 = []
        # relevant_masks = np.zeros((256,256), dtype=bool)
        # for k in md0['name_to_id_map'].keys():
        #     if k.startswith('meshes_obj'):
        #         mask = h5_file['frame0_mask'][0] == md0['name_to_id_map'][k]
        #         relevant_masks = relevant_masks & mask
        #         # relevant_labels0.append(md0['name_to_id_map'][k])
        



        cv2.imshow('frame0', im0)
        cv2.imshow('frame1', im1)
        cv2.imshow('label0', label0)
        cv2.imshow('label1', label1)
        cv2.imshow('depth0', depth0)
        cv2.imshow('depth1', depth1)
        cv2.waitKey(0)

    # import pdb; pdb.set_trace()
    
        
        # Capture the 1st frame and masks
    #     render_scene(render_cfg, scene_data, 1, i)

    #     # Add back the replacement object
    #     scene_data['objects']['meshes_obj_new'] = object_to_replace_data

    #     # Capture the 2nd frame and masks
    #     # TODO: Is this the right place to do this? probably not, right?
    #     # visii.initialize(headless=False, verbose=True, lazy_updates=True)
    #     render_scene(render_cfg, scene_data, 1, i)


            

    # # # for i in range(10):
    # # for i in range(10):
        
    # #     item = random.choice(groups[groupname])
    # #     obj_mesh = trimesh.load_mesh(f'./data/google_scanned_filtered/{item}/meshes/model.obj')
    # #     # import pdb; pdb.set_trace()

    # #     # obj_width = abs(obj_mesh.bounds[0,1] - obj_mesh.bounds[1,1])
    # #     x_len = abs(obj_mesh.bounds[0,0] - obj_mesh.bounds[1,0])
    # #     y_len = abs(obj_mesh.bounds[0,1] - obj_mesh.bounds[1,1])
    # #     z_len = abs(obj_mesh.bounds[0,2] - obj_mesh.bounds[1,2])
    # #     # obj_width = abs(obj_mesh.bounds[0,1] - obj_mesh.bounds[1,1])

    #     # if z_len >= MAX_HEIGHT:
    #     #     continue

    # #     transform = np.identity(4)
    # #     transform[2,3] = 0.045 # lift off of the surface of the table
    # #     # transform[0,3]= x_len * levels + 0.01
    # #     # transform[1,3] = 0.1125 # to -0.1
    # #     # transform[1,3] = obj_width/2 + 0.01
    # #     # offset += obj_width/2
    # #     # transform[1,3] = 0.13 - offset
    # #     # y_offset -= random.uniform(0.0, 0.01)

        
        
    # #     # print(f'offset: {offset}')

    # #     # if object is wider on the y-axis then we should flip it
    # #     origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    # #     if y_len > x_len:
    # #         Rz = trimesh.transformations.rotation_matrix(1.57, zaxis)
    # #         transform = trimesh.transformations.concatenate_matrices(Rz, transform)
    # #         x_len_ = x_len
    # #         x_len = y_len
    # #         y_len = x_len_
    # #     # else:
    # #     # z_rotate = random.uniform(0.0,4.0)
    # #     # Rz = trimesh.transformations.rotation_matrix(z_rotate, zaxis)
    # #     # transform = trimesh.transformations.concatenate_matrices(Rz, transform)
        
    # #     transform[1,3] = y_offset - (y_len/2)
    # #     if y_offset - y_len < -0.15:
    # #         if levels > 1:
    # #             continue
    # #         y_offset = 0.15
    # #         levels += 1
    # #         continue
    # #         # continue
    # #     y_offset -= y_len
        
    # #     # transform[1,3] = -0.125
    # #     print(y_offset)
    # #     if y_offset < -0.15:
    # #         if levels > 1:
    # #             break
    # #         y_offset = 0.15
    # #         levels += 1

    # #     scene_data['objects'][f'meshes_obj_{i}'] = {
    # #         'entity': {
    # #             'type': 'MESH',
    # #             'file': f'./data/google_scanned_filtered/{item}/meshes/model.obj',
    # #         },
    # #         'transform': {
    # #             'transform': transform,
    # #         },
    # #         'metadata': {
    # #             'export': True,
    # #             'obj_path': f'./data/google_scanned_filtered/{item}/meshes/model.obj',
    # #             'category': 'meshes',
    # #             'texture_path': f'./data/google_scanned_filtered/{item}/materials/textures/texture.png'
    # #         },
    # #     }
    
    # # # scene_data['objects'] = {}
    # # render_cfg['camera_sampling']['radius']['min'] = 0.95
    # # render_cfg['camera_sampling']['radius']['max'] = 1.95
    # # render_cfg['camera']['spp'] = 2000
    # # render_cfg['camera']['width'] = 1280
    # # render_cfg['camera']['height'] = 720

    # # render_cfg['camera']['width'] = 256
    # # render_cfg['camera']['height'] = 256

    
    # # render_scene(render_cfg, scene_data, 1, i)
    # #     # visii.deinitialize()
    # # import pdb; pdb.set_trace()
    # # random.seed(3)

    # # # TODO:
    # # # - read in list of files in the input path
    # # # - group files into unique groups
    # # # - randomly sample from list of groups
    # # # - randomly sample from within selected group
    # # # - either: keep sampling until we can't fit anymore  or: sample a certain number of groups given by a param
    # # # - 

    # # # Read in list of files from the input path
    # # folders = os.listdir(args.input)

    # # # Group by the first two components of the name
    # # groups = defaultdict(list)
    # # for x in folders:
    # #     groupname = '_'.join(x.split('_')[:2])
    # #     groups[groupname].append(x)
    
    # # # Randomly sample from list of groups
    # # group_keys = list(groups.keys())
    # # random.shuffle(group_keys)

    # # y_offset = 0.15

    # # for i in range(args.num_items):
    # #     groupname = group_keys[i]
    # #     item = random.choice(groups[groupname])

    # #     obj_mesh = trimesh.load_mesh(f'./data/google_scanned_filtered/{item}/meshes/model.obj')
    # #     # import pdb; pdb.set_trace()

    # #     obj_width = abs(obj_mesh.bounds[0,1] - obj_mesh.bounds[1,1])

    # #     transform = np.identity(4)
    # #     transform[2,3] = 0.045 # lift off of the surface of the table
    # #     # transform[1,3] = 0.1125 # to -0.1
    # #     # transform[1,3] = obj_width/2 + 0.01
    # #     # offset += obj_width/2
    # #     # transform[1,3] = 0.13 - offset
    # #     transform[1,3] = y_offset - (obj_width/2)
    # #     y_offset -= obj_width
    # #     y_offset -= random.uniform(0.0, 0.05)
        
    # #     # print(f'offset: {offset}')
        
    # #     # transform[1,3] = -0.125
    # #     print(y_offset)
    # #     if y_offset < -0.15:
    # #         break

    # #     scene_data['objects'][f'meshes_obj_{i}'] = {
    # #         'entity': {
    # #             'type': 'MESH',
    # #             'file': f'./data/google_scanned_filtered/{item}/meshes/model.obj',
    # #         },
    # #         'transform': {
    # #             'transform': transform,
    # #         },
    # #         'metadata': {
    # #             'export': True,
    # #             'obj_path': f'./data/google_scanned_filtered/{item}/meshes/model.obj',
    # #             'category': 'meshes',
    # #             'texture_path': f'./data/google_scanned_filtered/{item}/materials/textures/texture.png'
    # #         },
    # #     }

        

    



    # # # scene_data['objects']['meshes_obj_0'] = {
    # # #     'entity': {
    # # #         'type': 'MESH',
    # # #         'file': '/home/aurmr/liyi/Amazon/dataset/google_scanned/Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box/meshes/model.obj',
    # # #     },
    # # #     'transform': {
    # # #         'transform': np.identity(4),
    # # #     },
    # # #     'metadata': {
    # # #         'export': True,
    # # #         'obj_path': '/home/aurmr/liyi/Amazon/dataset/google_scanned/Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box/meshes/model.obj',
    # # #         'category': 'meshes',
    # # #         'texture_path': '/home/aurmr/liyi/Amazon/dataset/google_scanned/Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box/materials/textures/texture.png'
    # # #     },
    # # # }


    # # np.save('./0_scene_arrangement.npy', scene_data)

    # # # generate scene contain single object
    # # model_path = '/home/aurmr/liyi/Amazon/dataset/google_scanned'
    # # num_objects = 3
    # # num_sample_each_object = 1
    # # export_path = '/home/aurmr/liyi/Amazon/dataset/google_scanned_scene/scene_single_object'
    # # num_scene = 10
    # # for i in tqdm(range(num_scene)):
    # #     manager = SceneManager(model_path)
    # #     export_folder = os.path.join(export_path, f"{i:06d}")
    # #     os.makedirs(export_folder, exist_ok=True)
    # #     for sample_idx in range(num_sample_each_object):
    # #         manager.generate_scene_multiple_object(num_objects, 10, compute_stable_pose=True)
    # #         scene_data = manager.get_scene_cfg(type='export')
    # #         scene_file = os.path.join(export_folder, "0_scene_arrangement.npy")
    # #         np.save(scene_file, scene_data)
    # #         print(scene_file)
    # #         manager.reset()