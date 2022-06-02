import os.path

import nvisii as visii
import numpy as np
from typing import List

import visii_tools.render.render_utils
from visii_tools.loader.material_loader import load_cco_texture, update_single_material
from visii_tools.samplers.samplers import hemisphere_random_sampler
from visii_tools.utils import vec_to_numpy


support_materials = {
    "chair":["Wood", "Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Fabric", "Planks", "PaintedWood", "CorrugatedSteel", "Leather", "Paint", "WoodFloor", "Plastic", "Chipboard", "Wicker"],
    "coffeemachine":["Rust", "SheetMetal", "Metal", "PaintedMetal", "CorrugatedSteel", "Paint", "Plastic"],
    "corner":["Wood", "Concrete", "Rust", "Metal", "PaintedMetal", "Planks", "PaintedWood", "CorrugatedSteel", "Paint", "Chipboard"],
    "countertop":["Wood", "Concrete", "Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Tiles", "Planks", "PaintedWood", "Marble", "Paint", "WoodFloor", "Terrazzo", "Chipboard"],
    "dishwasher":["Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Paint", "Plastic"],
    "faucet":["Rust", "SheetMetal", "Metal", "PaintedMetal", "CorrugatedSteel", "Paint", "Plastic"],
    "hood":["Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Planks", "CorrugatedSteel", "Paint", "Plastic"],
    "microwave":["Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Paint", "Plastic"],
    "oven":["Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "CorrugatedSteel", "Paint", "Plastic"],
    "refrigerator":["Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "CorrugatedSteel", "Paint", "Plastic"],
    "sink":["Rust", "MetalPlates", "SheetMetal", "Metal", "CorrugatedSteel", "Paint", "Plastic"],
    "storagefurniture":["Wood", "Rust", "MetalPlates", "Metal", "PaintedMetal", "Planks", "PaintedWood", "CorrugatedSteel", "Paint", "Chipboard"],
    "table":["Wood", "Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Planks", "PaintedWood", "Marble", "Leather", "Paint", "WoodFloor", "Chipboard"],
    "toaster":["Rust", "MetalPlates", "SheetMetal", "Metal", "Paint"],
    "trash":["Rust", "MetalPlates", "Metal", "PaintedMetal", "Paint"],
    "wall":["Bricks", "Wood", "Concrete", "MetalPlates", "Metal", "PaintedMetal", "Planks", "PaintedPlaster", "PaintedWood", "Paint", "Plaster"],
    "window":["Wood", "Rust", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Planks", "PaintedWood", "Paint"],
    "floor":["Gravel", "Carpet", "Bricks", "Wood", "Rock", "Concrete", "Ground", "MetalPlates", "SheetMetal", "Metal", "PaintedMetal", "Asphalt", "PavingStones", "Tiles", "Planks", "PaintedPlaster", "PaintedWood", "Marble", "Paint", "WoodFloor", "Chipboard"]
}

all_valid_texture = list(set(sum([v for k,v in support_materials.items()], [])))

def set_support_materials(textures_dict: dict, class_textures: dict=support_materials):
    """Set the textures for the all non-object entities.

    Note: we need to be careful not to modify the textures for things
    like lights, etc.

    This function randomly selects a texture for each type of non-object entity (e.g. 'table', 'microwave', etc.) and applies that texture to
    the entities' material.
    """
    # get list of all textures
    all_texture_folders = []
    for _, val in textures_dict.items():
        all_texture_folders.append(val)
    easy_texture_folders = []
    for k in all_valid_texture:
        easy_texture_folders.append(textures_dict[k])

    support_materials = {}
    for name in sorted(list(visii.entity.get_name_to_id_map())):
        if (('obj' in name) or (name=='floor')):
            support_entity = visii.entity.get(name)

            # Choose material according the the class of this entity
            # e.g. Table, Microwave, etc.
            entity_class = name.split('_')[0]

            # if entity_class in class_textures:
            #     continue
            # elif 'support' in entity_class.lower():
            #     # supports get remapped as tables
            #     entity_class = 'table'
            # If we haven't already selected a texture for this entity_class
            # do so now
            if entity_class in support_materials:
                support_entity.set_material(support_materials[entity_class])
            else:
                material = support_entity.get_material()

                # choose a random texture from the set
                if(entity_class in class_textures):
                    texture_options = []
                    for k in class_textures[entity_class]:
                        if k in textures_dict:
                            texture_options.append(textures_dict[k])
                else:
                    texture_options = all_texture_folders

                texture_options = sum(texture_options, [])
                texture_dir = np.random.choice(
                    texture_options, 1).item()

                # load the texture into the given material
                load_cco_texture(material, texture_dir)
                support_materials[entity_class] = material
                # print(name, texture_dir)

def set_assigned_materials(scene_cfg):
    """Set the textures for the all non-object entities.

    Note: we need to be careful not to modify the textures for things
    like lights, etc.

    This function randomly selects a texture for each type of non-object entity (e.g. 'table', 'microwave', etc.) and applies that texture to
    the entities' material.
    """
    support_materials = {}
    for name in sorted(list(visii.entity.get_name_to_id_map())):
        if (('obj' in name) or (name=='floor')):
            print(name)
            support_entity = visii.entity.get(name)
            material = support_entity.get_material()

            entity_class = name.split('_')[0]
            # load the texture into the given material
            prefix_name = material.get_name()
            if 'obj' in name:
                color_texture_fname = scene_cfg['objects'][name]['metadata']['texture_path']
            else:
                color_texture_fname = './synthetic_data/cco_textures/PaintedPlaster004_2K-JPG/PaintedPlaster004_2K_Color.jpg'
            print(color_texture_fname)
            assert os.path.exists(color_texture_fname)
            update_single_material(material, prefix_name, color_texture_fname=color_texture_fname)
            support_materials[entity_class] = material
            # print(name, texture_dir)

def create_bin(table_dims):
    floor_width = table_dims[1]
    floor_depth = table_dims[0]
    floor_center = [0,0, table_dims[2]/2+0.11]
    unique_name = 'floor'
    mesh = visii.mesh.create_from_file(unique_name,
                                       './synthetic_data/bin/bin.obj')
    # sphere = visii.entity.create(
    #     name="sandbox_sphere",
    #     mesh= visii.mesh.create_sphere(name="sphere"),
    #     transform = visii.transform.create('spherE_transform', position=[0,0,0]),
    #     material = visii.material.create("sphere")
    # )

    # sphere.get_transform().set_position((-0.185,0.15,0.275))
    # sphere.get_transform().set_scale((0.01, 0.01, 0.01))
    # sphere.get_material().set_base_color((0.1,0.9,0.08))  

    entity = visii.entity.create(
        name=unique_name,
        mesh=mesh,
        transform=visii.transform.create(f"{unique_name}_transform",
                                         position=floor_center,
                                         ),
        material=visii.material.create(unique_name),
    )

def create_floor(table_dims, name: str = "floor") -> bool:
    """Create a floor object if one doesn't already exist.

    Return:
        bool: True if successfully created entity, False otherwise
    """
    # see if this entity already exists
    # if visii.entity.get(name) is not None:
    #     log.warning(
    #         f"Tried to create an entity named {name} but such an entity already exists, skipping")
    #     return False

    # aabb_min = visii.get_scene_min_aabb_corner()
    # aabb_max = visii.get_scene_max_aabb_corner()
    # floor_center = visii.get_scene_aabb_center()
    # floor_center[2] = aabb_min[2]
    # floor_width = aabb_max[0] - aabb_min[0]
    # floor_depth = aabb_max[1] - aabb_min[1]
    floor_width = table_dims[1]
    floor_depth = table_dims[0]
    floor_center = [0, 0, table_dims[2]/2]

    floor_entity = visii.entity.create(
        name=name,
        # QUESTION (lmanuelli): Why are we dividing by 1.5 here?
        mesh=visii.mesh.create_plane(f"{name}_mesh",
                                     [floor_width / 1.5, floor_depth / 1.5]),
        transform=visii.transform.create(f"{name}_transform",
                                         position=floor_center,
                                         ),
        material=visii.material.create(f"{name}_material"),
    )
    return True


def sample_camera_position(cfg,
                           camera: visii.entity,
                           look_at=None) -> bool:
    """Randomly samples a camera position looking at an object.

    Return False if didn't succeed
    """

    c = cfg.camera_sampling
    print(c)

    # sample random position on the hemisphere
    pos_camera_random = hemisphere_random_sampler(
        radius_min=c.radius.min,
        radius_max=c.radius.max,
        azimuth_min=c.azimuth.min,
        azimuth_max=c.azimuth.max,
        elevation_min=c.elevation.min,
        elevation_max=c.elevation.max,
        num_samples=1
    )[0]

    return pos_camera_random
    # log.debug(f"pos_camera:\n {pos_camera_random}")

    camera_distance = np.linalg.norm(pos_camera_random)
    print(f"camera distance: {camera_distance}")


    entity_name = look_at
    # log.debug(f"Centering camera on {entity_name}")
    entity = visii.entity.get(entity_name)
    obj_center = visii_tools.utils.vec_to_numpy(entity.get_aabb_center())

    # randomize a bit
    if c.random_displacement:
        obj_center += np.random.uniform(-camera_distance / 4.,
                                        camera_distance / 4.,
                                        3)

    transform = visii.transform.create('camera_transform')
    transform.look_at(
        at=visii_tools.utils.to_visii_vec(obj_center),
        up=visii.vec3(0, 0, 1),
        eye=visii_tools.utils.to_visii_vec(obj_center + pos_camera_random),
    )
    return transform
    # TODO (lmanuelli): Understand what 'at', 'up', 'eye' mean
    # TODO (lmanuelli): Allow for randomizing 'up' direction
    # in visii transform world
    # see https://gitlab-master.nvidia.com/lmanuelli/visii_tools#visii-camera-coordinate-system for details on camera coordinate
    # system
    # Also see the Going3D section here (https://learnopengl.com/Getting-started/Coordinate-Systems)
    # camera.get_transform().look_at(
    #     at=visii_tools.utils.to_visii_vec(obj_center),
    #     up=visii.vec3(0, 0, 1),
    #     eye=visii_tools.utils.to_visii_vec(obj_center + pos_camera_random),
    # )
    # return True, pos_camera_random


def check_camera_position():
    """Check if the sampled camera position meets our criteria.

    Typical criteria is that objects make up some minimum percentage
    of the scene pixels
    """
