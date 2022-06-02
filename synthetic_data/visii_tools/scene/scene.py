"""Utilities for constructing default scenes
"""


# Third Party
import numpy as np
# visii
import nvisii as visii
from visii_tools.utils import set_transform_from_matrix
import trimesh
import cv2

def construct_default_scene():
    add_default_camera()
    add_default_lights()
    # add_default_ground_plane()


def add_default_lights():
    """Adds some default lights to the scene

    """
    transform = visii.transform.create("areaLight1")
    # transform.set_angle_axis(1.57079633, [0,1,0])
    # transform.set_angle_axis(2.35619449, [0,1,0])
    transform.set_angle_axis(0.785398163, [0,1,0])
    transform.set_angle_axis(1.57079633- 0.5, [0,1,0])
    # just set some defaults for now
    areaLight1 = visii.entity.create(
        name="areaLight1",
        light=visii.light.create("areaLight1"),
        transform=transform,
        mesh=visii.mesh.create_plane("areaLight1"),
    )

    # areaLight1.get_transform().set_rotation(visii.angleAxis(3.14, (1, 0, 0)))
    areaLight1.get_light().set_intensity(1)
    areaLight1.get_light().set_intensity(10)
    areaLight1.get_light().set_exposure(0)
    areaLight1.get_light().set_falloff(0)
    # areaLight1.get_light().use_surface_area(True)
    # areaLight1.get_light().set_intensity(6)
    # areaLight1.get_light().set_exposure(-2)
    # areaLight1.get_light().set_temperature(8000)
    # areaLight1.get_transform().set_position((0, 0, .6))
    areaLight1.get_transform().set_position((-2.0, 0.0, 0.0))
    areaLight1.get_transform().set_scale((3.0, 3.0, 0.2))
    # areaLight1.get_transform().look_at([1,0,0])


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

def add_default_camera():
    """Adds a default camera

    """
    width = 640
    height = 480
    width = 1280
    height= 720

    # width=256
    # height=256

    camera = visii.entity.create(
        name="camera",
        transform=visii.transform.create("camera"),
        camera=visii.camera.create(
            name="camera",
            aspect=float(width) / float(height)
        )
    )

    K = np.array([[614.3858, 0, 318.0248], [0, 613.1120, 241.5148], [0, 0, 1]])
    K = np.array([[921.5787, 0, 637.0372], [0, 919.6680, 362.2722], [0, 0, 1]])
    

    camera.get_camera().set_intrinsics(K[0,0], K[1,1], K[0,2], K[1,2], width, height)

    camera.get_transform().look_at(
        at=(0, 0, 0),
        up=(0, 0, 1),
        eye=(0, 1, 1)
    )


    # top_left = project_point_to_image(visii.vec4(0.225,-0.185,-0.25,1), camera, width, height)
    # top_right = project_point_to_image(visii.vec4(-0.225,-0.185,-0.25,1), camera, width, height)
    # bot_right = project_point_to_image(visii.vec4(-0.225,-0.185,-0.3,1), camera, width, height)
    # bot_left = project_point_to_image(visii.vec4(0.225,-0.185,-0.3,1), camera, width, height)

    # x_min = min(top_left[0], bot_left[0])
    # x_max = max(top_right[0], bot_right[0])

    # y_min = min(top_left[1], top_right[1])
    # y_max = max(bot_left[1], bot_right[1])

    # # K = get_zoom_K((1280,720), (256,256), (504, 305, 759, 465), K)
    # K = get_zoom_K((1280,720), (256,256), (x_min, y_min, x_max, y_max), K)
    # import pdb; pdb.set_trace()

    
    # import pdb; pdb.set_trace()
    # width=256
    # height=256
    # camera.get_camera().set_intrinsics(K[0,0], K[1,1], K[0,2], K[1,2], width, height)
    visii.set_camera_entity(camera)


def add_default_ground_plane(scale=[2,2,1], position=[0,0,0], roughness=1.0):
    # Create a scene
    floor = visii.entity.create(
        name="floor",
        mesh=visii.mesh.create_plane("floor"),
        transform=visii.transform.create("floor"),
        material=visii.material.create("floor")
    )

    floor.get_transform().set_scale(scale)
    floor.get_transform().set_position(position)
    floor.get_material().set_roughness(roughness)


def add_custom_camera(K, height=480, width=640):
    """Adds a default camera

    """
    camera = visii.entity.create(
        name="camera",
        transform=visii.transform.create("camera"),
        # transform=visii.transform.create_from_matrix(name="camera", matrix=np.eye(4)),
        camera=visii.camera.create_from_intrinsics('camera', K[0,0], K[1,1], K[0,2], K[1,2], width, height)
    )

    set_transform_from_matrix(camera.get_transform(), np.eye(4))
    visii.set_camera_entity(camera)