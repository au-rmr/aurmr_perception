# Standard Library
import os
import re
import tempfile
from pathlib import Path
from typing import List

# Third Party
import cv2
import numpy as np
# nvisii
import nvisii
import nvisii as visii
# import pycocotools
import transforms3d
import trimesh
import yaml

# First Party
import visii_tools


def get_project_root():
    return Path(__file__).parents[2]


def yaml_dump(data, filename):
    """Saves data to a yaml file

    """
    with open(filename, 'w') as f:
        yaml.dump(data, f)


def yaml_load(filename):
    """
    Loads data from yaml
    """
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def set_transform_from_matrix(transform: visii.transform, T: np.ndarray, scale: None):
    """
    Sets the position and rotation of the visii transform object
    to those specified by the 4x4 homogeneous transform matrix T.

    We only set position/rotation since we don't want to mess with
    the scale

    Args:
        transform: visii.transform object
        T: 4x4 homoegenous transform

    """

    pos = T[:3, 3]
    quat = transforms3d.quaternions.mat2quat(T[:3, :3])  # w,x,y,z ordering

    # extract components of quaternion so we can convert to visii
    # convention
    w, x, y, z = quat

    # set these on the entities transform
    transform.set_position(pos)

    # note the different ordering here
    transform.set_rotation((x, y, z, w))

    if scale is not None:
        transform.set_scale(scale)


def set_transform(transform: visii.transform,
                  pos: np.ndarray, 
                  quat,
                  scale=None,):
    """Set transform from pos, quat, scale.

    Args:
        quat: Can either be dict-like with keys ['w', 'x', 'y', 'z'] 
            or list like with wxyz ordering
    """

    transform.set_position(pos)
    if hasattr(quat, 'w'):
        w = quat['w']
        x = quat['x']
        y = quat['y']
        z = quat['z']
    else:
        w, x, y, z = quat
    
    transform.set_rotation((x, y, z, w))

    if scale is not None:
        transform.set_scale(scale)


def numpy_to_mat4(matrix):
    """Converts numpy

    Args:
        matrix: 4x4 numpy array

    Returns:
        visii.mat4

    """

    return visii.mat4(*matrix.flatten().tolist())


def visii_transform_to_numpy(transform, include_affine=False):
    """Extract 4x4 transform from visii transform.

    By default we only extract the translation and rotation components

    Args:
        include_affine: If true return the full affine transformation matrix

    """
    mat4 = transform.get_local_to_world_matrix()
    a44 = mat_to_numpy(mat4)  # affine 4x4

    if include_affine:
        return a44
    else:
        # decompose into it's component parts
        T, R, Z, S = transforms3d.affines.decompose44(a44)
        t = np.eye(4)
        t[:3, :3] = R
        t[:-1, -1] = T

        return t


def visii_transform_to_dict(transform):
    """Extracts 4x4 transform matrix from visii_transform

    Note: be careful here since visii transforms have scale as well
    They aren't just homogeneous transforms

    Returns:
        out: dict with keys ['mat', 'scale']

    """

    mat = visii_transform_to_numpy(transform)
    return {'mat': mat,
            }


def vec_to_numpy(vec):
    """Converts a visii.vec3 or visii.vec4 to numpy array

    Returns:
        vec_np: numpy array
    """

    return np.array(tuple(vec))


def to_visii_vec(x):
    """
    Args:
        x: tuple or 1D array to visii vec3 or vec4
    """

    x = tuple(x)

    if len(x) == 3:
        return visii.vec3(*x)
    elif len(x) == 4:
        return visii.vec4(*x)
    else:
        raise ValueError(
            "can only convert length 3 or 4 vectors to visii.vec3 or visii.vec4")


def mat_to_numpy(mat):
    """Converts a visii:mat3, visii:mat4 object to numpy array

    visii:mat objects are in column major order
    """

    assert isinstance(mat, visii.mat3) or isinstance(mat, visii.mat4)

    num_cols = None
    if isinstance(mat, visii.mat3):
        num_cols = 3
    elif isinstance(mat, visii.mat4):
        num_cols = 4

    cols = []
    for i in range(num_cols):
        cols.append(vec_to_numpy(mat[i]))

    mat_np = np.array(cols)
    mat_np = mat_np.transpose()

    return mat_np


def visii_camera_frame_to_rdf(T_world_Cv):
    """Rotates the camera frame to "right-down-forward" frame

    Returns:
        T_world_camera: 4x4 numpy array in "right-down-forward" coordinates

    """

    # C = camera frame (right-down-forward)
    # Cv = visii camera frame (right-up-back)

    T_Cv_C = np.eye(4)
    T_Cv_C[:3, :3] = transforms3d.euler.euler2mat(np.pi, 0, 0)

    T_world_C = T_world_Cv @ T_Cv_C
    return T_world_C


def transform_points_3D(transform, points):
    """
    :param transform: homogeneous transform
    :type transform:  4 x 4 numpy array
    :param points:
    :type points: numpy array, [N,3]
    :return: numpy array [N,3]
    :rtype:
    """

    N = points.shape[0]
    points_homog = np.append(np.transpose(
        points), np.ones([1, N]), axis=0)  # shape [4,N]
    transformed_points_homog = transform.dot(points_homog)  # [4, N]

    transformed_points = np.transpose(
        transformed_points_homog[0:3, :])  # shape [N, 3]
    return transformed_points


def pinhole_projection(points, K):
    """Does the pinhole projection


    Notes: consider using the cameratransform package
    https://cameratransform.readthedocs.io/en/latest/camera.html

    Args:
        points: Nx3 array of points in camera frame
        K: 3x3, camera intrinsics matrix

    Returns:
        image_points: Nx2 array of points in (u,v) ordering

    """
    points = points.astype(np.float)
    rvec = np.array([0, 0, 0], dtype=np.float)  # rotation vector
    tvec = np.array([0, 0, 0], dtype=np.float)
    image_points, _ = cv2.projectPoints(objectPoints=points,
                                        rvec=rvec,
                                        tvec=tvec,
                                        cameraMatrix=K,
                                        distCoeffs=np.array([]))

    return image_points.squeeze(1)


# def binary_mask_to_coco_rle(mask):
#     """Converts binary mask to RLE format
#
#     Copied from https://gitlab-master.nvidia.com/ychao/dex-ycb-toolkit/-/blob/master/lib/coco_eval.py#L65
#
#     Returns:
#         out: dict with keys ['segmentation', 'bbox', 'bbox_mode', 'area']
#     """
#     mask = np.asfortranarray(mask)
#     rle = pycocotools.mask.encode(mask)
#     segmentation = rle
#     segmentation['counts'] = segmentation['counts'].decode('ascii')
#     # https://github.com/cocodataset/cocoapi/issues/36
#     area = pycocotools.mask.area(rle).item()
#     bbox = pycocotools.mask.toBbox(rle).tolist()
#
#     return {'segmentation': segmentation,
#             'bbox': bbox,
#             'bbox_mode': 1,  # see https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.BoxMode.XYWH_ABS
#             'area': area}


def object_cuboid_in_frame(camera_data, object_data):
    """
    Checks whether single object cuboid is inside frame
    """
    width = camera_data['width']
    height = camera_data['height']

    for uv in object_data['projected_cuboid']:
        u = uv[0]
        v = uv[1]

        if u < 0 or u > (width - 1):
            return False
        if v < 0 or v > (height - 1):
            return False

    return True


def check_all_cuboids_in_frame(ndds_data):
    """
    Checks whether the cuboids for all objects are inside the frame

    Returns:
        dict:
            - 'all_cuboids_in_frame': bool
            - 'objects_outside_frame': list of object data that fall outside the frame
        True,False

    """
    camera_data = ndds_data['camera_data']
    objects_outside_frame = []

    for object_data in ndds_data['objects']:
        if not object_cuboid_in_frame(camera_data, object_data):
            objects_outside_frame.append(object_data)

    all_cuboids_in_frame = (len(objects_outside_frame) == 0)

    return {'all_cuboids_in_frame': all_cuboids_in_frame,
            'objects_outside_frame': objects_outside_frame}


def visii_mesh_from_trimesh(name, mesh_t):
    """Create a visii.Mesh from trimesh mesh

    Does this by svaing mesh_t to an obj file and then loading it
    """
    dir = tempfile.TemporaryDirectory()
    obj_file = os.path.join(dir.name, "mesh.obj")
    trimesh.exchange.export.export_mesh(mesh_t, obj_file)

    return visii.mesh.create_from_file(name, obj_file)


def compute_entities_pixel_fraction(camera_name: str,
                                    height: int,
                                    width: int,
                                    entity_names: List[str]) -> float:
    """Compute fraction of pixels corresponding to a specific set of entities.
    """

    cam_entity = nvisii.entity.get(camera_name)
    nvisii.set_camera_entity(cam_entity)
    segmentation_img = visii_tools.render.render_utils.render_segmentation(width, height)

    ids = []
    name_to_id_map = nvisii.entity.get_name_to_id_map()
    for name in entity_names:
        ids.append(name_to_id_map[name])

    return np.isin(segmentation_img, ids).mean()


def create_object(
        name='name',
        path_obj="",
        path_tex=None,
        scale=1,
        metallic=0,
        transmission=0,  # should be 0 or 1
        roughness=1.0,  #
):
    """Create visii entity from file

    """

    obj_mesh = visii.mesh.create_from_file(name, path_obj)
    obj_entity = visii.entity.create(
        name=name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh=obj_mesh,
        transform=visii.transform.create(name),
        material=visii.material.create(name)
    )

    # set the material properties
    obj_entity.get_material().set_metallic(metallic)  # should 0 or 1
    obj_entity.get_material().set_transmission(transmission)  # should 0 or 1
    obj_entity.get_material().set_roughness(roughness)  # default is 1

    # set the texture
    obj_texture = visii.texture.create_from_file(name, path_tex)
    obj_entity.get_material().set_base_color_texture(obj_texture)
    obj_entity.get_transform().set_scale(visii.vec3(scale))

    return obj_entity


def get_entities_with_mesh_component() -> List[str]:
    """Get a list of entities that have a mesh component."""

    entity_name_list = []
    for name in nvisii.entity.get_name_to_id_map():
        if nvisii.entity.get(name).get_mesh() is not None:
            entity_name_list.append(name)

    return entity_name_list


def get_entity_names_matching_regex(regex_str, match_type='match'):
    """Return list of entity names matching regex."""
    entity_names = list(nvisii.entity.get_name_to_id_map().keys())
    r = re.compile(regex_str)
    matches = list(filter(getattr(r, match_type), entity_names))
    return matches


def control_viewer(camera):
    """Code to interactively control the viewer.

    Copied from https://github.com/owl-project/NVISII/blob/master/examples/15.camera_motion_car_blur.py
    """
    camera.get_transform().clear_motion()

    cursor = nvisii.vec4()
    speed_camera = 4.0
    rot = nvisii.vec2(nvisii.pi() * 1.25, 0)

    def interact():
        global speed_camera
        global cursor
        global rot

        # nvisii camera matrix
        cam_matrix = camera.get_transform().get_local_to_world_matrix()
        dt = nvisii.vec4(0, 0, 0, 0)

        # translation
        if nvisii.is_button_held("W"):
            dt[2] = -speed_camera
        if nvisii.is_button_held("S"):
            dt[2] = speed_camera
        if nvisii.is_button_held("A"):
            dt[0] = -speed_camera
        if nvisii.is_button_held("D"):
            dt[0] = speed_camera
        if nvisii.is_button_held("Q"):
            dt[1] = -speed_camera
        if nvisii.is_button_held("E"):
            dt[1] = speed_camera

        # control the camera
        if nvisii.length(dt) > 0.0:
            w_dt = cam_matrix * dt
            camera.get_transform().add_position(nvisii.vec3(w_dt))

        # camera rotation
        cursor[2] = cursor[0]
        cursor[3] = cursor[1]
        cursor[0] = nvisii.get_cursor_pos().x
        cursor[1] = nvisii.get_cursor_pos().y
        if nvisii.is_button_held("MOUSE_LEFT"):
            nvisii.set_cursor_mode("DISABLED")
            rotate_camera = True
        else:
            nvisii.set_cursor_mode("NORMAL")
            rotate_camera = False

        if rotate_camera:
            rot.x -= (cursor[0] - cursor[2]) * 0.001
            rot.y -= (cursor[1] - cursor[3]) * 0.001
            init_rot = nvisii.angleAxis(nvisii.pi() * .5, (1, 0, 0))
            yaw = nvisii.angleAxis(rot.x, (0, 1, 0))
            pitch = nvisii.angleAxis(rot.y, (1, 0, 0))
            camera.get_transform().set_rotation(init_rot * yaw * pitch)

        # change speed movement
        if nvisii.is_button_pressed("UP"):
            speed_camera *= 0.5
            print('decrease speed camera', speed_camera)
        if nvisii.is_button_pressed("DOWN"):
            speed_camera /= 0.5
            print('increase speed camera', speed_camera)

    nvisii.register_pre_render_callback(interact)
    # Standard Library
    import time
    while not nvisii.should_window_close():
        time.sleep(.1)
