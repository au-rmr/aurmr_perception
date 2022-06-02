# Standard Library
import copy
import io
import json
import os
import tempfile
import warnings

# Third Party
import cv2
import h5py
# import json_tricks
import numpy as np
# visii
import nvisii as visii
import json
# import pycocotools.mask
# import transforms3d
from PIL import Image

"""
Utilities for rendering data from visii
"""

# If value is above this threshold (in absolute value)
# then it corresponds to background pixel
EMPTY_PIXEL_THRESHOLD = 1e+36


def binary_mask_to_polygons(binary_mask):
    """Converts binary mask to polygon representation


    Copied from https://stackoverflow.com/a/64649730

    Args:
        binary_mask: HxW

    Returns:
        polygons: list[list[float]] [x1,y1,x2,y2] see https://detectron2.readthedocs.io/tutorials/datasets.html

    """
    contours, _ = cv2.findContours(binary_mask.astype(
        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []

    for obj in contours:
        coords = []

        for point in obj:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

        polygons.append(coords)

    return polygons


def render_rgb_deprecated(width, height, spp):
    raise NotImplementedError("Need to figure this out")
    img = visii.render(width=width,
                       height=height,
                       samples_per_pixel=spp,
                       )
    img = reshape_data_tuple(img, width, height)

    # these are float values in range [0,1]
    img = img[..., :3]

    # multiply by 255 and cast to uint8
    img = np.clip((255 * img), 0, 255).astype(np.uint8)
    return img


def render_rgb(width, height, spp, filename: str = None):
    """Renders rgb using tempfile.

    You can optionally specify a filename to render to instead.

    Returns:
        PIL Image
    """
    visii.sample_pixel_area(x_sample_interval=(0.0, 1.0),
                            y_sample_interval=(0.0, 1.0)
                            )

    if filename is not None:

        visii.render_to_png(
            width=width,
            height=height,
            samples_per_pixel=spp,
            image_path=filename,
        )

        with open(filename, 'rb') as f:
            rgb_bytes = f.read()
    else:
        with tempfile.NamedTemporaryFile() as f:
            visii.render_to_png(
                width=width,
                height=height,
                samples_per_pixel=spp,
                image_path=f.name,
            )
            rgb_bytes = f.read()

    return Image.open(io.BytesIO(rgb_bytes))


def render_distance(width, height):
    """Renders a distance image.

    - This is NOT A DEPTH IMAGE in the standard computer vision
    conventions.
    - Background pixels are assigned a distance of 0

    d = sqrt(X_c^2 + Y_c^2 + Z_c^2) in OpenCV convention
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    It's the distance from camera origin to the point P.

    Args:
        width: int, image width
        height: int, image height

    Returns:
        distance: numpy array H x W, dtype = np.float (meters)
        Missing/background values are set to 0 distance

    """

    visii.sample_pixel_area(
        x_sample_interval=(.5, .5),
        y_sample_interval=(.5, .5))

    # tuple
    # background/empty pixels get assigned the value
    # -3.4028234663852886e+38
    distance = visii.render_data(
        width=int(width),
        height=int(height),
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="depth"
    )

    # H x W x 4, dtype=fl
    distance = reshape_data_tuple(distance, width, height)

    # H x W
    distance = distance[..., 0]

    # render segmentation image to detect background pixels
    # and set their distance values to 0
    seg = render_segmentation(width, height)
    distance[seg < 0] = 0

    return distance


def render_segmentation(width, height):
    """Renders a segmentation image

    Entities are stored with an id.

    For an entity called <name> we can access it's id
    The id can be accessed using `visii.entity.get_id()`

    The segemntation image is according to a visii entitity's id.

    TODO (lmanuelli): Is what gets rendered in the segmentation image
    the entity.get_id(), texture.get_id()?

    ```
    ids = visii.texture.get_name_to_id_map()
    index = ids[<name>]
    visii.texture.get(<object_name>).get_id()
    ```

    Args:
        width: int, image width
        height: int, image height

    Returns:
        depth_array (np.array): HxW, dtype = np.uint16 (I;16 as PIL mode)
        background pixels are set to np.iinfo(np.uint16).max = 65535

    """

    visii.sample_pixel_area(
        x_sample_interval=(.5, .5),
        y_sample_interval=(.5, .5))

    # tuple
    # background/empty pixels get assigned FLT_MAX values
    # in C++
    # https://nvidia.slack.com/archives/C01344WFB9C/p1628181340008700
    seg_raw = visii.render_data(
        width=int(width),
        height=int(height),
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="entity_id"
    )

    # HxWx4 numpy array. The last dimension is just repeated values
    # extract only the first entry
    seg_raw = reshape_data_tuple(
        seg_raw, width, height)
    seg_raw = seg_raw[..., 0]

    # set background segmentation id to np.iinfo(np.uint16).max
    # see https://nvidia.slack.com/archives/C01344WFB9C/p1628181340008700 for more details
    background_id = np.iinfo(np.uint16).max
    seg = seg_raw.astype(np.uint16)
    seg[seg_raw < 0] = background_id
    seg[seg_raw > EMPTY_PIXEL_THRESHOLD] = background_id

    return seg


def remove_other_objects_and_render_segmentation(width, height, obj_name):
    """Render segmentation with only specified object in scene

    """
    id_keys_map = visii.entity.get_name_to_id_map()
    transforms_to_keep = {}

    # Clear transforms for all objects except for cameras
    # and object of interest
    for name in id_keys_map.keys():
        if 'camera' in name.lower() or obj_name in name:
            continue
        trans_to_keep = visii.entity.get(name).get_transform()
        transforms_to_keep[name] = trans_to_keep
        visii.entity.get(name).clear_transform()

    # Percentage visibility through full segmentation mask.
    segmentation_unique_mask = render_segmentation(width, height)

    # Return objects to their previous positions
    for entity_name in transforms_to_keep.keys():
        visii.entity.get(entity_name).set_transform(
            transforms_to_keep[entity_name])

    return segmentation_unique_mask


def binary_mask_from_segmentation(segmentation, label):
    """Extracts a binary mask for 'label' from a segmentation mask

    Args:
        segmentation: HxW, dtype=int
        label: int
    """

    return np.where(segmentation == label,
                    np.ones_like(segmentation, dtype=np.bool),
                    np.zeros_like(segmentation, dtype=np.bool),
                    )


def render_normals(width, height):
    """Render a normal image

    Args:
        width: int, image width
        height: int, image height

    Returns:
        normals_array: numpy array HxWx3, dtype=float, normals in world frame

    """
    raise ValueError("Need to properly handle background")
    visii.sample_pixel_area(
        x_sample_interval=(.5, .5),
        y_sample_interval=(.5, .5))

    normals_array = visii.render_data(
        width=int(width),
        height=int(height),
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="normal"
    )

    normals_array = reshape_data_tuple(normals_array, width, height)
    return normals_array[:, :, :3]


def render_positions(width, height):
    """Render a normal image

    Args:
        width: int, image width
        height: int, image height

    Returns:
        position_array: numpy array HxWx3, dtype=float, position in world frame

    """
    raise ValueError("Need to properly handle background pixels")
    visii.sample_pixel_area(
        x_sample_interval=(.5, .5),
        y_sample_interval=(.5, .5))

    position_array = visii.render_data(
        width=int(width),
        height=int(height),
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="position"
    )

    position_array = reshape_data_tuple(position_array, width, height)
    return position_array[:, :, :3]


def render_texture_coordinates(width, height):
    """Render the texture coordinates

    Note(lmanuelli): This doesn't work. Currently the texture coordinates on an object just gets rendered as 0 on all the KLT boxes.

    Args:
        width: int, image width
        height: int, image height

    Returns:
        position_array: numpy array HxWx3, dtype=float, texture coordinates

    """
    raise ValueError("Need to properly handle background pixels")
    visii.sample_pixel_area(
        x_sample_interval=(.5, .5),
        y_sample_interval=(.5, .5))

    warnings.warn("Doesn't work: code for rendering texture coordinates")
    texture_coords = visii.render_data(
        width=int(width),
        height=int(height),
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="texture_coordinates"
    )

    # HxWx3, dtype=float
    texture_coords = reshape_data_tuple(texture_coords, width, height)
    return texture_coords[:, :, :3]


def depth_image_from_distance_image(distance, intrinsics):
    """Computes depth image from distance image.

    Background pixels have depth of 0

    Args:
        distance: HxW float array (meters)
        intrinsics: 3x3 float array

    Returns:
        z: HxW float array (meters)

    """
    fx = intrinsics[0, 0]
    cx = intrinsics[0, 2]
    fy = intrinsics[1, 1]
    cy = intrinsics[1, 2]

    height, width = distance.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)

    x_over_z = (px - cx) / fx
    y_over_z = (py - cy) / fy

    z = distance / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    return z


def reshape_data_tuple(data, width, height):
    """Reshapes a data tuple to OpenCV coordinates

    """

    # H x W x 4 array but (0,0) is bottom left corner
    # need to flip it using PIL
    data = np.array(data).reshape(height, width, 4)

    # flip along first axis
    # this is equivalent to PIL.FlIP_TOP_BOTTOM
    data = np.flip(data, axis=0)

    return data


def PIL_image_to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='png')
    byte_im = buf.getvalue()
    return byte_im
