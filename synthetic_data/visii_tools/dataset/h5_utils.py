# Standard Library
import copy
import io
import logging

# Third Party
import h5py
import numpy as np
from PIL import Image

PRIMITIVE_TYPES = (int, str, np.ndarray, float, list)
log = logging.getLogger(__name__)


def PIL_to_PNG(img: Image) -> bytes:
    '''
    Save a numpy array as a PNG, then get it out as a binary blob

    img: PIL image
    '''
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def PNG_to_PIL(png: bytes) -> Image:
    """Convert png byte string to PIL Image

    """
    stream = io.BytesIO(png)
    return Image.open(stream)


def save_PIL_image(h5, key, img, img_type: str = None):
    """
    Img should be PIL image

    Args:
        img_type: optional image type annotation that will be stored
            in dset.attrs

    """

    # This is due to the fact that when saving to PNG format it
    # will clip any values outside the range [0, 65535].
    # This can lead to unintended behavior so we raise an error
    # Save directly as uncompressed numpy array if you need a different format
    assert img.mode in ['RGB', 'RGBA',
                        'I;16'], f"Unsupported image mode {img.mode}"

    # convert image to bytes
    byte_im = PIL_to_PNG(img)

    # TODO (lmanuelli): does the dtype here need to
    # match the dtype of the byte_img?
    # seems like you can save any type of PIL image here
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = h5.create_dataset(key, (1,), dtype=dt)
    dset[0] = np.fromstring(byte_im, dtype='uint8')

    # add some metadata to indicate this is an image
    dset.attrs['is_image'] = True
    if img_type is not None:
        dset.attrs['type'] = img_type


def read_PIL_image(dset) -> Image:
    """Read a PIL image stored in a hdf5.Dataset object

    """
    # numpy array, dtype=np.uint8
    data = dset[0]
    # bytes = np.ndarray.tostring(data)
    img_PIL = PNG_to_PIL(data.tobytes())
    return img_PIL


def save_depth_image(h5, key, img, scale=1000, dtype=np.uint16):
    """Save a depth image into hdf5 Dataset object


    Args:
        img: np.array HxW with float dtype
        scale: (float) how much we should scale the depth image by to save it
            default is 1000 so it's in millimeters
        dtype: dtype used to save the image

    """

    iinfo = np.iinfo(dtype)
    img_scaled = img * scale

    # don't allow negative values
    img = np.clip(img, 0, np.inf)

    # set it to zero if it exceeds max value
    # alternatively could set it to iinfo.max
    img_scaled[img_scaled >= iinfo.max] = 0

    # cast it to dtype
    img_scaled = img_scaled.astype(dtype)

    # set it as PIL image
    img_scaled_PIL = Image.fromarray(img_scaled)

    save_PIL_image(h5, key, img_scaled_PIL, img_type='depth')

    # save out the scale we used to encode this image
    h5[key].attrs['scale'] = scale


def read_image(dset) -> Image:
    """Read image from hdf5.Dataset

    Can do custom logic to decode specific image types.
    - Depth Image: will automatically convert depth image back to meters


    """

    # assert it is an image
    assert dset.attrs['is_image']
    img_PIL = read_PIL_image(dset)
    img_type = dset.attrs.get("type", "")

    # convert depth image to meters
    if img_type == "depth":
        img = np.asarray(img_PIL).astype(np.float32) / dset.attrs['scale']
        img_PIL = Image.fromarray(img)

    return img_PIL


def dict2hdf5(h5, data, key_list=None, ignore_unsupported_types=False):
    """Save dict to h5 object.

    Recurses through the dictionary saving to the hdf5 object
    """
    if key_list is None:
        key_list = []
    if isinstance(data, dict):
        for key, val in data.items():
            key_list_cur = copy.deepcopy(key_list)
            key_list_cur.append(key)
            dict2hdf5(h5, val, key_list_cur)
    elif isinstance(data, PRIMITIVE_TYPES):
        h5_key = '/'.join(key_list)
        h5.create_dataset(h5_key, data=data)
    elif isinstance(data, bytes):
        h5_key = '/'.join(key_list)
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        dset = h5.create_dataset(h5_key, (1,), dtype=dt)
        dset[0] = np.fromstring(data, dtype='uint8')
    else:
        if ignore_unsupported_types:
            log.debug(f"Encountered unsupported type {type(data)}, skipping")

            h5_key = '/'.join(key_list)
            h5.create_dataset(h5_key, data=data)
        else:
            raise ValueError(f"Encountered unsupported data type {type(data)}")
