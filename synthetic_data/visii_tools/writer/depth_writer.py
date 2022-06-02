

# Third Party
import numpy as np
import nvisii as visii

# First Party
from visii_tools.dataset import h5_utils
from visii_tools.render import render_utils
from visii_tools.utils import mat_to_numpy
from visii_tools.writer.writer_interface import WriterInterface


class DepthWriter(WriterInterface):
    """Renders a depth image, stores it in the 'write_globals` object

    Later that can be saved to disk by the HDF5Writer

    """

    def __init__(self, verbosity=0, dtype=np.uint16, scale=1000):
        WriterInterface.__init__(self, verbosity=verbosity)
        self._dtype = dtype
        self._scale = scale

    def run(self, h5, cameras: dict, dtype=np.int16, scale=1000, **kwargs):
        """Render RGB image, save it to file

        """

        for cam_name, cam_data in cameras.items():
            # create group if it doesn't already exist

            key = f"cameras/{cam_name}/image_data/depth"
            self.render_single_depth_image(h5,
                                           key,
                                           cam_name,
                                           cam_data,
                                           dtype=self._dtype,
                                           scale=self._scale)

    def render_single_depth_image(self,
                                  h5,
                                  key,
                                  camera_name,
                                  camera_data,
                                  dtype=np.int16,
                                  scale=1000):
        """Render depth image and save it to h5 file

        """

        cam_entity = visii.entity.get(camera_name)
        visii.set_camera_entity(cam_entity)

        height = camera_data['height']
        width = camera_data['width']

        intrinsics = visii.entity.get(
            camera_name).get_camera().get_intrinsic_matrix(width, height)
        intrinsics = mat_to_numpy(intrinsics)

        distance = render_utils.render_distance(width, height)
        depth = render_utils.depth_image_from_distance_image(
            distance, intrinsics)

        # save depth image to hdf5
        h5_utils.save_depth_image(h5,
                                  key=key,
                                  img=depth,
                                  scale=scale,
                                  dtype=dtype)
