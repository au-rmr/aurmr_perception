# Third Party
import nvisii as visii

# First Party
from visii_tools.utils import mat_to_numpy, visii_camera_frame_to_rdf, visii_transform_to_numpy
from visii_tools.writer.writer_interface import WriterInterface


class CameraStateWriter(WriterInterface):
    """Writes the camera state to 'metadata' entry

    """

    def __init__(self, verbosity=0):
        WriterInterface.__init__(self, verbosity)

    def run(self, h5, cameras: dict, **kwargs):
        """Compute intrinsics, camera pose

        Write them to the h5 object

        """

        for cam_name, cam_data in cameras.items():
            h5_camera = h5.create_group(f"cameras/{cam_name}/info")
            self.write_single_camera_data(h5_camera, cam_name, cam_data)

    def write_single_camera_data(self, h5, camera_name, camera_data):
        """Write the camera data to the h5 object

        """

        if self._verbosity > 0:
            print(f"Writing single camera data for camera: {camera_name}")

        width = camera_data['width']
        height = camera_data['height']

        intrinsics = visii.entity.get(
            camera_name).get_camera().get_intrinsic_matrix(width, height)
        intrinsics = mat_to_numpy(intrinsics)

        # opengl frame
        world_t_camera_visii = visii_transform_to_numpy(visii.entity.get(
            camera_name).get_transform(), include_affine=False)

        # opencv frame
        world_t_camera = visii_camera_frame_to_rdf(world_t_camera_visii)

        # write data to h5py.Group object
        h5["width"] = width
        h5["height"] = height
        h5["transform"] = world_t_camera
        h5["transform_opengl"] = world_t_camera_visii
        h5["intrinsics_matrix"] = intrinsics
