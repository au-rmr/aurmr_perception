# Third Party
import nvisii as visii
from PIL import Image

# First Party
# visii_tools
import visii_tools
from visii_tools import render
from visii_tools.dataset import h5_utils
from visii_tools.render import render_utils
from visii_tools.writer.writer_interface import WriterInterface


class SegmentationWriter(WriterInterface):
    """Renders a segmentation image, save it to hdf5 file

    """

    def __init__(self, verbosity=0):
        WriterInterface.__init__(self, verbosity)

    def run(self, h5, cameras: dict, **kwargs):
        """Render RGB image, save it to file

        """

        for cam_name, cam_data in cameras.items():
            # create group if it doesn't already exist
            key = f"cameras/{cam_name}/image_data/segmentation"
            self.render_single_image(h5, key, cam_name, cam_data)

    def render_single_image(self, h5, key, camera_name, camera_data):
        """Render segmentation image for specified camera
        Save it to h5 file using specified key

        """
        cam_entity = visii.entity.get(camera_name)
        visii.set_camera_entity(cam_entity)
        height = camera_data['height']
        width = camera_data['width']

        # np.uint16
        img = render_utils.render_segmentation(width, height)
        img_PIL = Image.fromarray(img)
        h5_utils.save_PIL_image(h5, key=key, img=img_PIL,
                                img_type='segmentation')
