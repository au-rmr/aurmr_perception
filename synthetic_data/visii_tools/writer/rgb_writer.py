# Third Party
import numpy as np
import nvisii as visii
from PIL import Image

# First Party
from visii_tools.dataset import h5_utils
from visii_tools.render import render_utils
# visii_tools
from visii_tools.writer.writer_interface import WriterInterface

# nvisii


class RGBWriter(WriterInterface):
    """Renders an rgb image, saves it to file

    """

    def __init__(self, verbosity=0, save_png=False, filename=None):
        WriterInterface.__init__(self, verbosity)
        self._save_png = save_png
        self._filename = filename

    @property
    def save_png(self):
        return self._save_png

    @save_png.setter
    def save_png(self, val):
        self._save_png = val

    def run(self, h5,
            cameras: dict,
            idx: int = None,
            output_dir=None,
            ** kwargs):
        """Render RGB image, save it to file

        """

        for cam_name, cam_data in cameras.items():
            # create group if it doesn't already exist
            key = f"cameras/{cam_name}/image_data/rgb"

            if self._save_png:
                if self._filename is None:
                    filename = f"{str(output_dir)}/{cam_name}_{idx}_rgb.png"
                    assert output_dir is not None, "You must specify output_dir if you want to save the png to a file"
                else:
                    filename = self._filename
            else:
                filename = None

            self.render_single_image(h5, key, cam_name,
                                     cam_data, filename=filename)

    def render_single_image(self,
                            h5,
                            key,
                            camera_name,
                            camera_data,
                            filename: str = None):
        """Render rgb image and save it to h5 object.

        Args:
            filename: If specified then the png image will actually
                get saved to this file.
        """
        print(f"GETTING CMAERA: {camera_name}")
        cam_entity = visii.entity.get(camera_name)
        visii.set_camera_entity(cam_entity)
        visii.sample_pixel_area(x_sample_interval=(0.0, 1.0),
                                y_sample_interval=(0.0, 1.0)
                                )

        height = camera_data['height']
        width = camera_data['width']
        spp = camera_data['spp']
        # spp = 5000

        # renders using a tempfile
        # PIL image, mode = 'RGBA'
        img = render_utils.render_rgb(width, height, spp, filename=filename)
        h5_utils.save_PIL_image(h5, key=key, img=img, img_type='rgb')
