
# First Party
from visii_tools.writer.camera_state_writer import CameraStateWriter
from visii_tools.writer.depth_writer import DepthWriter
from visii_tools.writer.object_state_writer import ObjectStateWriter
from visii_tools.writer.rgb_writer import RGBWriter
from visii_tools.writer.segmentation_writer import SegmentationWriter


def make_default_writer_list():
    """Construct a default writer pipeline with standard outputs

    """
    writer_object_list = []

    camera_state_writer = CameraStateWriter()
    writer_object_list.append(camera_state_writer)

    rgb_writer = RGBWriter()
    writer_object_list.append(rgb_writer)

    depth_writer = DepthWriter()
    writer_object_list.append(depth_writer)

    segmentation_writer = SegmentationWriter()
    writer_object_list.append(segmentation_writer)

    object_state_writer = ObjectStateWriter()
    writer_object_list.append(object_state_writer)

    d = {'camera_state': camera_state_writer,
         'rgb': rgb_writer,
         'depth': depth_writer,
         'segmentation': segmentation_writer,
         'object_state_writer': object_state_writer,
         }
    return writer_object_list, d
