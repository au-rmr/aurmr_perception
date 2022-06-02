# Standard Library
import copy
import json

# Third Party
import h5py
import numpy as np

# First Party
from visii_tools.utils import get_value_from_key_list as get_val
from visii_tools.utils import set_value_from_key_list as set_val
from visii_tools.writer.writer_interface import WriterInterface


class Detectron2Writer(WriterInterface):
    """Writes the detectron2 data to a json file

    """

    def __init__(self):
        raise ValueError("Deprecated")
        WriterInterface.__init__(self)

    def run(self, entity_data, writer_globals, **kwargs):

        output_dir = get_val(writer_globals, 'params/output_dir')
        file_prefix = get_val(writer_globals, 'params/file_prefix')
        height = get_val(writer_globals, 'params/image_height')
        width = get_val(writer_globals, 'params/image_width')

        # try to get the image_id, if not set it to zero
        image_id = get_val(writer_globals, 'params/image_id', default=0)

        # get the filename where the rgb image was stored
        file_name = get_val(
            writer_globals, 'writer_data/RGBWriter/rgb_filename')
        metadata = get_val(writer_globals, 'metadata')

        detectron2_file = f"{output_dir}/{file_prefix}detectron2.json"

        d2_annotations = []
        for obj_name, obj_data in metadata['objects'].items():
            if 'detectron2_annotation' in obj_data:
                d2_annotations.append(obj_data['detectron2_annotation'])

        d2_data = {'file_name': file_name,
                   'height': height,
                   'width': width,
                   'image_id': image_id,
                   'annotations': d2_annotations,
                   }

        with open(detectron2_file, 'w') as f:
            json.dump(d2_data, f, indent=4, sort_keys=True)
