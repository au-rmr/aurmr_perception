# Standard Library
import copy
import json
import logging
import os
import warnings

# Third Party
# import cv2
# import h5py
# import json_tricks
# import numpy as np
# import pycocotools.mask
# import transforms3d

log = logging.Logger(__name__)


class WriterPipeline:
    """Class that runs the writer pipeline
    """

    def __init__(self, h5, output_dir=None) -> None:
        self._h5 = h5
        self._output_dir = output_dir

    @property
    def h5(self):
        return self._h5

    def run_writer_pipeline(self,
                            cameras: dict,
                            entity_list,
                            writer_object_list: list = None,
                            idx: int = None,
                            h5=None,  # optional h5 object to write to
                            verbosity=0,
                            ):
        """Runs the writer pipeline

        """

        if h5 is None:
            assert idx is not None
            h5 = self.h5.create_group(str(idx))

        for writer in writer_object_list:
            log.debug(f"Running writer of type {type(writer)}")
            writer.run(h5=h5,
                       cameras=cameras,
                       entity_list=entity_list,
                       idx=idx,
                       output_dir=self._output_dir,
                       )


def create_object_data(unique_name,
                       class_name="",
                       name="",
                       instance_num=0,
                       category_id=0,  # used for exporting to Detectron2 format
                       ):
    """Create a dictionary for storing object data

    Return:
        object_data: dict

    """
    object_data = {
        'unique_name': unique_name,  # rename this to uuid?
        'class': class_name,
        'name': name,
        'instance_num': instance_num,
        'category_id': category_id,
    }
    return object_data
