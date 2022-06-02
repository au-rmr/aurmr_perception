# Standard Library
import copy

# Third Party
import numpy as np
import nvisii as visii

# First Party
from visii_tools.utils import vec_to_numpy, visii_transform_to_numpy
from visii_tools.writer.writer_interface import WriterInterface


class ObjectStateWriter(WriterInterface):
    """Writes the object state to 'metadata'

    """
    # location where data will be saved
    K_PREFIX = f'state/objects'

    def __init__(self, verbosity=0, cfg=None):
        WriterInterface.__init__(self, verbosity)

        if cfg is None:
            cfg = dict(visible_fraction_threshold=0.05,
                       export_detectron2=True)

        self._cfg = cfg

    def run(self, h5, cameras=None, pipeline=None, entity_list=None, **kwargs):
        """
        Export object data

        """

        if self._verbosity > 0:
            print("Running ObjectStateWriter")

        # create a group if it's not already there
        if not self.K_PREFIX in h5:
            h5_objects = h5.create_group(self.K_PREFIX)
        else:
            h5_objects = h5[self.K_PREFIX]

        for name in entity_list:
            # write the data for a single entity to the h5 file
            if self._verbosity > 0:
                print(f"Saving data for entity: {name}")
            obj = visii.entity.get(name)
            aabb_center = vec_to_numpy(obj.get_aabb_center())
            aabb_max = vec_to_numpy(obj.get_max_aabb_corner())
            aabb_min = vec_to_numpy(obj.get_min_aabb_corner())

            # information about mesh bounding box, etc.
            bbox3d = {'aabb_center': aabb_center,
                      'aabb_min': aabb_min,
                      'aabb_max': aabb_max,
                      #   'center': center,
                      }

            # segmentation id for this object
            seg_id = int(obj.get_id())

            # Note: This doesn't take into account any parent transforms
            world_t_object = visii_transform_to_numpy(visii.entity.get(
                name).get_transform(), include_affine=False)

            # create an h5 group for this object/entity
            h = h5_objects.create_group(name)
            h['transform'] = world_t_object
            h['segmentation_id'] = seg_id
            h['aabb_center'] = aabb_center
            h['aabb_min'] = aabb_min
            h['aabb_max'] = aabb_max
