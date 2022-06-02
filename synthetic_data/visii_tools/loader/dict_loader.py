
# from omegaconf import OmegaConf

from visii_tools.loader.object_loader import ObjectLoader
from visii_tools import utils
import numpy as np


class DictLoader:
    """Helper class for loading a scene with description from dict."""

    def __init__(self, cfg, mesh_root_path) -> None:

        self.cfg = cfg

        self._entity_list = []
        self.mesh_root_path = mesh_root_path
        self.create_entities(self.cfg, self.mesh_root_path)

    @property
    def entity_list(self):
        return self._entity_list

    def create_entities(self, cfg, mesh_root_path):
        for name, d in cfg['objects'].items():
            entity = ObjectLoader.load_object_mesh(name, d['entity'], set_material=True, mesh_root_path=mesh_root_path)

            # set the transform of this object
            if type(d['transform'])==np.ndarray:
                T = d['transform']
                scale = np.array([1,1,1])
            else:
                T = d['transform']['transform']
                scale = d['transform']['scale'] if 'scale' in d['transform'] else None
            utils.set_transform_from_matrix(entity.get_transform(), T=T, scale=scale)

            # set the metadata if it exists
            entity.metadata = d['metadata'] if 'metadata' in d else None

            if d['metadata'] is None or 'export' not in d['metadata']:
                export = True
            else:
                export = d['metadata']

            if export:
                self._entity_list.append(name)
