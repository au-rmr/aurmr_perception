
from omegaconf import OmegaConf

from visii_tools import utils


class AcronymLoader:
    """Helper class for loading a scene with description from dict."""

    def __init__(self, cfg) -> None:

        self.cfg = cfg

        # convert it to an OmegaConf object if it isn't already
        if isinstance(self.cfg, dict):
            self.cfg = OmegaConf.create(self.cfg)

        # set it to be read only so we don't modify it by accident
        OmegaConf.set_readonly(self.cfg, True)
        self._entity_list = []
        self.create_entities(self.cfg)

    @property
    def entity_list(self):
        return self._entity_list

    def create_entities(self, cfg):
        for name, d in cfg['objects'].items():
            entity = ObjectLoader.load_object_mesh(name, d['entity'], set_material=True)

            # set the transform of this object
            utils.set_transform(entity.get_transform(), **d['transform'])

            # set the metadata if it exists
            entity.metadata = d.get('metadata', None)

            if d.metadata is None:
                export = True
            else:
                export = d.metadata.get('export', True)

            if export:
                self._entity_list.append(name)

class ObjectLoader:

    @staticmethod
    def load_object_mesh(unique_name: str,
                         config: dict,
                         set_material: bool = True):
        """Create an envisii entity from the config.

        Args:
            set_material: If true set the material as well

        """

        if config['type'] == "MESH":

            mesh_file = config['file']
            # visii Entity object
            mesh = visii.mesh.create_from_file(unique_name, mesh_file)

            entity = visii.entity.create(
                name=unique_name,
                mesh=mesh,
                transform=visii.transform.create(unique_name),
                material=visii.material.create(unique_name),
            )

        else:
            raise ValueError(f"We don't support loading objects of type {config['type']}")

        if set_material and 'material' in config:
            update_single_material(mat=entity.get_material(),
                                   name_prefix=unique_name,
                                   **config['material'])

        return entity