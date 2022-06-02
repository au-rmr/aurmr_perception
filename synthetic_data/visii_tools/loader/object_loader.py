# Third Party
import nvisii as visii
import os

# First Party
from visii_tools import utils
from visii_tools.loader.material_loader import update_single_material


class ObjectLoader:

    @staticmethod
    def load_object_mesh(unique_name: str,
                         config: dict,
                         set_material: bool = True,
                         mesh_root_path: str = None):
        """Create an envisii entity from the config.

        Args:
            set_material: If true set the material as well

        """

        if config['type'] == "MESH":
            if 'file' in config:
                mesh_file = config['file']
                # visii Entity object
                mesh = visii.mesh.create_from_file(unique_name, os.path.join(mesh_root_path, mesh_file))
            elif 'data' in config:
                mesh = visii.mesh.create_from_data(name=unique_name,
                                                   **config['data'])

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

