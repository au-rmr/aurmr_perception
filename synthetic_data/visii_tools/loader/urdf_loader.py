

# Standard Library
import logging
import os
from pathlib import Path

# Third Party
import numpy as np
import nvisii
import trimesh
import urdfpy
from numpy.lib.arraysetops import isin
from urdfpy import URDF

# First Party
# visii_tools
from visii_tools import utils

log = logging.getLogger(__name__)


class URDFScene:
    """Class for managing a urdf scene in visii

    """

    def __init__(self,
                 urdf: URDF,
                 urdf_dir: str) -> None:
        """

        Args:
            urdf: URDF object
            dir: directory for resolving filepaths

        """

        self._urdf = urdf
        if urdf_dir is not None:
            self._urdf_dir = Path(urdf_dir)
        else:
            self._urdf_dir = Path("")

        # parse the links with visual components
        self._visuals = {}
        for link in self._urdf.links:
            vis_len = len(link.visuals)

            # DEBUG
            log.debug(
                f"link.name={link.name}, len(link.visuals)={len(link.visuals)}")
            if vis_len == 0:
                # skip links with no visual elements
                continue
            elif vis_len == 1:
                # set nvisii entity name to link name
                visual = link.visuals[0]
                self._visuals[link.name] = {'link': link,
                                            'visual': visual}
            else:
                # use naming convention f"{link_name}-{visual_name}
                for visual_idx, visual in enumerate(link.visuals):

                    if visual.name is not None and visual.name != "":
                        name = f"{link.name}-{visual.name}"
                    else:
                        name = f"{link.name}-visual_{visual_idx}"

                    # Add it to the dictionary
                    if name in self._visuals:
                        raise ValueError(
                            "An nvisii.entity with name {name} already exists, this would overwrite it. You are attempting to add a URDF visual element with link.name={link.name}")

                    self._visuals[name] = {'link': link,
                                           'visual': visual}

        # create nvisii entities
        self.create_nvisii_entities()

        # set pose
        self.set_transforms()

    def create_nvisii_entities(self):
        """Create nvisii entity for each <visual> in the URDF

        """
        for name, data in self._visuals.items():
            visual = data['visual']
            entity = self.create_nvisii_entity_from_urdf_visual(
                name, visual, self._urdf_dir)

            data['entity'] = entity

    def set_transforms(self, cfg=None):
        """Run forward kinematics, set the transforms
        appropriately

        - run visuals_fk
        - loop through self._visuals and grab the pose of each visual geometry
        - set the pose in visii

        """
        if cfg is None:
            cfg = {}

        fk = self._urdf.visual_geometry_fk(cfg=cfg)
        for name, data in self._visuals.items():

            # get the pose of the Geometry object
            transform = fk[data['visual'].geometry]

            # set the pose of the visii entity
            entity = data['entity']
            utils.set_transform_from_matrix(entity.get_transform(), transform)

    @staticmethod
    def create_nvisii_entity_from_urdf_visual(name, visual, dir):

        geom_element = visual.geometry.geometry

        # if isinstance(geom_element.meshes, NoneType):
        #     raise ValueError(f"geom_element.meshes is None, name={name}")

        # assert len(
        # geom_element.meshes) == 1, f"We only support Geometries with one mesh,
        # yours had {len(geom_element.meshes)}"

        scale = None
        if isinstance(geom_element, urdfpy.Mesh):
            assert len(
                geom_element.meshes) == 1, f"We only support Geometries with one mesh, yours had {len(geom_element.meshes)}"
            mesh_fname = str(dir / Path(visual.geometry.mesh.filename))
            mesh = nvisii.mesh.create_from_file(name, mesh_fname)
            scale = geom_element.scale
        elif isinstance(geom_element, urdfpy.Box):
            side_len = np.array(geom_element.size)
            half_side_len = side_len / 2.0
            mesh = nvisii.mesh.create_box(
                name, utils.to_visii_vec(half_side_len))
        elif isinstance(geom_element, urdfpy.Cylinder):
            half_cylinder_len = geom_element.length / 2.0
            mesh = nvisii.mesh.create_capped_cylinder(name,
                                                      radius=geom_element.radius, size=half_cylinder_len)
        elif isinstance(geom_element, urdfpy.Sphere):
            mesh = nvisii.mesh.create_icosphere(name,
                                                radius=geom_element.radius)
        else:
            raise ValueError(f"Unsupported geometry type {type(geom_element)}")

        entity = nvisii.entity.create(
            name=name,
            mesh=mesh,
            transform=nvisii.transform.create(name),
            material=nvisii.material.create(name),
        )

        if scale is not None:
            # consider using nvisii.transform.set_scale
            # https://nvisii.com/transform.html#nvisii.transform.set_scale
            entity.get_transform().set_scale(utils.to_visii_vec(scale))
            log.debug(f"Setting scale to {scale}")

            if np.linalg.norm(np.array(scale) - 1) > 1e-3:
                log.warning(
                    f"You added an object with non-identity scaling of {list(scale)}, things might break")

        # set some properties of the material
        material = entity.get_material()
        material.set_metallic(0)
        material.set_transmission(0)
        material.set_roughness(1)

        if visual.material is not None:
            if visual.material.texture is not None:
                # load the texture if it exists
                texture_fname = str(dir / Path(visual.material.texture.filename))
                log.debug(f"texture_fname: {texture_fname}")
                texture = nvisii.texture.create_from_file(name,
                                                          texture_fname)
                entity.get_material().set_base_color_texture(texture)
            elif visual.material.color is not None:
                # otherwise set the color
                entity.get_material().set_base_color(visual.material.color[:3])

        return entity

    @ staticmethod
    def load(fname):
        """Load a URDF scene from a file

        """
        urdf_dir = os.path.dirname(fname)
        urdf = URDF.load(fname)
        return URDFScene(urdf, urdf_dir)
