
# Standard Library
import glob
import logging
import pathlib
import re
from typing import Tuple

# Third Party
import nvisii

log = logging.getLogger(__name__)


def update_single_material(mat,
                           name_prefix,
                           color=None,
                           metallic=None,
                           roughness=None,
                           transmission=None,
                           specular=None,
                           color_texture_fname=None,
                           roughness_texture_fname=None,
                           normal_map_texture_fname=None,
                           transmission_texture_fname=None,
                           metallic_texture_fname=None,

                           ):
    """Set properties for a material.

    Sets some default values for metallic, roughness and transmission
    Args:
        mat: visii.material

    """

    if color is not None:
        mat.set_base_color(color)

    if metallic is not None:
        mat.set_metallic(metallic)

    if roughness is not None:
        mat.set_roughness(roughness)

    if transmission is not None:
        mat.set_transmission(transmission)

    if specular is not None:
        specular = mat.set_specular(specular)

    if color_texture_fname is not None:
        name = f"{name_prefix}_color"
        texture = nvisii.texture.create_from_file(name, color_texture_fname)
        mat.set_base_color_texture(texture)

    if roughness_texture_fname is not None:
        name = f"{name_prefix}_roughness"
        texture = nvisii.texture.create_from_file(
            name,
            roughness_texture_fname)
        mat.set_roughness_texture(texture)

    if normal_map_texture_fname is not None:
        name = f"{name_prefix}_normal_map"
        texture = nvisii.texture.create_from_file(
            name,
            normal_map_texture_fname)
        mat.set_normal_map_texture(texture)

    if transmission_texture_fname is not None:
        name = f"{name_prefix}_transmission"
        texture = nvisii.texture.create_from_file(
            name,
            transmission_texture_fname)
        mat.set_transmission_texture(texture)

    if metallic_texture_fname is not None:
        name = f"{name_prefix}_metallic"
        texture = nvisii.texture.create_from_file(
            name,
            transmission_texture_fname)
        mat.set_metallic_texture(texture)


def update_materials(materials_data: dict):
    """Update materials with the information in the dictionary

    Args:
        materials_data: dict

        key: entity_name
        val:
            - color_texture_fname: path to color texture file
            - roughness_texture_fname: path to roughness texture file
            - normal_map_texture_fname: path to normal map texture file
            - metallic: float
            - roughness: float
            - transmission: float

    """

    for name, data in materials_data.items():
        entity = nvisii.entity.get(name)
        mat = entity.get_material()
        update_single_material(mat,
                               name_prefix=name,
                               **data,
                               )


def check_cco_texture(texture_dir: str) -> Tuple:
    """Check if a cco texture has all correct files.

    Checks if color, roughness and normal files exist

    Returns:
        tuple: (bool, dict)
            bool: True if all necessary files exist
            file_dict: dict pointing to the files
    """
    rgb_files = glob.glob(f"{texture_dir}/*_Color.jpg")
    roughness_files = glob.glob(f"{texture_dir}/*_Roughness.jpg")
    #TODO what's the difference between normalDX and normalGL
    normal_files = glob.glob(f"{texture_dir}/*_NormalGL.jpg")
    displacement_files = glob.glob(f"{texture_dir}/*_Displacement.jpg")

    if rgb_files and roughness_files and normal_files and displacement_files:
        file_dict = {'color': rgb_files[0],
                     'roughness': roughness_files[0],
                     'normal': normal_files[0],
                     'displacement': displacement_files[0],
                     }
        return True, file_dict
    else:
        return False, None


def load_cco_texture(mat: nvisii.material,
                     texture_dir: str) -> bool:
    """Load a cco texture into the given material.

    Returns:
        True if successful, False otherwise
    """

    valid, files_dict = check_cco_texture(texture_dir)
    if not valid:
        log.warning("Invalid texture, skipping")
        return False
    mat_name = mat.get_name()
    rgb_tex = nvisii.texture.create_from_file(f"{mat_name}_color",
                                              files_dict['color'],
                                              )
    roughness_tex = nvisii.texture.create_from_file(f"{mat_name}_roughness",
                                                    files_dict['roughness'],
                                                    )
    normal_tex = nvisii.texture.create_from_file(f"{mat_name}_normal",
                                                 files_dict['normal'],
                                                 )

    # print(files_dict['color'])
    mat.set_base_color_texture(rgb_tex)
    mat.set_roughness_texture(roughness_tex)
    mat.set_normal_map_texture(normal_tex)

    return True

def find_cco_textures(dir: str) -> dict:
    """Find all cco textures, return a dict organized by type.

    Assumes that the textures folders are those gotten using the
    cco_textures_download.py script. Folder names are of the form

    Asphalt001_2K-JPG

    Which would have category 'Asphalt'


    Returns:
        textures: dict
            - keys: ['Bark', 'Bricks', etc.]
            - values: List[str] where each list entry is a fullpath to folder where textures are stored

    """
    texture_folders = glob.glob(dir + '/*')
    textures = {}
    for texture_folder in texture_folders:
        # example texture folder name 'Bark008_2K-JPG'
        f = pathlib.Path(texture_folder)
        texture_name = f.name
        valid, _ = check_cco_texture(f)
        if(re.search('[a-zA-Z]+[0-9]{3}_2K-JPG', texture_name)) and valid:
            # e.g. `Bark, Bricks, etc.`
            texture_category = re.match('([a-zA-Z]+)', texture_name).group()

            # check if it is a valid cco texture

            # create an entry if it doesn't exist
            if(texture_category not in textures):
                textures[texture_category] = []
            textures[texture_category].append(f)

    if len(textures) == 0:
        log.warning(f"No textures found in {dir}")

    return textures
