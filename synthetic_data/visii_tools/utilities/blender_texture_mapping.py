"""Blender texture mapping"""

# Standard Library
import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# Third Party
import bpy

MAX_SIZE = 5e7


class TimeOutException(Exception):
    def __init__(self, *args, **kwargs):
        super(TimeOutException, self).__init__(*args, **kwargs)
        pass


def signal_handler(signum, frame):
    raise TimeOutException("Function Call Timed Out")

# This is based on Keunhong Park's code.
# /snap/bin/blender --background --python uv_mapping.py --  /tmp/kitchen/


parser = argparse.ArgumentParser()
parser.add_argument(dest='obj_dir', type=Path)
parser.add_argument('--strip-materials', action='store_true')
parser.add_argument('--function-timeout', type=int, default=0)
parser.add_argument('--skip-prefixes', type=str, nargs='+', default=[])
parser.add_argument('--logging-level', type=str, default='INFO')


def main():
    """Use blender to add a uv map to a mesh so we can apply textures to it.

    This will iterate through all .obj files in the specified directory and add
    uv maps to them.
    """
    # Drop blender arguments.
    argv = sys.argv[5:]
    opt = parser.parse_args(args=argv)

    logging.basicConfig(level=opt.logging_level)
    log = logging.getLogger(__name__)

    paths = sorted(opt.obj_dir.glob('**/*.obj'))
    log.info(f'Processing {len(paths)} files')
    signal.signal(signal.SIGALRM, signal_handler)

    for i, path in enumerate(paths):
        signal.alarm(opt.function_timeout)
        try:
            model_size = os.path.getsize(path)
            if model_size > MAX_SIZE:
                log.error(f'Model too big ({model_size} > {MAX_SIZE})')
                continue

            obj_filename = path.name
            skip_file = False
            for pre in opt.skip_prefixes:
                if(obj_filename.startswith(pre)):
                    skip_file = True
                    break
            if(skip_file):
                continue

            log.info(f'Processing {path!s}')
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.obj(filepath=str(path),
                                     use_edges=True,
                                     use_smooth_groups=True,
                                     use_split_objects=True,
                                     use_split_groups=True,
                                     use_groups_as_vgroups=False,
                                     use_image_search=True)

            if len(bpy.data.objects) > 10:
                log.error("Too many objects. Skipping for now..")
                continue

            if opt.strip_materials:
                log.info("Deleting materials.")
                for material in bpy.data.materials:
                    material.user_clear()
                    bpy.data.materials.remove(material)

            for obj_idx, obj in enumerate(bpy.data.objects):
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                log.info("Clearing split normals and removing doubles.")
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
                bpy.ops.mesh.remove_doubles()
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.mesh.mark_sharp(use_verts=True)
                bpy.ops.mesh.faces_shade_flat()

                log.info("Unchecking auto_smooth")
                obj.data.use_auto_smooth = False

                bpy.ops.object.modifier_add(type='EDGE_SPLIT')
                log.info("Adding edge split modifier.")
                mod = obj.modifiers['EdgeSplit']
                mod.split_angle = 20

                bpy.ops.object.mode_set(mode='OBJECT')

                log.info("Applying smooth shading.")
                bpy.ops.object.shade_smooth()

                bpy.ops.object.editmode_toggle()
                bpy.ops.mesh.select_all(action='SELECT')
                log.info("Running smart UV project.")
                bpy.ops.uv.smart_project()

                bpy.context.active_object.select_set(state=False)

            log.info(f'Saved to {path}')
            bpy.ops.export_scene.obj(filepath=str(path),
                                     group_by_material=True,
                                     keep_vertex_order=True,
                                     use_normals=True, use_uvs=True,
                                     use_materials=True,
                                     check_existing=False)
        except (TimeOutException, RuntimeError) as err:
            log.error(f'Error at File Index {i}, object index {obj_idx}: {path!s}')
            log.error(err)

        finally:
            signal.alarm(0)
    logging.info('Completed UV Mapping of Scene')


if __name__ == '__main__':
    main()
