# Standard Library
import os
import subprocess
from pathlib import Path

BLENDER_EXEC = ["blender", "--background", "--python", ]
_package_dir = os.path.dirname(os.path.realpath(__file__))
BLENDER_SCRIPT = Path(_package_dir) / 'blender_texture_mapping.py'


def texture_map_scene(
        scene_dir,
        skip_prefixes=[],
        strip_materials=True,
        function_timeout=5,
        logging_level='WARN',
        return_output=False,
        capture_output=False):
    """Runs blender_texture_mapping script on the given scene.

    Args:
        skip_prefixes (list): Don't remap materials with these specific prefixes
    """
    subprocess_args = ['--function-timeout', str(function_timeout),
                       '--logging-level', logging_level,
                       ]
    if(strip_materials):
        subprocess_args.append('--strip-materials')

    if(len(skip_prefixes) > 0):
        subprocess_args.append('--skip-prefixes')
        for pre in skip_prefixes:
            subprocess_args.append(pre)

    try:
        p = subprocess.run([*BLENDER_EXEC, str(BLENDER_SCRIPT), '--', scene_dir,
                            *subprocess_args], capture_output=capture_output)
        # p = subprocess.Popen([*BLENDER_EXEC, str(BLENDER_SCRIPT), '--', scene_dir, *subprocess_args],
        #     shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True
        # )
    except subprocess.CalledProcessError as e:
        print(f'Error on file {str(scene_dir)}')
        print(str(e))
        return False

    if(return_output):
        return True, p.stdout
    return True
