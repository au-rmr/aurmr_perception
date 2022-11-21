import sys
sys.path.append(
    "/home/aurmr/workspaces/segnetv2_normal_grasp/aurmr_perception/UIE_main/mask2former_frame/")
from normal_std.inference import run_normal_std

object = run_normal_std()
object.inference()
