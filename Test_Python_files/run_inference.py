import sys
sys.path.append(
    "/home/aurmr/workspaces/scenario_based_grasp_point_selection/src/segnetv2_mask2_former/UIE_main/mask2former_frame/")
from normal_std.inference import run_normal_std

object = run_normal_std()
object.inference()
