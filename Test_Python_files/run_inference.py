import sys
sys.path.append(
    "/home/aurmr/workspaces/soofiyan_ws_force_calibration_suction_point/src/segnetv2_mask2_former/UIE_main/mask2former_frame/")
from normal_std.inference import run_normal_std

object = run_normal_std()
object.inference()
