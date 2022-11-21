import sys
sys.path.append(
    "/home/soofiyanatar/Documents/AmazonHUB/UIE_main/mask2former_frame/")
from normal_std.inference import run_normal_std

object = run_normal_std()
object.inference()
