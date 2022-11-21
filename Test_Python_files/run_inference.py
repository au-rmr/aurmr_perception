import sys
sys.path.insert(
    1, "/home/soofiyanatar/Documents/AmazonHUB/UIE-main/mask2former_frame/demo/normal_std")

from inference import run_normal_std

object = run_normal_std()
object.inference()
