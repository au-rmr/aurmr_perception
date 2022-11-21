import sys

from mask2former_frame import add_maskformer2_frame_config
from mask2former import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
import tqdm
import numpy as np
import cv2
import warnings
import time
import tempfile
import argparse
import glob
import multiprocessing as mp
import os

from demo.predictor import VisualizationDemo

WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_frame_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/soofiyanatar/Documents/AmazonHUB/UIE_main/configs/amazon/frame_maskformer2_v1_v14_c11_lr1e-5_no_clip_fix_query.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', '/home/soofiyanatar/Documents/AmazonHUB/UIE_main/model_final.pth',
                 'DATALOADER.NUM_WORKERS', '0', 'INPUT.SAMPLING_FRAME_NUM', '1'],
        nargs=argparse.REMAINDER,
    )
    return parser


class SegnetV2():
    def __init__(self):
        mp.set_start_method("spawn", force=True)
        self.args = get_parser().parse_args()
        self.cfg = setup_cfg(self.args)
        self.demo = VisualizationDemo(self.cfg)

    def mask_generator(self, input_image):
        img = read_image(input_image, format="BGR")
        predictions, visualized_output, masks = self.demo.run_on_image(
            img)
        for i in masks:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, i)
            cv2.waitKey(0)  # esc to quit
        return masks


# SegnetV2("/home/soofiyanatar/Documents/AmazonHUB/UIE-main/annotated_real_v1_resized/images/scene_03/bin_1E/bin_1E_color_0006.png")
