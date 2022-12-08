import sys
sys.path.append("/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/")
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
import matplotlib.pyplot as plt
import torch

sys.path.append("/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/mask2former_frame/demo")
from predictor import VisualizationDemo

WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_frame_config(cfg)
    cfg.merge_from_file("/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/configs/amazon/frame_maskformer2_v1_v14_c11_lr1e-5_no_clip_fix_query.yaml")
    cfg.MODEL.WEIGHTS = '/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/model_final.pth'
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/configs/amazon/frame_maskformer2_v1_v14_c11_lr1e-5_no_clip_fix_query.yaml",
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
        default=['MODEL.WEIGHTS', '/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/model_final.pth',
                 'DATALOADER.NUM_WORKERS', '0', 'INPUT.SAMPLING_FRAME_NUM', '1'],
        nargs=argparse.REMAINDER,
    )
    return parser


class SegnetV2():
    def __init__(self):
        print("init first")
        mp.set_start_method("spawn", force=True)
        
        print('*****************************************1')
        # self.args = get_parser().parse_args()
        print('*****************************************2')
        self.args = {'config_file': '/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/configs/amazon/frame_maskformer2_v1_v14_c11_lr1e-5_no_clip_fix_query.yaml',
                     'confidence_threshold': 0.5,
                     'opts': ['MODEL.WEIGHTS', '/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/UIE_main/model_final.pth',
                     'DATALOADER.NUM_WORKERS', '0', 'INPUT.SAMPLING_FRAME_NUM', '1']}
        # print(vars(self.args))
        self.cfg = setup_cfg(self.args)
        self.demo = VisualizationDemo(self.cfg)

    def mask_generator(self, input_image):
        img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        # img = read_image(input_image, format="BGR")
        predictions, visualized_output, masks = self.demo.run_on_image(
            img)
        # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        # cv2.waitKey(0)  # esc to quit
        full_mask = np.zeros((410,340), dtype = np.uint8)
        if(masks):
            full_mask = np.zeros(masks[0].shape, dtype = masks[0].dtype)
            counter = 1
            for i in masks:
                full_mask += i*counter
                full_mask[full_mask > counter] = counter
                # for i in range(full_mask.shape[0]):
                #     for j in range(full_mask.shape[1]):
                #         if(full_mask[i][j] > counter):
                #             full_mask[i][j] = counter
                cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/mask_full_post"+str(counter)+".png", full_mask)
                counter += 1
        # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # cv2.imshow(WINDOW_NAME, full_mask)
        # cv2.waitKey(0)  # esc to quit
        # plt.imshow(full_mask)
        # plt.title("full mask")
        # plt.show()
        return masks, full_mask


# object = SegnetV2()
# object.mask_generator("/home/aurmr/workspaces/soofiyan_ws/src/aurmr_perception/UIE_main/annotated_real_v1_resized/images/scene_03/bin_1E/bin_1E_color_0006.png")
