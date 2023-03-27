# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
print(os.path.join(sys.path[0], '../..'))
# fmt: on


import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former_frame import add_maskformer2_frame_config
from predictor import VisualizationDemo

from mask2former_group import (
    add_mask2former_group_config,
)
import cv2

class uois_single_frame():
    def __init__(self):
        mp.set_start_method("spawn", force=True)

        self.args = {'config_file': '/home/aurmr/workspaces/single_frame_embeddings_ws/src/UOIS-single-frame/configs/amazon/seq_frame_v6_v92_grid_search_06_bin_no_frame.yaml',
                'test_type': 'ytvis',
                'confidence_threshold': 0.5,
                'opts': ['MODEL.WEIGHTS', '/home/aurmr/workspaces/single_frame_embeddings_ws/src/UOIS-single-frame/model/model_final.pth', 
                'DATALOADER.NUM_WORKERS', '0', 'INPUT.SAMPLING_FRAME_NUM', '1', 'MODEL.REID.TEST_MATCH_THRESHOLD', '0.2', 'MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD', '0.6']}

        self.cfg = self.setup_cfg(self.args)

        self.demo = VisualizationDemo(self.cfg, test_type=self.args["test_type"])

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_maskformer2_frame_config(cfg)
        if "seq_frame_v2" in args["config_file"]:
            add_mask2former_group_config(cfg)
        cfg.merge_from_file(args["config_file"])
        cfg.merge_from_list(args["opts"])
        cfg.freeze()
        return cfg

    def inference(self, img):
        sequence_img = []
        sequence_img.append(img)

        pred_masks, embeddings = self.demo.run_on_sequence_ytvis(sequence_img)

        embeddings_array = []
        full_mask = np.zeros((400,400), dtype = np.uint8)
        if(len(pred_masks) > 0):
            seg_mask = pred_masks[0].detach().cpu().numpy().astype(np.uint8)
            full_mask = np.zeros(seg_mask.shape, dtype = seg_mask.dtype)
            for i in range(len(pred_masks)):
                seg_mask = pred_masks[i].detach().cpu().numpy().astype(np.uint8)
                embed_mask = embeddings[i].detach().cpu().numpy()
                full_mask += (i+1)*seg_mask
                full_mask[full_mask > (i+1)] = i+1
                # cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/mask_full_post"+str(i)+".png", full_mask)
                embeddings_array.append(embed_mask)

        return full_mask, embeddings_array

object = uois_single_frame()
img = cv2.imread("/home/aurmr/workspaces/single_frame_embeddings_ws/src/UOIS-single-frame/datasets/bin_4G_color_0005.png")
full_mask, embeddings_array = object.inference(img)
print(embeddings_array)
