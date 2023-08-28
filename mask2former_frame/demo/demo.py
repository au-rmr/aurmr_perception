# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
# fmt: on

import tempfile
import time
import warnings
import json
from collections import defaultdict

import cv2
import numpy as np
import tqdm
import h5py
import pycocotools.mask as mask_util

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_frame import add_maskformer2_frame_config
from predictor import VisualizationDemo

from mask2former_group import (
    add_mask2former_group_config,
)

# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_frame_config(cfg)
    if "seq_frame_v2" in args['config_file']:
        add_mask2former_group_config(cfg)
    cfg.merge_from_file(args['config_file'])
    cfg.merge_from_list(args['opts'])
    cfg.freeze()
    return cfg


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    
    sequence = "/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/data/bin_3F/"
    file_list = os.listdir(sequence)
    image_list = [f for f in file_list if f.endswith('.png') or f.endswith('.jpg')]
    image_list.sort()
    frame_names = [os.path.join(sequence, f) for f in image_list]

    args = {'config_file': '/home/aurmr/workspaces/aurmr_demo_perception/src/UOIS_multi_frame/configs/amazon/seq_frame_v6_v90_grid_search_06_bin_16k.yaml',
            'sequence': '/home/aurmr/workspaces/test_ros_bare_bone/src/aurmr_perception/VITA_my/data/bin_3F/',
            'test_type': 'ytvis',
            'confidence_threshold': 0.5,
            'opts': ['MODEL.WEIGHTS', '/home/aurmr/Downloads/model_final(2).pth', 'TEST.DETECTIONS_PER_IMAGE', '100',
            'DATALOADER.NUM_WORKERS', '0', 'INPUT.SAMPLING_FRAME_NUM', '10', 'MODEL.REID.TEST_MATCH_THRESHOLD', '0.2', 'MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD', '0.6']}

    print(args['config_file'])

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, test_type=args['test_type'])

    if args['sequence']:
        sequence_imgs = []
        for frame_idx, frame_path in enumerate(frame_names):
            print(frame_path)
            # use PIL, to be consistent with evaluation
            img = read_image(frame_path, format="RGB")
            sequence_imgs.append(img)
        
        start_time = time.time()
        
        if(len(sequence_imgs) == 1):
            pred_masks, embedded_array = demo.run_on_sequence_single(sequence_imgs)
            full_mask = pred_masks.detach().cpu().numpy().astype(np.uint8)
            full_mask = np.zeros((400,400), dtype = np.uint8)
        else:
            pred_masks, embed = demo.run_on_sequence(sequence_imgs)
            # embeddings_array = []
            full_mask = np.zeros((400,400), dtype = np.uint8)
            if(len(pred_masks) > 0):
                seg_mask = pred_masks[0].detach().cpu().numpy().astype(np.uint8)
                full_mask = np.zeros(seg_mask.shape, dtype = seg_mask.dtype)
                for i in range(len(pred_masks)):
                    seg_mask = pred_masks[i].detach().cpu().numpy().astype(np.uint8)
                    # embed_mask = embeddings[i].detach().cpu().numpy()
                    full_mask += (i+1)*seg_mask
                    full_mask[full_mask > (i+1)] = i+1
                    # cv2.imwrite("/home/aurmr/workspaces/soofiyan_ws/src/segnetv2_mask2_former/Mask_Results/mask_full_post"+str(i)+".png", full_mask)
                    # embeddings_array.append(embed_mask)
            embedded_array = []
            for object_idx in np.unique(full_mask):
                if(object_idx != 0):
                    embedded_array.append(embed[object_idx-1][len(sequence_imgs)-1])
            embedded_array = np.stack(embedded_array)
                    
        print(np.unique(full_mask))
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, full_mask*50)
        cv2.waitKey(0)
