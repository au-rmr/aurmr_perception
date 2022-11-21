# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_frame_config(cfg):
    # Network
    cfg.MODEL.MASK_FORMER.MEMORY_FORWARD_STRATEGY = "top-k_lastest_frame:-1"
    cfg.MODEL.MASK_FORMER.REID_DIM = 256
    # cfg.MODEL.MASK_FORMER.REID_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.REID_POSITIVE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.REID_NEGATIVE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.REID_FG_RATIO = 0.25
    
    # Input
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
