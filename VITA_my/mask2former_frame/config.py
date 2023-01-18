# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_frame_config(cfg):
    # Network
    cfg.MODEL.MASK_FORMER.MEMORY_FORWARD_STRATEGY = "top-k_lastest_frame:-1"
    cfg.MODEL.MASK_FORMER.TRACK_FEAT_TYPE = 'memory' # memory, zero, learned
    cfg.MODEL.MASK_FORMER.TRACK_POS_TYPE = 'zero' # memory, zero, learned
    cfg.MODEL.MASK_FORMER.REID_DIM = 256
    
    # cfg.MODEL.MASK_FORMER.REID_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.REID_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.INV_REID_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.REID_COEFF = 0.1
    cfg.MODEL.MASK_FORMER.INV_REID_COEFF = 0.0
    cfg.MODEL.MASK_FORMER.REID_LOSS_OVER = 'all' # macthed, all
    
    cfg.MODEL.REID = CN()
    cfg.MODEL.REID.ENCODER_TYPE = 'linear' # linear, mlp, 2frame-encoder
    cfg.MODEL.REID.REID_NORMALIZE = True # whether to normalize the reid embedding
    cfg.MODEL.REID.ASSOCIATOR = 'hungarian' # hungarian, decoder
    cfg.MODEL.REID.ENCODER_POS_TYPE = 'zero' # zero, fixed, learned, external
    cfg.MODEL.REID.INTERACTION_OVER = 'all' # matched, all
    cfg.MODEL.REID.ENCODER_DIM_FEEDFORWARD = 2048
    cfg.MODEL.REID.ENCODER_NUM_LAYERS = 1
    cfg.MODEL.REID.ENCODER_LAYER_STRUCTURE = ['full'] # self, cross, frame, ffn
    cfg.MODEL.REID.ENCODER_ACTIVATION = 'relu'
    cfg.MODEL.REID.ENCODER_NORMALIZE_BEFORE = False
    cfg.MODEL.REID.ENCODER_ONE_LAYER_AFTER_ATTENTION = False
    cfg.MODEL.REID.ENCODER_LAST_LAYER_BIAS = True
    cfg.MODEL.REID.ENCODER_UPDATE_CONFIDENCE = False
    cfg.MODEL.REID.ENCODER_PREDICT_MASK = False
    cfg.MODEL.REID.ENCODER_NORM_BEFORE_HEAD = False
    
    cfg.MODEL.REID.ASSOCIATOR_LAYER_STRUCTURE = []
    cfg.MODEL.REID.ASSOCIATOR_LAYER_POSITION = 'none'
    
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
    
