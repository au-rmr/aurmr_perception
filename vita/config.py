# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN

def add_maskformer2_frame_config(cfg):
    # Datset
    cfg.DATASETS.DATASET_RATIO = []

    # Training
    cfg.MODEL.FREEZE_WEIGHTS = []
    
    # Network
    cfg.MODEL.MASK_FORMER.MEMORY_FORWARD_STRATEGY = "top-k_lastest_frame:-1"
    cfg.MODEL.MASK_FORMER.TRACK_FEAT_TYPE = 'memory' # memory, zero, learned
    cfg.MODEL.MASK_FORMER.TRACK_POS_TYPE = 'zero' # memory, zero, learned
    cfg.MODEL.MASK_FORMER.REID_DIM = 256
    cfg.MODEL.MASK_FORMER.SELF_INIT_FRAME_ATTN = False
    
    # cfg.MODEL.MASK_FORMER.REID_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.REID_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.INV_REID_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.REID_COEFF = 0.1
    cfg.MODEL.MASK_FORMER.INV_REID_COEFF = 0.0
    cfg.MODEL.MASK_FORMER.REID_LOSS_OVER = 'all' # macthed, all
    cfg.MODEL.MASK_FORMER.MASK_USE_DIFFERENT_FEATURE = False
    cfg.MODEL.MASK_FORMER.IGNORE_FIRST_OUTPUT = False
    cfg.MODEL.MASK_FORMER.IGNORE_FIRST_REID = False
    cfg.MODEL.MASK_FORMER.USE_LAYER_ATTNMASK = True
    cfg.MODEL.MASK_FORMER.REID_FRAME_LOSS_TYPE = 'sigmoid' # softmax, sigmoid, contrastive_cosdist, contrastive_cossim
    cfg.MODEL.MASK_FORMER.REID_SOFTMAX_TEMPERATURE = 1.0
    cfg.MODEL.MASK_FORMER.FRAME_SELFMASK = False
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_ALPHA = 0.02
    cfg.MODEL.MASK_FORMER.CONTRASTIVE_SIGMA = 0.5
    cfg.MODEL.MASK_FORMER.FRAME_DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.ASSOC_DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.FEAT_DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.IGNORE_IOU_RANGE = [0.5, 0.5] # < first is negative, > second is positive, hungarian matched ones have value 1.0
    cfg.MODEL.MASK_FORMER.MAX_NUM_BG = 100
    cfg.MODEL.MASK_FORMER.MAX_NUM_FG = 100
    
    cfg.MODEL.REID = CN()
    cfg.MODEL.REID.ENCODER_TYPE = 'linear' # linear, mlp, 2frame-encoder
    cfg.MODEL.REID.REID_NORMALIZE = False # whether to normalize the reid embedding
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
    
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Inference
    cfg.MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD = 0.6
    
    # Multi-GPU Training
    cfg.SOLVER.NUM_GPUS = 1


def add_vita_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # VITA
    cfg.MODEL.VITA = CN()
    cfg.MODEL.VITA.NHEADS = 8
    cfg.MODEL.VITA.DROPOUT = 0.0
    cfg.MODEL.VITA.DIM_FEEDFORWARD = 2048
    cfg.MODEL.VITA.ENC_LAYERS = 6
    cfg.MODEL.VITA.DEC_LAYERS = 3
    cfg.MODEL.VITA.ENC_WINDOW_SIZE = 0
    cfg.MODEL.VITA.PRE_NORM = False
    cfg.MODEL.VITA.HIDDEN_DIM = 256
    cfg.MODEL.VITA.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.VITA.ENFORCE_INPUT_PROJ = True

    cfg.MODEL.VITA.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.VITA.DEEP_SUPERVISION = True
    cfg.MODEL.VITA.LAST_LAYER_NUM = 3
    cfg.MODEL.VITA.MULTI_CLS_ON = True
    cfg.MODEL.VITA.APPLY_CLS_THRES = 0.01

    cfg.MODEL.VITA.SIM_USE_CLIP = True
    cfg.MODEL.VITA.SIM_WEIGHT = 0.5

    cfg.MODEL.VITA.FREEZE_DETECTOR = False
    cfg.MODEL.VITA.TEST_RUN_CHUNK_SIZE = 18
    cfg.MODEL.VITA.TEST_INTERPOLATE_CHUNK_SIZE = 5

    cfg.LOAD_DATASET_INTO_MEMORY = False
