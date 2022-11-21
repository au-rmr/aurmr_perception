# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .mapper.amazon_syn_multi_frame_mapper import AmazonSynMultiFrameMapper

from .datasets.frame_amazon_syn_v1 import frameBinDataset
from .datasets import *

from .dataset_mapper import YTVISDatasetMapper, CocoClipDatasetMapper
from .build import *

from .ytvis_eval import YTVISEvaluator