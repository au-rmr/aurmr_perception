# Copyright (c) Facebook, Inc. and its affiliates.# model code
from . import modeling

# config
from .config import add_mask2former_group_config
#         update_dict[key.replace('transformer_self_attention_layers', 'transformer_frame_attention_layers')] = data['model'][key].copy()
