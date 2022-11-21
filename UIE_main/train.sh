#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHONPATH=$PYTHONPATH:$PWD

cfg_name=$1
# echo "cfg_name: $cfg_name"
python ./train_net_frame.py \
  --config-file configs/amazon/${cfg_name}.yaml \
  --num-gpus 1 \
  OUTPUT_DIR ./output/${cfg_name}