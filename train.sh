#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHONPATH=$PYTHONPATH:$PWD

cfg_name=$1
# echo "cfg_name: $cfg_name"
# python ./train_net_frame.py \
#   --config-file configs/amazon/${cfg_name}.yaml \
#   --num-gpus 1 \
#   OUTPUT_DIR ./output/${cfg_name} \
#   LOAD_DATASET_INTO_MEMORY True
python ./train_net_frame.py --config-file ./configs/amazon/${1}.yaml --num-gpus 1 OUTPUT_DIR ./output/${1} LOAD_DATASET_INTO_MEMORY True
python train_net_frame.py --config-file ./configs/amazon/${1}.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ./output/${1}/model_final.pth OUTPUT_DIR ./output/${cfg_name} INPUT.SAMPLING_FRAME_NUM 15 SOLVER.NUM_GPUS 1 MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD 0.6 MODEL.WEIGHTS ./output/${1}/model_final.pth TEST.DETECTIONS_PER_IMAGE 20
