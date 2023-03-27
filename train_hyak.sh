#!/bin/bash

BASEDIR=$(dirname $0)
BASEDIR=$(readlink -f ${BASEDIR})
OUTPUTDIR=${BASEDIR}/output

mkdir -p $HOME/jobs

BATCH_FILE=$(mktemp -p $HOME/jobs --suffix=.job)

cat > ${BATCH_FILE} <<EOF
#!/bin/bash -x
#SBATCH --job-name=$(echo $1 | cut -d "_" -f 4)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=18:00:00
#SBATCH --gpus=1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a40
#SBATCH --account=${2}
python ./train_net_frame.py --config-file ./configs/amazon/${1}.yaml --num-gpus 1 OUTPUT_DIR ./output/${1} LOAD_DATASET_INTO_MEMORY True
python train_net_frame.py --config-file ./configs/amazon/${1}.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ./output/${1}/model_final.pth OUTPUT_DIR ./output/${cfg_name} INPUT.SAMPLING_FRAME_NUM 15 SOLVER.NUM_GPUS 1 MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD 0.6 MODEL.WEIGHTS ./output/${1}/model_final.pth TEST.DETECTIONS_PER_IMAGE 20 MODEL.REID.TEST_MATCH_THRESHOLD 0.2
exit 0

EOF

sbatch ${BATCH_FILE}
