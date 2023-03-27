#!/bin/bash

BASEDIR=$(dirname $0)
BASEDIR=$(readlink -f ${BASEDIR})
OUTPUTDIR=${BASEDIR}/output


BATCH_FILE=$(mktemp --suffix=.job)
cat > ${BATCH_FILE} <<EOF
#!/bin/bash -x
#SBATCH --job-name=test_$(echo $1 | cut -d "_" -f 4)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a40
#SBATCH --account=${2}
python train_net_frame.py --config-file ./configs/amazon/${1}.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ./output/${1}/model_final.pth OUTPUT_DIR ./output/${cfg_name} INPUT.SAMPLING_FRAME_NUM 15 SOLVER.NUM_GPUS 1 MODEL.MASK_FORMER.TEST.INFERENCE_THRESHOLD 0.6 TEST.DETECTIONS_PER_IMAGE 20 MODEL.REID.TEST_MATCH_THRESHOLD 0.2
exit 0

EOF

sbatch ${BATCH_FILE}
