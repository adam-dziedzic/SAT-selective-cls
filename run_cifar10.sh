#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2
#SBATCH --job-name=train-cifar10-vgg16_bn_ce_loss
#SBATCH --output=train-imagenet.out
#SBATCH --array=0-1
# everything below this line is optional, but are nice to have quality of life things
#SBATCH --output=job.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out
#SBATCH --error=job.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err

# source /ssd003/home/ady/.bashrc
# conda activate /ssd003/home/ady/envnew
source /h/ady/.envnew

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

ARCH=vgg16_bn
LOSS=ce
DATASET=cifar10
PRETRAIN=0
MOM=0.9
SAVE_DIR='/ssd003/home/ady/'${DATASET}_${ARCH}_${LOSS}
GPU_ID=0

mkdir -p ${SAVE_DIR}

### train
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} \
--pretrain ${PRETRAIN} --sat-momentum ${MOM} \
--loss ${LOSS} \
--dataset ${DATASET} --save ${SAVE_DIR} \
--manualSeed $SLURM_ARRAY_TASK_ID \
2>&1 | tee -a ${SAVE_DIR}.log

### eval
python -u train.py --arch ${ARCH} --gpu-id ${GPU_ID} \
--loss ${LOSS} --dataset ${DATASET} \
--save ${SAVE_DIR} --evaluate \
2>&1 | tee -a ${SAVE_DIR}.log
