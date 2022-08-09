loss="sat"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python imagenet_main_sota.py \
  --data "/home/nicolas/data/imagenet" \
  --save "/datadrive1/adam/imagenet-${loss}" \
  --resume "/datadrive1/adam/imagenet-checkpoints/checkpoint-epoch-115.pth.tar" \
  --loss ${loss} \
  >>train_imagenet_main_${timestamp}_${loss}.txt 2>&1 &
echo train_imagenet_main_${timestamp}_${loss}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
ARCH=vgg16_bn
LOSS=sat
DATASET=cifar10
PRETRAIN=150
MOM=0.9
SAVE_DIR='/datadrive1/adam/'${DATASET}_${ARCH}_${LOSS}
GPU_ID=0
mkdir -p ${SAVE_DIR}
### train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train_sota.py --arch ${ARCH} --gpu-id ${GPU_ID} \
  --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
  --loss ${LOSS} \
  --dataset ${DATASET} --save ${SAVE_DIR} \
  --manualSeed 1 >> ${SAVE_DIR}-${timestamp}.log 2>&1 &
echo ${SAVE_DIR}-${timestamp}.log
[1] 2315814                                                                                                                                                                │
(python39) ady@equus:~/code2/SAT-selective-cls$ echo ${SAVE_DIR}-${timestamp}.log                                                                                          │
/datadrive1/adam/cifar10_vgg16_bn_sat-2022-08-09-14-41-46-091583234.log