#!/bin/bash

# custom config
DATA=data
TRAINER=ZeroshotCLIP2
DATASET=cifar10
CFG=rn50  # rn50, rn101, vit_b32 or vit_b16
export CUDA_VISIBLE_DEVICES=1,2,3

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only