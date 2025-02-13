#!/bin/bash

MAIN_PATH=$(grep "^LOCAL_PATH=" .env | cut -d "=" -f2-)
SAVE_PATH=$(grep "^GIT_PATH=" .env | cut -d "=" -f2-)

python3 duke_train_model.py \
    --target-tissue breast \
    --train-image "${MAIN_PATH}/data/mri_data/train" \
    --val-image "${MAIN_PATH}/data/mri_data/val" \
    --train-mask "${MAIN_PATH}/data/segmentations/train" \
    --val-mask "${MAIN_PATH}/data/segmentations/val" \
    --epochs 20 \
    --batch-size 16 \
    --num-workers 8 \
    --model-save-dir "${SAVE_PATH}/trained_models" \
    --model-save-name "breast_model_finetuned" \
    --load-model-path "${SAVE_PATH}/trained_models/breast_model.pth"
