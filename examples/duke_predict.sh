#!/bin/bash

MAIN_PATH=$(grep "^LOCAL_PATH=" .env | cut -d "=" -f2-)
SAVE_PATH=$(grep "^GIT_PATH=" .env | cut -d "=" -f2-)

python3 utils/duke_predict.py \
    --target-tissue breast \
    --image "${MAIN_PATH}/mri_npy/test" \
    --save-masks-dir "${MAIN_PATH}/breast_mask_duke_pred" \
    --model-save-path "${SAVE_PATH}//trained_models/breast_model_finetuned.pth"
