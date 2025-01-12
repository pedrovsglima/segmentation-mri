#!/bin/bash

MAIN_PATH=$(grep '^MAIN_PATH=' .env | cut -d '=' -f2-)

PATIENT_ID="Breast_MRI_052"

IMAGE_PATH="$MAIN_PATH/nrrd_images/${PATIENT_ID}/post_1.nrrd"
BBOX_PATH="$MAIN_PATH/masks_threshold/${PATIENT_ID}.seg.nrrd"
MASK_PATH="$MAIN_PATH/nrrd_masks/${PATIENT_ID}.seg.nrrd"

MODE="cropped" # full, cropped

python3 utils/show_bbox_nrrd.py --mode "$MODE" --image-path "$IMAGE_PATH" --bbox-path "$BBOX_PATH" --mask-path "$MASK_PATH"
