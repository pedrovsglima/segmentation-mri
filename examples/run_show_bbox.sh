#!/bin/bash

MAIN_PATH=$(grep '^MAIN_PATH=' .env | cut -d '=' -f2-)

PATIENT_ID="Breast_MRI_052"

IMAGE_PATH="$MAIN_PATH/nrrd_images/${PATIENT_ID}/post_1.nrrd"
MASK_PATH="$MAIN_PATH/nrrd_masks/output_${PATIENT_ID}.seg.nrrd"

python3 utils/show_bbox_nrrd.py --image-path "$IMAGE_PATH" --mask-path "$MASK_PATH"
