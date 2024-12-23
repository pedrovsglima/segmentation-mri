#!/bin/bash

MAIN_PATH=$(grep '^MAIN_PATH=' .env | cut -d '=' -f2-)

DICOM_PATH="$MAIN_PATH/MRI_SEG_DICOM/Breast_MRI_086/01-01-1990-NA-BREASTROUTINE-86704/5.000000-ax dyn 1st pass-39884/"
ANNOTATION_PATH="$MAIN_PATH/Supplemental-Data/Annotation_Boxes.xlsx"

python3 utils/show_dicom.py --data-path "$DICOM_PATH" --annotation-file "$ANNOTATION_PATH"
