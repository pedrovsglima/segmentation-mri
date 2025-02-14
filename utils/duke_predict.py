import os
import sys
import argparse

import torchio as tio
from unet import UNet3D

sys.path.append(os.path.join(os.getcwd(), "duke_segmentation_code"))
from dataset_3d import Dataset3DSimple
from model_utils import pred_and_save_masks_3d_simple


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target-tissue", required=True, dest="target_tissue")
    parser.add_argument("--image", required=True, dest="image_dir")
    parser.add_argument("--save-masks-dir", required=True, dest="save_masks_dir")
    parser.add_argument("--model-save-path", required=False, dest="model_save_path")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    if args.target_tissue.lower() == "breast":
        n_channels = 1
        n_classes = 1
    else:
        raise ValueError("Unsupported target tissue. Only 'breast' is supported.")

    unet = UNet3D(
        in_channels = n_channels, 
        out_classes = n_classes,
        num_encoding_blocks = 3,
        padding = True,
        normalization = "batch"
    )

    # code for breast tissue
    input_dim = (144, 144, 96)

    transforms = tio.Compose([
        tio.Resize(input_dim)
    ])

    dataset = Dataset3DSimple(
        image_dir = args.image_dir,
        mask_dir = None,
        transforms = transforms,
        image_only = True
    )

    pred_and_save_masks_3d_simple(
        unet = unet,
        saved_model_path = args.model_save_path,
        dataset = dataset,
        n_classes = n_classes,
        n_channels = n_channels,
        save_masks_dir = args.save_masks_dir
    )
