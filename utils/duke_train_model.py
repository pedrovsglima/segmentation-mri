import os
import sys
import argparse

import torchio as tio
from unet import UNet3D

# sys.path.append(os.path.join(os.getcwd(), "duke_segmentation_code"))
from dataset_3d import Dataset3DSimple
from train import train_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target-tissue", required=True, dest="target_tissue")
    parser.add_argument("--train-image", required=True, dest="train_image_dir")
    parser.add_argument("--val-image", required=True, dest="val_image_dir")
    parser.add_argument("--train-mask", required=True, dest="train_mask_dir")
    parser.add_argument("--val-mask", required=True, dest="val_mask_dir")
    parser.add_argument("--epochs", required=True, type=int, default=10, dest="epochs")
    parser.add_argument("--batch-size", required=True, type=int, default=8, dest="batch_size")
    parser.add_argument("--model-save-dir", required=True, dest="model_save_dir")
    parser.add_argument("--model-save-name", required=True, dest="model_save_name")
    parser.add_argument("--load-model-path", required=False, dest="load_model_path")
    parser.add_argument("--num-workers", required=False, type=int, dest="num_workers")

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

    train_transforms = tio.Compose([
        tio.Resize(input_dim),
        tio.RandomBiasField(),
        tio.RandomMotion(degrees=5, translation=5, num_transforms=2),
        tio.RandomNoise(mean=0, std=(0.1, 0.5))
    ])

    val_transforms = tio.Compose([
        tio.Resize(input_dim)
    ])

    train_dataset = Dataset3DSimple(
        image_dir = args.train_image_dir,
        mask_dir = args.train_mask_dir,
        transforms = train_transforms
    )

    val_dataset = Dataset3DSimple(
        image_dir = args.val_image_dir,
        mask_dir = args.val_mask_dir,
        transforms = val_transforms
    )

    trained_unet = train_model(
        model = unet,
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        n_classes = n_channels,
        n_channels = n_classes,
        batch_size = args.batch_size,
        learning_rate = 3e-4,
        epochs = args.epochs,
        model_save_dir = args.model_save_dir,
        model_save_name = args.model_save_name,
        loss="cross",
        num_workers = args.num_workers,
        load_model_path = args.load_model_path
    )
