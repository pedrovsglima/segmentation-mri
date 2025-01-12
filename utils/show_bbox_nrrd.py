import argparse
import numpy as np
import napari
import SimpleITK as sitk


def extract_bounding_box(seg_array):
    coords = np.argwhere(seg_array > 0)
    start = coords.min(axis=0)
    end = coords.max(axis=0) + 1  # include the last index
    return start, end

def visualize_3d_with_napari(image_path, bbox_path, mask_path, mode):

    image = sitk.ReadImage(image_path)
    bbox = sitk.ReadImage(bbox_path)
    mask = sitk.ReadImage(mask_path)

    image_array = sitk.GetArrayFromImage(image)
    bbox_array = sitk.GetArrayFromImage(bbox)
    mask_array = sitk.GetArrayFromImage(mask)

    assert image_array.shape == bbox_array.shape, "Image and bounding box dimensions must match!"
    assert image_array.shape == mask_array.shape, "Image and mask dimensions must match!"

    if mode == "cropped":
        start, end = extract_bounding_box(mask_array)
        image_array = image_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        bbox_array = bbox_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        mask_array = mask_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    viewer = napari.Viewer(ndisplay=3)  # 3D mode
    viewer.add_image(image_array, name="MRI", colormap="gray", rendering="mip")
    viewer.add_labels(bbox_array, name="Bounding Box", opacity=0.5)
    viewer.add_labels(mask_array, name="Mask", opacity=0.6)
    napari.run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--bbox-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--mode", required=True, default="full", choices=["full", "cropped"])
    args = parser.parse_args()

    visualize_3d_with_napari(args.image_path, args.bbox_path, args.mask_path, args.mode)
