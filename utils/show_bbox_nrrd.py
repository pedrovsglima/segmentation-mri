import argparse
import napari
import SimpleITK as sitk

def visualize_3d_with_napari(image_path, mask_path):

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    assert image_array.shape == mask_array.shape, "Image and mask dimensions must match!"

    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=3)  # 3D mode
        viewer.add_image(image_array, name="MRI Image", colormap="gray", rendering="mip")
        viewer.add_labels(mask_array, name="Mask", opacity=0.5)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--mask-path", required=True)
    args = parser.parse_args()

    visualize_3d_with_napari(args.image_path, args.mask_path)
