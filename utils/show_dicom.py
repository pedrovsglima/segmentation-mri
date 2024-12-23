import os
import argparse
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def show_one_dicom_image(filepath:str) -> None:

    dicom_file = pydicom.dcmread(filepath)
    image = dicom_file.pixel_array

    print("image shape:", np.shape(image))

    if len(np.shape(image)) == 2:
        plt.imshow(image)
        plt.show()
    elif len(np.shape(image)) == 3:
        plt.imshow(image[0])
        plt.show()

def load_dicom_files(patient_images_folder:str) -> list:
    dicom_files = []
    for filename in sorted(os.listdir(patient_images_folder)):
        if filename.endswith(".dcm"):
            filepath = os.path.join(patient_images_folder, filename)
            dicom_files.append(pydicom.dcmread(filepath))
    return dicom_files

def display_with_slider(dicom_files:list) -> None:

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img_display = ax.imshow(dicom_files[0].pixel_array)#, cmap="gray")
    ax.axis("off")

    # slider for navigating through images
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgray")
    slider = Slider(ax_slider, "Image", 0, len(dicom_files) - 1, valinit=0, valstep=1)

    # update function to change the displayed image
    def update(val):
        img_display.set_data(dicom_files[int(slider.val)].pixel_array)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path')
    args = parser.parse_args()

    if args.data_path.endswith(".dcm"):
        show_one_dicom_image(args.data_path)
    else:
        dicom_files = load_dicom_files(args.data_path)
        display_with_slider(dicom_files)
