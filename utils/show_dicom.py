import os
import argparse
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle


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

def display_with_slider(dicom_files:list, annotations:dict) -> None:

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img_display = ax.imshow(dicom_files[0].pixel_array, cmap="gray")
    ax.axis("off")

    # slider for navigating through images
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor="lightgray")
    slider = Slider(ax_slider, "Image", 0, len(dicom_files) - 1, valinit=0, valstep=1)

    annotation_box = None
    def update(val):
        nonlocal annotation_box
        slice_index = int(slider.val)

        img_display.set_data(dicom_files[slice_index].pixel_array)

        if annotation_box:
            annotation_box.remove()
            annotation_box = None

        if annotations is not None:
            for annotation in annotations:
                start_row, end_row = annotation["Start Row"], annotation["End Row"]
                start_col, end_col = annotation["Start Column"], annotation["End Column"]
                start_slice, end_slice = annotation["Start Slice"], annotation["End Slice"]

                if start_slice <= slice_index <= end_slice:
                    annotation_box = Rectangle(
                        (start_col, start_row),
                        end_col - start_col,
                        end_row - start_row,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none"
                    )
                    ax.add_patch(annotation_box)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def load_annotations(annotation_file:str, patient:str) -> dict:
    annotations_df = pd.read_excel(annotation_file)
    annotations_df = annotations_df[annotations_df["Patient ID"] == patient]

    int_columns = ["Start Row", "End Row", "Start Column", "End Column", "Start Slice", "End Slice"]
    annotations_df = annotations_df[int_columns].astype(int)
    return annotations_df.to_dict(orient="records")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to patient DICOM folder or single DICOM file.")
    parser.add_argument('--annotation-file', help="Path to Excel file with annotations.")
    args = parser.parse_args()

    patient_id = args.data_path.split("/")[-4]

    if args.data_path.endswith(".dcm"):
        show_one_dicom_image(args.data_path)
    else:
        dicom_files = load_dicom_files(args.data_path)
        annotations = load_annotations(args.annotation_file, patient_id) if args.annotation_file else None
        display_with_slider(dicom_files, annotations)
