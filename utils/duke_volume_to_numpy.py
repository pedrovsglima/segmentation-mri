import os
import sys
import argparse
import numpy as np

from file_path_mapping import map_new_paths, collect_patient_data

sys.path.append(os.getcwd())
from duke_segmentation_code import preprocessing


def volume_to_numpy(dicom_folder, output_path):
    """Converts a volume of DICOM images to a NumPy array and saves it to a file."""
    # read MRI images
    image_array, _ = preprocessing.read_precontrast_mri(dicom_folder)

    # create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # save array as a .npy file
    np.save(output_path, image_array)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--n-patients", required=True, type=int)
    args = parser.parse_args()

    MAIN_PATH = args.image_path
    OUTPUT_PATH = args.output_path
    N_PATIENTS = args.n_patients

    mapping_table = map_new_paths(f"{MAIN_PATH}/File_Path_Mapping_Tables.csv", mri_phase="pre")
    valid_paths = collect_patient_data(f"{MAIN_PATH}/Duke-Breast-Cancer-MRI/", mapping_table, max_patients=N_PATIENTS)

    for patient_dicom_folder in valid_paths:
        patient_id = patient_dicom_folder.split("/")[0]

        volume_to_numpy(
            f"{MAIN_PATH}/Duke-Breast-Cancer-MRI/{patient_dicom_folder}",
            f"{OUTPUT_PATH}/{patient_id}/{patient_id}.npy"
        )
