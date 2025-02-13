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

def volume_and_seg_to_numpy(dicom_folder, seg_path, output_path, patient_id):
    """Converts MRI volume and segmentation data to NumPy arrays and saves them to specified output paths."""
    # read MRI images
    image_array, _, nrrd_breast_data, _ = preprocessing.read_precontrast_mri_and_segmentation(
        dicom_folder,
        seg_path
    )

    # normalize and z-score image
    image_array = preprocessing.z_score_image(preprocessing.normalize_image(image_array))

    # create directory if it doesn't exist
    output_path_volume = f"{output_path}/mri_npy/{patient_id}/{patient_id}.npy"
    output_path_breast = f"{output_path}/mri_npy_seg/{patient_id}/Segmentation_{patient_id}_Breast.npy"
    # output_path_dv = f"{output_path}/mri_npy_seg/{patient_id}/Segmentation_{patient_id}_Dense_and_Vessels.npy"

    output_dirs = [
        os.path.dirname(output_path_volume),
        os.path.dirname(output_path_breast),
        # os.path.dirname(output_path_dv)
    ]
    for dir in output_dirs:
        os.makedirs(dir, exist_ok=True)

    # save arrays as .npy files
    np.save(output_path_volume, image_array)
    np.save(output_path_breast, nrrd_breast_data)
    # np.save(output_path_dv, nrrd_dv_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--segmentation-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--n-patients", required=True, type=int)
    args = parser.parse_args()

    MAIN_PATH = args.image_path
    SEG_PATH = args.segmentation_path
    OUTPUT_PATH = args.output_path
    N_PATIENTS = args.n_patients

    mapping_table = map_new_paths(f"{MAIN_PATH}/File_Path_Mapping_Tables.csv", mri_phase="pre")
    valid_paths = collect_patient_data(f"{MAIN_PATH}/Duke-Breast-Cancer-MRI/", mapping_table, max_patients=N_PATIENTS)

    for patient_dicom_folder in valid_paths:
        patient_id = patient_dicom_folder.split("/")[0]

        if os.path.exists(f"{SEG_PATH}/{patient_id}"):
            # print(f"Processing volume and segmentation data for patient {patient_id}...")
            volume_and_seg_to_numpy(
                f"{MAIN_PATH}/Duke-Breast-Cancer-MRI/{patient_dicom_folder}",
                f"{SEG_PATH}/{patient_id}",
                OUTPUT_PATH,
                patient_id
            )
        else:
            # print(f"Processing volume data for patient {patient_id}...")
            volume_to_numpy(
                f"{MAIN_PATH}/Duke-Breast-Cancer-MRI/{patient_dicom_folder}",
                f"{OUTPUT_PATH}/mri_npy/{patient_id}/{patient_id}.npy"
            )
