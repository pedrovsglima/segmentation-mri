import os
import argparse
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk


def map_new_paths(mapping_file):
    """Return dictionary of patients and their corresponding paths for 'post_1' images."""
    mapping_paths = {}
    for chunk in pd.read_csv(mapping_file, chunksize=5_000):
        chunk = chunk[chunk["original_path_and_filename"].str.contains("post_1")]
        chunk["original_path_and_filename"] = chunk["original_path_and_filename"].apply(os.path.dirname).str.replace("DICOM_Images/", "").str.replace("/post_1", "")

        chunk["descriptive_path"] = chunk["descriptive_path"].apply(os.path.dirname)
        chunk["descriptive_path"] = chunk["descriptive_path"].str.replace(r"BreastMRI(\d+)", r"Breast_MRI_\1", regex=True)
        chunk["descriptive_path"] = chunk["descriptive_path"].apply(lambda x: x[x.find("Breast_MRI_"):])

        chunk.drop_duplicates(inplace=True)
        mapping_paths.update(chunk.set_index("original_path_and_filename")["descriptive_path"].to_dict())

    return {k: v for k, v in mapping_paths.items() if list(mapping_paths.values()).count(v) == 1}

def collect_patient_data(root_dir, dict_ids, max_patients=0):
    """Collects paths to 'post_1' folders for a given number of patients."""
    collected_paths = []
    patient_count = 0

    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)

        if not os.path.isdir(patient_path):
            continue

        if patient_count >= max_patients:
            break

        # get the 'random' subfolder inside the patient folder
        random_folder = next((f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))), None)
        if not random_folder:
            continue
        random_folder_path = os.path.join(patient_path, random_folder)

        folder_id = dict_ids.get(patient_folder)
        folder_id = folder_id.split("-")[-1]

        for modality_folder in os.listdir(random_folder_path):
            if folder_id in modality_folder:
                modality_path = os.path.join(random_folder_path, modality_folder)
                if os.path.isdir(modality_path):
                    idx = modality_path.find("Breast_MRI_")
                    collected_paths.append(modality_path[idx:])

        patient_count += 1

    return collected_paths

def create_3d_mask_from_dicom(dicom_folder, annotation, output_path):
    """Create a 3D Mask based on Annotation Boxes. Tumor voxels are set to 1 and other voxels are set to 0."""
    dicom_files = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in sorted(os.listdir(dicom_folder)) if f.endswith(".dcm")]
    total_slices = len(dicom_files)

    start_slice, end_slice = int(annotation["Start Slice"]), int(annotation["End Slice"])
    start_row, end_row = int(annotation["Start Row"]), int(annotation["End Row"])
    start_column, end_column = int(annotation["Start Column"]), int(annotation["End Column"])

    z_coordinates = [float(dcm.ImagePositionPatient[2]) for dcm in dicom_files]
    ascending_order = z_coordinates[0] < z_coordinates[-1]
    if not ascending_order:
        dicom_files = dicom_files[::-1]
        z_coordinates = z_coordinates[::-1]
        start_slice, end_slice = total_slices - end_slice + 1, total_slices - start_slice + 1

    image_array = np.stack([dcm.pixel_array for dcm in dicom_files], axis=0)
    mask_array = np.zeros_like(image_array)

    mask_array[start_slice:end_slice, start_row:end_row, start_column:end_column] = 1

    mask_image = sitk.GetImageFromArray(mask_array)

    image_3d = sitk.ReadImage(dicom_files[0].filename)
    mask_image.SetSpacing(image_3d.GetSpacing())
    mask_image.SetOrigin(image_3d.GetOrigin())
    mask_image.SetDirection(image_3d.GetDirection())

    sitk.WriteImage(mask_image, output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--n-patients", required=True, type=int)
    args = parser.parse_args()

    MAIN_PATH = args.image_path
    MASK_PATH = args.mask_path
    N_PATIENTS = args.n_patients

    mapping_table = map_new_paths(f"{MAIN_PATH}/Supplemental-Data/File_Path_Mapping_Tables.csv")
    valid_paths = collect_patient_data(f"{MAIN_PATH}/MRI_SEG_DICOM/", mapping_table, max_patients=N_PATIENTS)

    annotation_df = pd.read_excel(f"{MAIN_PATH}/Supplemental-Data/Annotation_Boxes.xlsx")

    for patient_dicom_folder in valid_paths:
        patient_id = patient_dicom_folder.split("/")[0]

        annotation_dict = annotation_df[annotation_df["Patient ID"] == patient_id].iloc[0].to_dict()

        create_3d_mask_from_dicom(
            f"{MAIN_PATH}/MRI_SEG_DICOM/{patient_dicom_folder}",
            annotation_dict,
            f"{MASK_PATH}/{patient_id}.seg.nrrd"
        )
