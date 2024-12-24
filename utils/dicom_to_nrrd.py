import os
import argparse
import pandas as pd
import SimpleITK as sitk


def collect_patient_data(root_dir, max_patients=0):
    collected_paths = []
    patient_count = 0

    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)

        if not os.path.isdir(patient_path):
            continue

        if patient_count >= max_patients:
            break

        # get the random subfolder inside the patient folder
        random_folder = next((f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))), None)
        if not random_folder:
            continue

        random_folder_path = os.path.join(patient_path, random_folder)

        for modality_folder in os.listdir(random_folder_path):
            modality_path = os.path.join(random_folder_path, modality_folder)
            if os.path.isdir(modality_path) and "segment" not in modality_path.lower():
                idx = modality_path.find("Breast_MRI_")
                collected_paths.append(modality_path[idx:])

        patient_count += 1

    return collected_paths

def map_new_paths(mapping_file, main_path):
    mapping_paths = {}
    for chunk in pd.read_csv(mapping_file, chunksize=5_000):
        chunk["original_path_and_filename"] = chunk["original_path_and_filename"].apply(os.path.dirname).str.replace("DICOM_Images/", main_path+"/MRI_NRRD/") + ".nrrd"
        
        chunk["descriptive_path"] = chunk["descriptive_path"].apply(os.path.dirname)
        chunk["descriptive_path"] = chunk["descriptive_path"].str.replace(r'BreastMRI(\d+)', r'Breast_MRI_\1', regex=True)
        chunk["descriptive_path"] = chunk["descriptive_path"].apply(lambda x: x[x.find("Breast_MRI_"):])

        chunk.drop_duplicates(inplace=True)
        mapping_paths.update(chunk.set_index("descriptive_path")["original_path_and_filename"].to_dict())

    return {k: v for k, v in mapping_paths.items() if list(mapping_paths.values()).count(v) == 1}

def dicom_to_nrrd(dicom_folder, nrrd_path):
    """load the DICOM images, stack into a 3D array, and save as an NRRD file"""
    dicom_reader = sitk.ImageSeriesReader()
    dicom_files = dicom_reader.GetGDCMSeriesFileNames(dicom_folder)
    dicom_reader.SetFileNames(dicom_files)
    image_3d = dicom_reader.Execute()

    os.makedirs(os.path.dirname(nrrd_path), exist_ok=True)
    sitk.WriteImage(image_3d, nrrd_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--n-patients", required=True, type=int)
    args = parser.parse_args()

    MAIN_PATH = args.data_path
    MRI_DICOM_FOLDER = f"{MAIN_PATH}/MRI_SEG_DICOM/"

    valid_paths = collect_patient_data(MRI_DICOM_FOLDER, max_patients=args.n_patients)

    mapping_table = map_new_paths(f"{MAIN_PATH}/Supplemental-Data//File_Path_Mapping_Tables.csv", MAIN_PATH)

    new_dict = {}
    for p in valid_paths:
        for m,v in mapping_table.items():
            if p.split("/")[0] == m.split("/")[0] and p.split("-")[-1] == m.split("-")[-1]:
                new_dict[f"{MAIN_PATH}/MRI_SEG_DICOM/{p}"] = v

    for d_folder, nrrd_file  in new_dict.items():
        # print(d_folder, nrrd_file)
        dicom_to_nrrd(d_folder, nrrd_file)
