import os
import numpy as np
import pandas as pd


def map_new_paths(mapping_file, mri_phase):
    """Return dictionary of patients and their corresponding paths for 'mri_phase' images."""
    assert mri_phase in ["pre", "post1", "post2", "post3", "post4", "T1"]

    mapping_paths = {}
    for chunk in pd.read_csv(mapping_file, chunksize=5_000):
        chunk = chunk[chunk["original_path_and_filename"].str.contains(mri_phase)]
        chunk["original_path_and_filename"] = chunk["original_path_and_filename"].apply(os.path.dirname).str.replace("DICOM_Images/", "").str.replace("/"+mri_phase, "")

        chunk["descriptive_path"] = chunk["descriptive_path"].apply(os.path.dirname)
        chunk["descriptive_path"] = chunk["descriptive_path"].str.replace(r"BreastMRI(\d+)", r"Breast_MRI_\1", regex=True)
        chunk["descriptive_path"] = chunk["descriptive_path"].apply(lambda x: x[x.find("Breast_MRI_"):])

        chunk.drop_duplicates(inplace=True)
        mapping_paths.update(chunk.set_index("original_path_and_filename")["descriptive_path"].to_dict())

    return {k: v for k, v in mapping_paths.items() if list(mapping_paths.values()).count(v) == 1}

def collect_patient_data(root_dir, dict_ids, max_patients=0):
    """Collects paths to 'mri_phase' folders for a given number of patients."""
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
