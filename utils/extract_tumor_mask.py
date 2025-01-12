import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label, binary_fill_holes, binary_opening


def extract_bounding_box(seg_array):
    coords = np.argwhere(seg_array > 0)
    start = coords.min(axis=0)
    end = coords.max(axis=0) + 1  # include the last index
    return start, end

def apply_threshold(image_array, mask_array, lower_limit, upper_limit):

    intensities = image_array[mask_array > 0]

    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)
    mean_intensity = np.mean(intensities)
    std_intensity = np.std(intensities)

    lower_threshold = max_intensity * lower_limit # mean_intensity - 0.25 * std_intensity
    upper_threshold = max_intensity * upper_limit # mean_intensity + 0.25 * std_intensity
    # print(f"Threshold Range: [{lower_threshold:.2f}, {upper_threshold:.2f}]")
    
    return ((image_array >= lower_threshold) & (image_array <= upper_threshold)).astype(np.uint8)

def refine_mask(thresholded_mask):

    filled_mask = binary_fill_holes(thresholded_mask)

    # apply morphological opening (erosion followed by dilation)
    structure = np.ones((1, 1, 1))  # define a 3D structure element
    opened_mask = binary_opening(filled_mask, structure=structure)

    # retain only the largest connected component
    labeled_array, num_features = label(opened_mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled_array.ravel())
        largest_component = np.argmax(component_sizes[1:]) + 1  # exclude background
        refined_mask = (labeled_array == largest_component).astype(np.uint8)
    else:
        refined_mask = np.zeros_like(thresholded_mask, dtype=np.uint8)

    return refined_mask

def thresholded_mask(image_array, mask_array, lower_limit=0.5, upper_limit=0.75):

    assert image_array.shape == mask_array.shape, "Image and mask dimensions must match!"

    start, end = extract_bounding_box(mask_array)
    crop_image_array = image_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    crop_mask_array = mask_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    thresholded_mask = apply_threshold(crop_image_array, crop_mask_array, lower_limit, upper_limit)

    # refine thresholded mask
    refined_mask = refine_mask(thresholded_mask)

    # refined mask to full size
    full_refined_mask = np.zeros_like(image_array, dtype=np.uint8)
    bbox_region = crop_mask_array > 0
    full_refined_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]][bbox_region] = refined_mask[bbox_region]

    return full_refined_mask
