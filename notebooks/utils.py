"""Get NeuroVault images for collections linked to PubMed articles."""

import os

import nibabel as nib
import pandas as pd
import requests
from nimare.dataset import Dataset
from nimare.io import DEFAULT_MAP_TYPE_CONVERSION
from nimare.transforms import ImageTransformer


def download_images(image_ids, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    image_info_dict = {"id": [], "path": []}
    for image_id in image_ids:
        # Construct the NeuroVault API URL for image info
        image_info_url = f"https://neurovault.org/api/images/{image_id}/"

        try:
            # Make a GET request to fetch image info
            response = requests.get(image_info_url)

            if response.status_code == 200:
                image_info = response.json()

                # Download the image file
                image_url = image_info["file"]
                collection_id = image_info["collection_id"]
                image_filename = os.path.basename(image_url)
                rel_path = f"{collection_id}-{image_id}_{image_filename}"
                image_path = os.path.join(output_directory, rel_path)
                if not os.path.exists(image_path):
                    # Download the image
                    response = requests.get(image_url)
                    with open(image_path, "wb") as image_file:
                        image_file.write(response.content)

                try:
                    # Some image may be invalid
                    nib.load(image_path)
                except Exception:
                    pass
                else:
                    # Append image info to the list
                    image_info_dict["id"].append(image_id)
                    image_info_dict["path"].append(image_path)

        except Exception as e:
            print(
                f"An error occurred while processing image ID {image_id}: {str(e)}",
                flush=True,
            )

    return pd.DataFrame(image_info_dict)


def convert_to_nimare_dataset(
    images_df,
    study_col,
    contrast_col,
    sample_size_col,
    map_type_col,
    path_col,
    metadata_cols,
):
    """
    Convert a DataFrame containing neuroimaging data into a NiMARE Dataset object.

    This function processes a DataFrame of neuroimaging data and transforms it into a
    nested dictionary structure, and then use to create a NiMARE Dataset object.

    Parameters
    ----------
    images_df : pandas.DataFrame
        DataFrame containing neuroimaging data with columns for study identifiers,
        contrast identifiers, map types, file paths, sample sizes, and other metadata.
    study_col : str
        Name of the column in images_df that contains study identifiers.
    contrast_col : str
        Name of the column in images_df that contains contrast identifiers.
    sample_size_col : str
        Name of the column in images_df that contains sample size information.
    map_type_col : str
        Name of the column in images_df that contains map type information.
        The map_type should be compatible with the DEFAULT_MAP_TYPE_CONVERSION
        dictionary which maps between common map type names and NiMARE's naming conventions
    path_col : str
        Name of the column in images_df that contains absolute path to each image.
    metadata_cols : list of str
        List of column names in images_df to include as metadata in the dataset.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        Dataset object containing experiment information from DataFrame.
    """
    dataset_dict = {}
    for _, image in images_df.iterrows():
        col_id = image[study_col]
        img_id = image[contrast_col]

        map_type = f"{image[map_type_col]} map"

        if col_id not in dataset_dict:
            dataset_dict[col_id] = {}

        if "contrasts" not in dataset_dict[col_id]:
            dataset_dict[col_id]["contrasts"] = {}

        # Add all metadata columns to the metadata dictionary
        metadata = {"sample_sizes": [image[sample_size_col]]}
        for col in metadata_cols:
            metadata[col] = image[col]

        dataset_dict[col_id]["contrasts"][img_id] = {
            "metadata": metadata,
            "images": {DEFAULT_MAP_TYPE_CONVERSION[map_type]: image[path_col]},
        }

    dset = Dataset(dataset_dict)
    dset = ImageTransformer("z").transform(dset)
    dset = ImageTransformer("t").transform(dset)

    return dset
