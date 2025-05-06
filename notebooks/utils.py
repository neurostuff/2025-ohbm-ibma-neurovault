"""Get NeuroVault images for collections linked to PubMed articles."""

import os

import nibabel as nib
import numpy as np
import pandas as pd
import requests
from nimare.dataset import Dataset
from nimare.io import DEFAULT_MAP_TYPE_CONVERSION
from nimare.transforms import ImageTransformer
from scipy.stats import pearsonr


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


def _rm_nonstat_maps(data_df):
    """
    Remove non-statistical maps from a dataset.

    Notes
    -----
    This function requires the dataset to have a metadata field called
    "image_name" and "image_file".
    """
    sel_ids = []
    for _, row in data_df.iterrows():
        image_name = row["name"].lower()
        file_name = row["file"].lower()

        exclude = False
        for term in [
            "ica",
            "pca",
            "ppi",
            "seed",
            "functional connectivity",
            "correlation",
        ]:
            if term in image_name:
                exclude = True
                break

        if "cope" in file_name and (
            "zstat" not in file_name and "tstat" not in file_name
        ):
            exclude = True

        if "tfce" in file_name:
            exclude = True

        if not exclude:
            sel_ids.append(row["id"])

    data_df = data_df[data_df["id"].isin(sel_ids)]
    data_df = data_df.reset_index()
    return data_df


def convert_to_nimare_dataset(images_df):
    dataset_dict = {}
    for _, image in images_df.iterrows():
        id_ = image["id"]
        new_contrast_name = f"{id_}-1"
        map_type = f"{image['map_type']} map"

        if id_ not in dataset_dict:
            dataset_dict[id_] = {}

        if "contrasts" not in dataset_dict[id_]:
            dataset_dict[id_]["contrasts"] = {}

        dataset_dict[id_]["contrasts"][new_contrast_name] = {
            "metadata": {"sample_sizes": [image["number_of_subjects"]]},
            "images": {DEFAULT_MAP_TYPE_CONVERSION[map_type]: image["path"]},
        }

    dset = Dataset(dataset_dict)
    dset = ImageTransformer("z").transform(dset)
    dset = ImageTransformer("t").transform(dset)

    return dset
