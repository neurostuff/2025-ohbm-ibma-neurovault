import nibabel as nib
import numpy as np
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import concat_imgs, resample_to_img
from scipy.stats import pearsonr


def _rm_nonstat_maps(data_df, verbose=0):
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
                if verbose > 0:
                    print(f"Removing {row["id"]}, term {term} in image name")
                break

        if "cope" in file_name and (
            "zstat" not in file_name and "tstat" not in file_name
        ):
            exclude = True
            if verbose > 0:
                print(f"Removing {row["id"]}, term 'cope' in file name")

        if "tfce" in file_name:
            if verbose > 0:
                print(f"Removing {row["id"]}, term 'tfce' in file name")
            exclude = True

        if not exclude:
            sel_ids.append(row["id"])

    data_df = data_df[data_df["id"].isin(sel_ids)]
    data_df = data_df.reset_index()
    return data_df


def _rm_extreme_maps(data_df, zmin=1.96, z_max=50, verbose=0):
    outliers_ids = []
    for _, row in data_df.iterrows():
        img = nib.load(row["path"])
        data = img.get_fdata()
        max_val = np.max(data)
        min_val = np.min(data)

        # Catch any inverted p-value, effect size or correlation maps
        if max_val < zmin and min_val > -zmin:
            outliers_ids.append(row["id"])
            if verbose > 0:
                print(f"Removing {row["id"]}, possible effect size or correlation map")
            continue

        # Catch any map with extreme values
        if max_val > z_max or min_val < -z_max:
            outliers_ids.append(row["id"])
            if verbose > 0:
                print(f"Removing {row["id"]}, possible map with extreme values")
            continue

        # Catch any map with all positive or all negative values
        if ((data > 0).sum() == len(data)) or ((data < 0).sum() == len(data)):
            outliers_ids.append(row["id"])
            if verbose > 0:
                print(
                    f"Removing {row["id"]}, map with all positive or all negative values"
                )
            continue

    data_df = data_df[~data_df["id"].isin(outliers_ids)]
    data_df = data_df.reset_index()
    return data_df


def _get_data(dset, imtype="z"):
    """Get data from a Dataset object.

    Parameters
    ----------
    dset : :obj:`nimare.dataset.Dataset`
        Dataset object.
    imtype : :obj:`str`, optional
        Type of image to load. Default is 'z'.

    Returns
    -------
    data : :obj:`numpy.ndarray`
        Data from the Dataset object.
    """
    images = dset.get_images(imtype=imtype)
    _resample_kwargs = {"clip": True, "interpolation": "linear"}
    masker = dset.masker

    imgs = [
        (
            nib.load(img)
            if check_same_fov(nib.load(img), reference_masker=masker.mask_img)
            else resample_to_img(nib.load(img), masker.mask_img, **_resample_kwargs)
        )
        for img in images
    ]

    img4d = concat_imgs(imgs, ensure_ndim=4)
    return masker.transform(img4d)


def _rm_duplicates_maps(dset, verbose=0):
    new_dset = dset.copy()
    data = _get_data(dset, imtype="z")
    ids = dset.ids

    sel_ids = []
    ecl_ids = []
    data_ave = np.mean(data, axis=0)
    for data_i, data_arr_i in enumerate(data):
        if ids[data_i] in sel_ids or ids[data_i] in ecl_ids:
            continue

        data_ave_i_corr, _ = pearsonr(data_arr_i, data_ave)

        for data_j, data_arr_j in enumerate(data):
            if data_i >= data_j:
                continue

            if ids[data_j] in sel_ids or ids[data_j] in ecl_ids:
                continue

            data_ij_corr, _ = pearsonr(data_arr_i, data_arr_j)
            is_duplicate = np.isclose(np.abs(data_ij_corr), 1, atol=1e-2)

            if is_duplicate:
                data_ave_j_corr, _ = pearsonr(data_arr_j, data_ave)
                if data_ave_i_corr > data_ave_j_corr:
                    sel_ids.append(ids[data_i])
                    ecl_ids.append(ids[data_j])
                    if verbose > 0:
                        print(
                            f"Removing {ids[data_j]}, possible duplicate of {ids[data_i]}"
                        )
                else:
                    sel_ids.append(ids[data_j])
                    ecl_ids.append(ids[data_i])
                    if verbose > 0:
                        print(
                            f"Removing {ids[data_i]}, possible duplicate of {ids[data_j]}"
                        )

        if (ids[data_i] not in sel_ids) and (ids[data_i] not in ecl_ids):
            sel_ids.append(ids[data_i])

    return new_dset.slice(sel_ids)
