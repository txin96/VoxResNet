import os
import numpy as np
from dipy.align.reslice import reslice
from dipy.segment.mask import median_otsu, bounding_box, crop, multi_median, otsu, applymask
import nibabel as nib


def preprocess(input, output, is_label):
    print("processing " + input + "...")
    img = nib.load(input)
    data = img.get_data()
    affine = img.affine
    zoom = img.header.get_zooms()[:3]
    new_data, new_affine = reslice(data, affine, zoom, (1., 1., 1.), 0)
    new_data = np.squeeze(new_data)
    print(new_data.shape)
    if (np.max(new_data.shape) > 256):
        print("slice to 256.")
        new_shape = [slice(0, 256) if _len > 256 else slice(0, _len) for _len in new_data.shape]
        new_data = new_data[new_shape]
    new_data = np.pad(new_data, [(0, 256 - len_) for len_ in new_data.shape], "constant")
    if not is_label:
        new_data = new_data[..., np.newaxis]
    new_img = nib.Nifti1Image(new_data, new_affine)
    nib.save(new_img, output)
    print(data.shape, new_data.shape)


def preprocess_dir(input_dir, output_dir, is_label):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    for file in files:
        preprocess(input_dir + "/" + file, output_dir + "/" + file, is_label)


def crop_image(input, output, median_radius=4, numpass=4, mask_path=None):
    print("processing " + input + "...")
    img = nib.load(input)
    data = img.get_data()
    affine = img.affine
    if len(data.shape) == 4:
        b0vol = data[..., 0].copy()
    else:
        b0vol = data.copy()
    mask = multi_median(b0vol, median_radius, numpass)
    thresh = otsu(mask)
    mask = mask > thresh
    mins, maxs = bounding_box(mask)
    mask_crop = mask
    mask_img_crop = nib.Nifti1Image(mask_crop.astype(np.float32), affine)
    if mask_path is not None:
        nib.save(mask_img_crop, mask_path + '_binary_mask_crop.nii.gz')
    else:
        nib.save(mask_img_crop, output + '_binary_mask_crop.nii.gz')
    mask = crop(mask, mins, maxs)
    croppedvolume = crop(data, mins, maxs)
    b0_mask_crop = applymask(croppedvolume, mask)
    # b0_mask_crop, mask_crop = median_otsu(data, median_radius, numpass, autocrop=True)
    print(mask_crop.shape, mask_crop.dtype, b0_mask_crop.shape, b0_mask_crop.dtype)
    b0_img_crop = nib.Nifti1Image(
        b0_mask_crop.astype(np.float32), affine)
    nib.save(b0_img_crop, output + '_mask_crop.nii.gz')


def crop_image_dir(input_dir, output_dir, median_radius=4, numpass=4, mask_dir=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    for file in files:
        file_name_without_ext = os.path.splitext(os.path.splitext(file)[0])[0]
        if mask_dir is not None:
            crop_image(input_dir + "/" + file, output_dir + "/" + file_name_without_ext, median_radius, numpass,
                       mask_dir + "/" + file_name_without_ext)
        else:
            crop_image(input_dir + "/" + file, output_dir + "/" + file_name_without_ext, median_radius, numpass)


def crop_label(input, mask, output):
    print("processing " + input + "...")
    img = nib.load(input)
    mask = nib.load(mask)
    mask_data = mask.get_data()
    data = img.get_data()
    affine = img.affine
    mins, maxs = bounding_box(mask_data)
    mask_data = crop(mask_data, mins, maxs)
    croppedvolume = crop(data, mins, maxs)
    mask_crop = applymask(croppedvolume, mask_data)
    img_mask_crop = nib.Nifti1Image(mask_crop.astype(np.float32), affine)
    nib.save(img_mask_crop, output)


def crop_label_dir(input_dir, mask_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    m_files = os.listdir(mask_dir)
    masks = []
    for file in m_files:
        if file.endswith("_binary_mask_crop.nii.gz"):
            masks.append(mask_dir + "/" + file)
    for index, file in enumerate(files):
        crop_label(input_dir + "/" + file, masks[index], output_dir + "/" + file)


def normalize(input_dir, output_dir, is_label):
    print("normalizing ...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    print("total files:", len(files))
    x_dims = []
    y_dims = []
    z_dims = []
    datas = []
    for index, file in enumerate(files):
        img = nib.load(input_dir + "/" + file)
        data = img.get_data()
        datas.append(img)
        x_dims.append(data.shape[0])
        y_dims.append(data.shape[1])
        z_dims.append(data.shape[2])
    max_dims = (np.max(x_dims), np.max(y_dims), np.max(z_dims))
    print("max_dims:", max_dims)
    for i, data in enumerate(datas):
        arr = data.get_data()
        print("normalizing no.", i, arr.shape, "...")
        arr = np.squeeze(arr)
        arr = np.pad(arr, [(0, max_dims[index] - len_) for index, len_ in enumerate(arr.shape)], "constant")
        if not is_label:
            arr = arr[..., np.newaxis]
        new_img = nib.Nifti1Image(arr, data.affine)
        nib.save(new_img, output_dir + "/" + files[i])

normalize("pack/crop/image","pack/normalize/image",False)
normalize("pack/crop/label","pack/normalize/label",True)
