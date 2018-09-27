import os
import numpy as np
import nibabel as nib
from picprocess import pic_process


def new_normalize(input_file, output_file, shape, is_label):
    img = nib.load(input_file)
    data = img.get_data()
    data = np.squeeze(data)
    shape = list(shape)
    if len(shape) > 3:
        shape = shape[:3]
    print(data.shape, shape)
    slice_in = [
        slice(int((data.shape[i] - _len) / 2), int((data.shape[i] + _len) / 2)) if data.shape[i] > _len else slice(0,
                                                                                                                   data.shape[
                                                                                                                       i])
        for i, _len in enumerate(shape)]
    data = data[slice_in]
    print(slice_in)
    data = np.pad(data, [(0, shape[i] - len_) for i, len_ in enumerate(data.shape)], "constant")
    if not is_label:
        data = data[..., np.newaxis]
    print(data.shape)
    new_img = nib.Nifti1Image(data, img.affine)
    nib.save(new_img, output_file)


def process(input_dir, output_dir, shape, is_label):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = os.listdir(input_dir)
    for file in files:
        new_normalize(input_dir + "/" + file, output_dir + "/" + file, shape, is_label)


if __name__ == '__main__':
    print("Begin preprocessing.")
    # Modify the save path here, second parameter is temporary file path
    process('origin/image', 'pack/image', (160, 188, 128), False)
    process('origin/label', 'pack/label', (160, 188, 128), True)
    # The second parameter here is where preprocessed file are saved
    pic_process('pack/image', 'train/image')
    print("Finish preprocessing.")

