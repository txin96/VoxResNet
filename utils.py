import nibabel as nib
import os
import numpy as np

def list_shape_info(dir):
    files = os.listdir(dir)
    for file in files:
        img = nib.load(dir+"/"+file)
        array = img.get_data()
        print(array.dtype, array.shape)

def one_hot(labels, nb_classes):
    array2 = (np.arange(nb_classes) == labels[..., None]).astype(dtype=np.int32)
    return array2
