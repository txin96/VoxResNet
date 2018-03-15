import nibabel as nib
import os


def list_shape_info(dir):
    files = os.listdir(dir)
    for file in files:
        img = nib.load(dir + "/" + file)
        array = img.get_data()
        print(img.shape)
        print(array.dtype, array.shape)


list_shape_info("pack/preprocess/crop/label")


# print("-"*30)
# list_shape_info("pack/2018fwwb/label-part1")
