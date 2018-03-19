import nibabel as nib
import os
import numpy as np

def list_shape_info(dir):
    files = os.listdir(dir)
    for file in files:
        img = nib.load(dir+"/"+file)
        array = img.get_data()
        print(array.dtype, array.shape)

#list_shape_info("pack/preprocess/crop/label")
#print("-"*30)
#list_shape_info("pack/2018fwwb/label-part1")

def reshape(label):
    w = label.shape[0]
    x = label.shape[1]
    y = label.shape[2]
    z = label.shape[3]
    array2 = np.zeros((w, x, y, z, 3), dtype=np.int32)
    for l in range(0, w):
        for i in range(0, x):
            for j in range(0, y):
                for k in range(0, z):
                    if label[l][i][j][k][0] == 0:
                        array2[l][i][j][k][0] = 1
                    elif label[l][i][j][k][0] == 1:
                        array2[l][i][j][k][1] = 1
                    elif label[l][i][j][k][0] == 2:
                        array2[l][i][j][k][2] = 1
    return array2


