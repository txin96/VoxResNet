import numpy as np
from input_data import *


# two images must have same shape
def dice_metric(predict_seg, ground_truth):
    shape = predict_seg.shape
    x = shape[0]
    y = shape[1]
    z = shape[2]

    predict_sum = np.sum(predict_seg > 0)
    truth_sum = np.sum(ground_truth > 0)

    intersect = 0
    for i in range(0, x):
        for j in range(0, y):
            for k in range(0, z):
                if predict_seg[i][j][k] > 0 and predict_seg[i][j][k] == ground_truth[i][j][k]:
                    intersect += 1

    return 2 * intersect / float(predict_sum + truth_sum)


if __name__ == '__main__':
    data_set = DataSet("segment/", "test/label")
    result = np.asarray(data_set.images)
    truth = np.asarray(data_set.labels)
    acc = 0.0
    for i in range(0, len(result)):
        acc += dice_metric(result[i], truth[i])
    acc = acc/len(result)
    print(acc)
