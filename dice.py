import numpy as np

#两个参数要规格一致
def dice_metric(sample_seg,real_seg):
    shape=sample_seg.shape
    x=shape[0]
    y=shape[1]
    z=shape[2]

    sample_min=np.min(sample_seg)
    real_min=np.min(real_seg)

    sample_sum=np.sum(sample_seg>sample_min)
    real_sum=np.sum(real_seg>real_min)

    intersect=0
    for i in range(0,x):
        for j in range(0,y):
            for k in range(0,z):
                if sample_seg[i][j][k]>sample_min and real_seg[i][j][k]>real_min:
                    intersect+=1

    return 2*intersect/float(sample_sum+real_sum)