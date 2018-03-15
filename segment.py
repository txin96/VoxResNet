from model import *
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import nibabel as nib

n_epoch = 1
batch_size = 1
n_class = 3
model_path = "model.npz"
save_path = "pack/segment/"


def read_data_for_segment(dir):
    res = []
    files = os.listdir(dir)
    for file in files:
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            res.append(nib.load(dir + '/' + file))
    return res


if not os.path.exists(save_path):
    os.makedirs(save_path)

print("reading data...")
image_data = read_data_for_segment("pack/test/image")
label_data = read_data_for_segment("pack/test/label")
print("finished reading data...")
images = np.asarray(image_data)
labels = np.asarray(label_data)
print(images.shape, images.dtype, labels.shape, labels.dtype)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
session_conf = tf.ConfigProto(
    #    device_count={'CPU' : 0, 'GPU' : 1},
    allow_soft_placement=True,
    #    log_device_placement=True
)
# session_conf.gpu_options.allow_growth = True
with tf.device('/GPU:0'):
    with tf.Session(config=session_conf) as sess:
        shape = images[0].get_data().shape
        x = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2], 1], name='x')
        valid_seg = tf.placeholder(tf.int32, [None, shape[0], shape[1], shape[2]], name="valid_seg")
        net = vox_res_net(x, is_train=False, n_out=n_class)
        tl.files.load_and_assign_npz(sess, model_path, net)
        output = tf.nn.softmax(net.outputs)
        out = tf.cast(tf.argmax(output, 4), tf.int32)
        correct_prediction = tf.equal(tf.cast(tf.argmax(output, 4), tf.int32), valid_seg)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("begin predicting ...")
        datas = [t.get_data() for t in images]
        datas = np.asarray(datas)
        print(datas.shape, out, x)
        result = tl.utils.predict(sess, net, datas, x, out, batch_size)
        img = nib.Nifti1Image(result[0], images[0].affine)
        nib.save(img, save_path + "0.nii.gz")
        print(result)
