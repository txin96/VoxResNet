from model import *
import tensorflow as tf
import tensorlayer as tl
from utils import *

# basic param
batch_size = 1
n_class = 3

# file path
test_image_path = "test/image"
model_path = "model/160x188x128/model.npz"
save_path = "segment/"


def read_data_for_segment(file_dir):
    res = []
    filename = []
    filelist = os.listdir(file_dir)
    for file in filelist:
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            res.append(nib.load(file_dir + '/' + file))
            filename.append(file)
    return res, filename


def segment(save_dir, image_path, model_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("reading data...")
    image_data, image_name = read_data_for_segment(image_path)
    images = np.asarray(image_data)
    print("finished reading data...")
    print(images.shape, images.dtype)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
    )

    with tf.device('/GPU:0'):
        with tf.Session(config=session_conf) as sess:
            shape = images[0].get_data().shape
            x = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2], 1], name='x')
            valid_seg = tf.placeholder(tf.int32, [None, shape[0], shape[1], shape[2], 3], name="valid_seg")
            net = vox_res_net(x, is_train=False, n_out=n_class)
            tl.files.load_and_assign_npz(sess, model_dir, net)
            output = tf.nn.softmax(net.outputs)
            out = tf.cast(tf.argmax(output, 4), tf.int32)
            correct_prediction = tf.equal(tf.cast(tf.argmax(output, 4), tf.int32), valid_seg)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            print("begin predicting ...")
            data = [t.get_data() for t in images]
            data = np.asarray(data)
            print(data.shape, out, x)
            result = tl.utils.predict(sess, net, data, x, out, batch_size)
            for i in range(0, len(result)):
                img = nib.Nifti1Image(result[i], images[0].affine)
                nib.save(img, save_path + image_name[i])


if __name__ == '__main__':
    segment(save_path, test_image_path, model_path)
