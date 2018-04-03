from model import *
import tensorflow as tf
import tensorlayer as tl
from utils import *

n_epoch = 1
batch_size = 1
n_class = 3
model_path = "model.npz"
save_path = "segment/"


def read_data_for_segment(dir):
    res = []
    filename = []
    files = os.listdir(dir)
    for file in files:
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            res.append(nib.load(dir + '/' + file))
            filename.append(file)
    return res, filename


if not os.path.exists(save_path):
    os.makedirs(save_path)

print("reading data...")
image_data, image_name = read_data_for_segment("test/image")
images = np.asarray(image_data)
print("finished reading data...")
print(images.shape, images.dtype)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
        valid_seg = tf.placeholder(tf.int32, [None, shape[0], shape[1], shape[2], 3], name="valid_seg")
        net = vox_res_net(x, is_train=False, n_out=n_class)
        tl.files.load_and_assign_npz(sess, model_path, net)
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

        # n_count = 0
        # for i in range(n_epoch):
        # start_time = time.time()
        # val_acc,n_batch = 0,0
        # for X_train_a, y_train_a in tl.iterate.minibatches(images, labels, batch_size, shuffle=True):
        # data_x = np.asarray([t.get_data() for t in X_train_a])
        # data_y = np.asarray([t.get_data() for t in y_train_a])
        # print(data_x.shape,data_x.dtype,data_y.shape,data_y.dtype)
        # valid_batch = {x: data_x, valid_seg: data_y}
        # feed_dict = {x: data_x}
        # feed_dict.update(net.all_drop)
        # predict_y,ac = sess.run([output,acc], feed_dict=valid_batch)
        # #predict_y = sess.run([output], feed_dict=feed_dict)
        # val_acc+=ac
        # n_batch += 1
        # output_data = tf.cast(tf.argmax(predict_y, 4),tf.int32)
        # print(predict_y.shape,predict_y.dtype,output_data.shape,output_data.dtype)
        # print("Epoch ",i," ,",batch_size," batch_size took ",time.time()-start_time," acc:", (val_acc / n_batch))
        # for t in range(batch_size):
        # data_to_save = predict_y[t]
        # affine = X_train_a[t].affine
        # print(data_to_save.shape,affine.shape)
        # img = nib.Nifti1Image(data_to_save,affine)
        # path = save_path+str(n_count)+".nii.gz"
        # nib.save(img,path)
        # n_count += 1
        # print("Success predict and save to",path)
        # print("   val acc: %f" % (val_acc / n_batch))
