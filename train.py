from model import *
from input_data import *
import tensorflow as tf
from utils import *

# Basic param
n_image = 100
n_epoch = 200
n_class = 3
batch_size = 2
total_step = n_image * n_epoch / batch_size
global_step = tf.Variable(0, name='global_step')
learning_rate = tf.train.exponential_decay(0.001, global_step, 150, 0.99, staircase=True)
print_freq = 1

# Reading data
print("reading data...")
data_set = DataSet("train/image", "train/label")
images = np.asarray(data_set.images)
labels = np.asarray(data_set.labels)

test_set = DataSet("test/image", "test/label")
image_test = np.asarray(test_set.images)
label_test = np.asarray(test_set.labels)
print("finished reading data.")

# Convert data to array
images = np.asarray(images, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int32)
labels = tf.one_hot(labels, n_class)
image_test = np.asarray(image_test, dtype=np.float32)
label_test = np.asarray(label_test, dtype=np.int32)
label_test = tf.one_hot(label_test, n_class)
print("images shape:" + str(images.shape), images.dtype)
print("labels shape:" + str(labels.shape), labels.dtype)

session_conf = tf.ConfigProto(
    allow_soft_placement=True,
)

with tf.device("/GPU:0"):
    with tf.Session(config=session_conf) as sess:
        sess.as_default()
        shape = images.shape
        x = tf.placeholder(tf.float32, [None, shape[1], shape[2], shape[3], 1], name='x')
        valid_seg = tf.placeholder(tf.int32, [None, shape[1], shape[2], shape[3], n_class], name="valid_seg")
        logits = vox_res_net(x, is_train=True, n_out=3)
        # net = vox_res_net(x, is_train=True, n_out=3)
        outputs = [x.outputs for x in logits]
        net = logits[-1]
        out_seg = net.outputs
        print("out_seg shape:  " + str(out_seg.shape))
        print("valid_seg shape:  " + str(valid_seg.shape))

        correct_prediction = tf.equal(tf.cast(tf.argmax(out_seg, 4), tf.int32),
                                      tf.cast(tf.argmax(valid_seg, 4), tf.int32))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=out_seg, labels=valid_seg))
        print(loss)
        train_params = net.all_params
        for t in logits[:-1]:
            w = tf.Variable(1.0, name="classifier_w") - 0.999 * tf.cast(global_step, tf.float32) / total_step
            train_params.append(w)
            loss += w * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=t.outputs, labels=valid_seg))
        print("output shape:  " + str(t.outputs.shape))
        l2 = 0
        for w in train_params[:-4]:
            l2 += tf.contrib.layers.l2_regularizer(1e-4)(w)
        cost = l2 + loss
        print(cost)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-08, use_locking=False).minimize(cost, global_step=global_step,
                                                                                     var_list=train_params[:-4])

        tl.layers.initialize_global_variables(sess)
        net.print_params()
        net.print_layers()

        # train
        tl.utils.fit(sess, net, train_op, cost, images, labels, x, valid_seg,
                     acc=acc, batch_size=batch_size, n_epoch=n_epoch, print_freq=print_freq,
                     eval_train=False, tensorboard=True, tensorboard_epoch_freq=1, tensorboard_graph_vis=True,
                     tensorboard_weight_histograms=True)

        # evaluate
        tl.utils.test(sess, net, acc, image_test, label_test, x, valid_seg, batch_size=batch_size, cost=cost)

        # Save model
        tl.files.save_npz(net.all_params[:-4], name='model.npz')

        sess.close()
