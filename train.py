from model import *
from input_data import *
import tensorflow as tf
import nibabel as nib
import numpy as np
import os

# img = nib.load('pack/preprocess/image/ADNI_002_S_1018_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217030439623_S23128_I40817.nii.gz')
# data = img.get_data()
# print(data.shape)

# basic param
n_epoch = 1
batch_size = 1
learning_rate = 0.0001
print_freq = 1

print("reading data...")
data_set = DataSet("pack/normalize/image", "pack/normalize/label")
# images = np.asarray(data_set.images)
# labels = np.asarray(data_set.labels)
print("finished reading data.")
images, labels = data_set.next_batch(batch_size)
img_test, label_test = data_set.next_batch(batch_size)
# img_val,label_val = data_set.next_batch(batch_size)
images = np.asarray(images, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int32)
img_test = np.asarray(img_test, dtype=np.float32)
label_test = np.asarray(label_test, dtype=np.int32)
# img_val = np.asarray(img_val,dtype=np.float32)
# label_val = np.asarray(label_val,dtype=np.int32)
print("images shape:" + str(images.shape), images.dtype)
print("labels shape:" + str(labels.shape), labels.dtype)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
session_conf = tf.ConfigProto(
    #    device_count={'CPU' : 0, 'GPU' : 1},
    allow_soft_placement=True,
    #    log_device_placement=True
)
# session_conf.gpu_options.allow_growth = True
with tf.device("/CPU:0"):
    with tf.Session(config=session_conf) as sess:
        sess.as_default()
        shape = images.shape
        x = tf.placeholder(tf.float32, [None, shape[1], shape[2], shape[3], 1], name='x')
        valid_seg = tf.placeholder(tf.int32, [None, shape[1], shape[2], shape[3]], name="valid_seg")
        logits = vox_res_net(x, is_train=True, n_out=3)
        # net = vox_res_net(x, is_train=True, n_out=3)
        outputs = [x.outputs for x in logits]
        net = logits[-1]
        out_seg = net.outputs
        print(out_seg, valid_seg.shape)

        # 定义评估函数 监控量
        correct_prediction = tf.equal(tf.cast(tf.argmax(out_seg, 4), tf.int32), valid_seg)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tl.cost.cross_entropy(out_seg, valid_seg, "cost")
        print(loss)
        train_params = net.all_params
        for t in logits[:-1]:
            w = tf.Variable(1.0, name="classifier_w")
            train_params.append(w)
            loss += w * tl.cost.cross_entropy(t.outputs, valid_seg, "cost")
        print("output shape:  "+str(t.outputs.shape))
        cost = tf.contrib.layers.l2_regularizer(1e-4)(net.all_params[-1])
        # cost -= tl.cost.cross_entropy(out_seg, valid_seg, "cost")
        # for output in outputs[:-1]:
        #     cost -= tl.cost.cross_entropy(output,valid_seg,"cost")
        cost += loss
        print(cost)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
        tl.layers.initialize_global_variables(sess)
        net.print_params()
        net.print_layers()

        # 训练网络
        tl.utils.fit(sess, net, train_op, cost, images, labels, x, valid_seg,
                     acc=acc, batch_size=batch_size, n_epoch=n_epoch, print_freq=print_freq,
                     eval_train=False, tensorboard=True, tensorboard_epoch_freq=1, tensorboard_graph_vis=True,
                     tensorboard_weight_histograms=True)

        # print("Starting training network...")
        # for epoch in range(n_epoch):
        #     start_time = time.time()
        #     for X_train_a, y_train_a in tl.iterate.minibatches(images, labels, batch_size, shuffle=True):
        #         feed_dict = {x: X_train_a, valid_seg: y_train_a}
        #         feed_dict.update(net.all_drop)
        #         sess.run(train_op, feed_dict=feed_dict)
        #
        #     if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        #         print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        #         train_loss, train_acc, n_batch = 0, 0, 0
        #         for X_train_a, y_train_a in tl.iterate.minibatches(images, labels, batch_size, shuffle=True):
        #             dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        #             feed_dict = {x: X_train_a, valid_seg: y_train_a}
        #             feed_dict.update(dp_dict)
        #             err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        #             train_loss += err
        #             train_acc += ac
        #             n_batch += 1
        #         print("   train loss: %f" % (train_loss / n_batch))
        #         print("   train acc: %f" % (train_acc/ n_batch))
        #         val_loss, val_acc, n_batch = 0, 0, 0
        #         for X_val_a, y_val_a in tl.iterate.minibatches(img_val, label_val, batch_size, shuffle=True):
        #             dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        #             feed_dict = {x: X_val_a, valid_seg: y_val_a}
        #             feed_dict.update(dp_dict)
        #             err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        #             val_loss += err
        #             val_acc += ac
        #             n_batch += 1
        #         print("   val loss: %f" % (val_loss / n_batch))
        #         print("   val acc: %f" % (val_acc / n_batch))
        #         try:
        #             tl.vis.draw_weights(net.all_params[0].eval(), second=10, saveable=True, shape=[28, 28],
        #                                 name='w1_' + str(epoch + 1), fig_idx=2012)
        #         except:
        #             print("You should change vis.W(), if you want to save the feature images for different dataset")

        # 评估模型
        tl.utils.test(sess, net, acc, img_test, label_test, x, valid_seg, batch_size=batch_size, cost=cost)

        # 模型保存
        tl.files.save_npz(net.all_params, name='model.npz')
        sess.close()
