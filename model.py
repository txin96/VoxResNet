import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import nibabel as nib


def vox_res_module(x, prefix, is_train=True, reuse=False):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    bn1 = BatchNormLayer(x, act=tf.nn.relu, is_train=is_train, name=prefix + "bn1")
    conv1 = Conv3dLayer(bn1, shape=[1, 3, 3, 64, 64], strides=[1, 1, 1, 1, 1], W_init=w_init, name=prefix + "conv1")
    bn2 = BatchNormLayer(conv1, act=tf.nn.relu, is_train=is_train, name=prefix + "bn2")
    conv2 = Conv3dLayer(bn2, shape=[3, 3, 3, 64, 64], strides=[1, 1, 1, 1, 1], W_init=w_init, name=prefix + "conv2")
    out = ElementwiseLayer(layers=[x, conv2], combine_fn=tf.add, name=prefix + "out")
    return out


def vox_res_net(x, is_train=True, reuse=False, n_out=3):
    with tf.variable_scope("VoxResNet", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        inputs = InputLayer(x, name="input")
        conv1a = Conv3dLayer(inputs, shape=[3, 3, 3, 1, 32], strides=[1, 1, 1, 1, 1], W_init=w_init, name="conv1a")
        bn1 = BatchNormLayer(conv1a, act=tf.nn.relu, is_train=is_train, name="bn1")
        conv1b = Conv3dLayer(bn1, shape=[1, 3, 3, 32, 32], strides=[1, 1, 1, 1, 1], W_init=w_init, name="conv1b")
        bn2 = BatchNormLayer(conv1b, act=tf.nn.relu, is_train=is_train, name="bn2")
        conv1c = Conv3dLayer(bn2, shape=[3, 3, 3, 32, 64], strides=[1, 2, 2, 2, 1], W_init=w_init, name="conv1c")
        res1a = vox_res_module(conv1c, "res1a-", is_train)
        res2a = vox_res_module(res1a, "res2a-", is_train)
        bn3 = BatchNormLayer(res2a, act=tf.nn.relu, is_train=is_train, name="bn3")
        conv4 = Conv3dLayer(bn3, shape=[3, 3, 3, 64, 64], strides=[1, 2, 2, 2, 1], W_init=w_init, name="conv4")
        res3a = vox_res_module(conv4, "res3a-", is_train)
        res4a = vox_res_module(res3a, "res4a-", is_train)
        bn4 = BatchNormLayer(res4a, act=tf.nn.relu, is_train=is_train, name="bn4")
        conv7 = Conv3dLayer(bn4, shape=[3, 3, 3, 64, 64], strides=[1, 2, 2, 2, 1], W_init=w_init, name="conv7")
        res5a = vox_res_module(conv7, "res5a-", is_train)
        res6a = vox_res_module(res5a, "res6a-", is_train)
        inputs_shape = tf.shape(x)
        decon_output_shape = [inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3], 4]
        decon0a = DeConv3dLayer(conv1b, shape=[3, 3, 3, 4, 32], output_shape=decon_output_shape,
                                strides=[1, 1, 1, 1, 1], W_init=w_init, name="decon0a")
        decon1a = DeConv3dLayer(res2a, shape=[2, 2, 2, 4, 64], output_shape=decon_output_shape, strides=[1, 2, 2, 2, 1],
                                W_init=w_init, name="decon1a")
        decon2a = DeConv3dLayer(res4a, shape=[4, 4, 4, 4, 64], output_shape=decon_output_shape, strides=[1, 4, 4, 4, 1],
                                W_init=w_init, name="decon2a")
        decon3a = DeConv3dLayer(res6a, shape=[8, 8, 8, 4, 64], output_shape=decon_output_shape, strides=[1, 8, 8, 8, 1],
                                W_init=w_init, name="decon3a")
        classifier0a = Conv3dLayer(decon0a, shape=[1, 1, 1, 4, n_out], strides=[1, 1, 1, 1, 1], W_init=w_init,
                                   name="classifier0a")
        classifier1a = Conv3dLayer(decon1a, shape=[1, 1, 1, 4, n_out], strides=[1, 1, 1, 1, 1], W_init=w_init,
                                   name="classifier1a")
        classifier2a = Conv3dLayer(decon2a, shape=[1, 1, 1, 4, n_out], strides=[1, 1, 1, 1, 1], W_init=w_init,
                                   name="classifier2a")
        classifier3a = Conv3dLayer(decon3a, shape=[1, 1, 1, 4, n_out], strides=[1, 1, 1, 1, 1], W_init=w_init,
                                   name="classifier3a")
        out = ElementwiseLayer(layers=[classifier0a, classifier1a, classifier2a, classifier3a], combine_fn=tf.add,
                               name="out")
    if is_train:
        return [classifier0a, classifier1a, classifier2a, classifier3a, out]
    else:
        return out
    #     out = Conv3dLayer(inputs, shape=[3, 3, 3, 1, 3], strides=[1, 1, 1, 1, 1], W_init=w_init, name="test")
    #     return out
