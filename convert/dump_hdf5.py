import tensorflow as tf
import math
import h5py
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from model.I3D_TF import InceptionI3d


def make_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [math.floor(int(conv_shape[0]) / 2), math.floor(int(conv_shape[1]) / 2)]
    else:
        sys.exit('Invalid padding name ' + padding_name)


def dump_conv3d(sess, modality, target_dir, name='Conv3d_1a_7x7'):
    network_root = modality + 'inception_i3d/'
    conv_operation = sess.graph.get_operation_by_name(
        network_root + name + '/conv_3d/convolution')  # remplacer convolution par Conv2D si erreur

    weights_tensor = sess.graph.get_tensor_by_name(network_root + name + '/conv_3d/w:0')
    weights = weights_tensor.eval(session=sess)

    padding = make_padding(conv_operation.get_attr('padding'), weights_tensor.get_shape())
    strides = conv_operation.get_attr('strides')

    # conv_out = sess.graph.get_operation_by_name(network_root + name + '/conv_3d/convolution').outputs[0].eval(session=sess) # remplacer convolution par Conv2D si erreur

    beta = sess.graph.get_tensor_by_name(network_root + name + '/batch_norm/beta:0').eval(session=sess)
    # gamma = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/gamma:0').eval(session=sess)
    mean = sess.graph.get_tensor_by_name(network_root + name + '/batch_norm/moving_mean:0').eval(session=sess)
    var = sess.graph.get_tensor_by_name(network_root + name + '/batch_norm/moving_variance:0').eval(session=sess)

    # relu_out = sess.graph.get_operation_by_name(network_root+name+'/Relu').outputs[0].eval(session=sess)

    os.system('mkdir -p ' + target_dir  + name)
    h5f = h5py.File(target_dir + name + "/" + name.split('/')[-1] + '.h5', 'w')

    h5f.create_dataset("weights", data=weights)
    h5f.create_dataset("strides", data=strides)
    h5f.create_dataset("padding", data=padding)
    # h5f.create_dataset("conv_out", data=conv_out)
    # batch norm
    h5f.create_dataset("beta", data=beta)
    # h5f.create_dataset("gamma", data=gamma)
    h5f.create_dataset("mean", data=mean)
    h5f.create_dataset("var", data=var)
    # h5f.create_dataset("relu_out", data=relu_out)
    h5f.close()

def dump_Mixed_5b(sess, modality, target_dir='rgb_imagenet/', name='Mixed_5b'):
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_0/Conv3d_0a_1x1')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_1/Conv3d_0a_1x1')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_1/Conv3d_0b_3x3')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_2/Conv3d_0a_1x1')

    # I think Conv3d_0a_3x3 is author's typo
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_2/Conv3d_0a_3x3')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_3/Conv3d_0b_1x1')

def dump_Logits(sess, modality, target_dir="flow_imagenet/", name='Logits/Conv3d_0c_1x1'):
    network_root = modality + 'inception_i3d/'
    conv_operation = sess.graph.get_operation_by_name(
        network_root + name + '/conv_3d/convolution')  # remplacer convolution par Conv2D si erreur

    weights_tensor = sess.graph.get_tensor_by_name(network_root + name + '/conv_3d/w:0')
    weights = weights_tensor.eval(session=sess)

    bias_tensor = sess.graph.get_tensor_by_name(network_root + name + '/conv_3d/b:0')
    bias = bias_tensor.eval(session=sess)

    padding = make_padding(conv_operation.get_attr('padding'), weights_tensor.get_shape())
    strides = conv_operation.get_attr('strides')

    # conv_out = sess.graph.get_operation_by_name(network_root + name + '/conv_3d/convolution').outputs[0].eval(session=sess) # remplacer convolution par Conv2D si erreur

    # beta = sess.graph.get_tensor_by_name(network_root+name+'/batch_norm/beta:0').eval(session=sess)
    # gamma = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/gamma:0').eval(session=sess)
    # mean = sess.graph.get_tensor_by_name(network_root+name+'/batch_norm/moving_mean:0').eval(session=sess)
    # var = sess.graph.get_tensor_by_name(network_root+name+'/batch_norm/moving_variance:0').eval(session=sess)

    # relu_out = sess.graph.get_operation_by_name(network_root+name+'/Relu').outputs[0].eval(session=sess)

    os.system('mkdir -p ' + target_dir + name)
    h5f = h5py.File(target_dir + name + "/" + name.split('/')[-1] + '.h5', 'w')
    # conv
    h5f.create_dataset("weights", data=weights)
    h5f.create_dataset("strides", data=strides)
    h5f.create_dataset("padding", data=padding)
    h5f.create_dataset("bias", data=bias)
    # h5f.create_dataset("conv_out", data=conv_out)
    # batch norm
    # h5f.create_dataset("beta", data=beta)
    # h5f.create_dataset("gamma", data=gamma)
    # h5f.create_dataset("mean", data=mean)
    # h5f.create_dataset("var", data=var)
    # h5f.create_dataset("relu_out", data=relu_out)
    h5f.close()

def dump_Mixed(sess, modality, target_dir='flow_imagenet/', name='Mixed_3b'):
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_0/Conv3d_0a_1x1')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_1/Conv3d_0a_1x1')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_1/Conv3d_0b_3x3')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_2/Conv3d_0a_1x1')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_2/Conv3d_0b_3x3')
    dump_conv3d(sess, modality, target_dir, name=name + '/Branch_3/Conv3d_0b_1x1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Parmeters
    parser.add_argument('--checkpoints_dir', type=str, default='../data/checkpoints/',
                        help='source directory')
    parser.add_argument('--target_dir', type=str, default=None,
                        help='source directory')
    parser.add_argument('--modality', type=str, default='rgb_imagenet',
                        help='rgb_scratch , rgb_imagenet, flow_scratch, flow_imagenet')

    args = parser.parse_args()

    if args.target_dir is None:
        target_dir = "../data/dump/" + args.modality + "/"
    else:
        target_dir = args.target_dir

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)


    if 'rgb' in args.modality:
        modality = 'RGB/'
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, 64, 224, 224, 3))

        with tf.variable_scope('RGB'):
            rgb_model = InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        rgb_saver.restore(sess, args.checkpoints_dir + args.modality + "/model.ckpt")


    elif 'flow' in args.modality:
        modality = 'Flow/'

        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(1, 64, 224, 224, 2))
        with tf.variable_scope('Flow'):
            flow_model = InceptionI3d(
                400, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
        flow_saver.restore(sess, args.checkpoints_dir + args.modality + "/model.ckpt")



    dump_conv3d(sess, modality, target_dir, 'Conv3d_1a_7x7')
    dump_conv3d(sess, modality, target_dir, 'Conv3d_2b_1x1')
    dump_conv3d(sess, modality, target_dir, 'Conv3d_2c_3x3')
    dump_Mixed(sess, modality, target_dir, name='Mixed_3b')
    dump_Mixed(sess, modality, target_dir, name='Mixed_3c')
    dump_Mixed(sess, modality, target_dir, name='Mixed_4b')
    dump_Mixed(sess, modality, target_dir, name='Mixed_4c')
    dump_Mixed(sess, modality, target_dir, name='Mixed_4d')
    dump_Mixed(sess, modality, target_dir, name='Mixed_4e')
    dump_Mixed(sess, modality, target_dir, name='Mixed_4f')
    dump_Mixed_5b(sess, modality, target_dir)
    dump_Mixed(sess, modality, target_dir, name='Mixed_5c')
    dump_Logits(sess, modality, target_dir, 'Logits/Conv3d_0c_1x1')