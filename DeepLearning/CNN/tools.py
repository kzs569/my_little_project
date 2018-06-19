import tensorflow as tf


def conv(name, input_data, out_channel):
    in_channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32)
        biases = tf.get_variable("biases", [out_channel], dtype=tf.float32)
        conv = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
        out = tf.nn.relu(tf.nn.bias_add(conv, biases))
    return out


def fc(name, input_data, out_channel):
    shape = input_data.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    input_data_flat = tf.reshape(input_data, [-1, size])
    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights", shape=[size, out_channel], dtype=tf.float32)
        biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32)
        out = tf.nn.relu(tf.nn.bias_add(tf.matmul(input_data_flat, weights), biases))
    return out


def maxpool(name, input_data):
    return tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name=name)
