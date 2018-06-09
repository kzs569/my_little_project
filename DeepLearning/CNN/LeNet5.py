# coding: utf-8


# 导入数据集mnist
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./../data/MNIST/", one_hot=True)

import tensorflow as tf
import os

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABEL = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的结点个数
FC_SIZE = 512


# 定义卷积神经网络的前向传播过程。这里添加了一个新的参数train，用于区分训练过程和测试过程。在这个程序中将用到dropout方法，
# dropout方法可进一步提升模型的可靠性并防止过拟合，dropout过程只在训练时使用
def inference(input_tensor, regularizer, train=True, ):
    # 声明第一层卷积层的变量并实现前向传播过程。通过使用不同命名空间来隔离不同层的变量，让每一层中的变量命名只需要考虑在当前层的作用，
    # 不需担心重命名的问题。第一层输出为28×28×32的张量

    input_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])

    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.0))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABEL],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [NUM_LABEL], initializer=tf.constant_initializer(0.0))

        logit = tf.nn.matmul(fc1, fc2_weights) + fc2_biases

    return logit


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model'
MODEL_NAME = 'LeNet5.ckpt'

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
y = inference(x, regularizer=regularizer)
global_step = tf.Variable(0, trainable=False)

variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())
cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)))

learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,
                                           global_step=global_step,
                                           decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                           decay_rate=LEARNING_RATE_DECAY)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean, global_step=global_step)

with tf.control_dependencies([train_step, variable_averages_op]):
    train_op = tf.no_op(name='train')

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(TRAINING_STEPS):
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

        if i % 1000 == 0:
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
