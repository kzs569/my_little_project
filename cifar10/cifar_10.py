#!/usr/bin/python3
# coding:utf-8

# 建立CIFAR-10的模型
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from cifar10 import cifar10_input


FLAGS = tf.app.flags.FLAGS
# 基本模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/w/mycode/data/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,# 半精度浮点数
                            """Train the model using fp16.""")
# 描述CIFAR-10数据集的全局常量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# 描述训练过程的常量
MOVING_AVERAGE_DECAY = 0.9999     # 滑动平均衰减率
NUM_EPOCHS_PER_DECAY = 350.0      # 在学习速度衰退之后的Epochs
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习速率衰减因子
INITIAL_LEARNING_RATE = 0.1       # 初始学习率

# 如果模型使用多个GPU进行训练,则使用tower_name将所有Op名称加前缀以区分操作
# 可视化模型时从摘要名称中删除此前缀
TOWER_NAME = 'tower'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# 激活摘要创建助手
def _activation_summary(x):
    # 若多个GPU训练,则从名称中删除'tower_[0-9]/',利于TensorBoard显示
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    # 提供激活直方图的summary
    tf.summary.histogram(tensor_name + '/activations', x)
    # 衡量激活稀疏性的summary
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

# 创建存储在CPU内存上的变量(变量的名称,整数列表,变量的初始化程序)
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

# 创建一个权重衰减的初始化变量(变量的名称,整数列表,截断高斯的标准差,加L2Loss权重衰减)
# 变量用截断正态分布初始化的.只有指定时才添加权重衰减
def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    # 用截断正态分布进行初始化
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
    if wd is not None:
        # wd用于向losses添加L2正则化,防止过拟合,提高泛化能力
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        # 把变量放入一个集合
        tf.add_to_collection('losses', weight_decay)
    return var


# -------------------------模型输入-----------------------------------
# 训练输入
# 返回：images:[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]; labels:[batch_size]
def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    # 读入并增广数据
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

# 预测输入
# 返回：images:[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]; labels:[batch_size]
def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    # 图像预处理及输入
    images, labels = cifar10_input.inputs(eval_data=eval_data,data_dir=data_dir,batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels
# -------------------------------------------------------------------


# -------------------------模型预测-----------------------------------
# 构建CIFAR-10模型
# 使用tf.get_variable()而不是tf.Variable()来实例化所有变量,以便跨多个GPU训练运行共享变量
# 若只在单个GPU上运行,则可通过tf.Variable()替换tf.get_variable()的所有实例来简化此功能
def inference(images):
    # 卷积层1
    with tf.variable_scope('conv1') as scope:
        # weight不进行L2正则化
        kernel = _variable_with_weight_decay('weights',shape=[5, 5, 3, 64],stddev=5e-2, wd=None)
        # 卷积
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        # biases初始化为0
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        # 卷积层1的结果由ReLu激活
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # 汇总
        _activation_summary(conv1)
    # 池化层1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    # lrn层1　局部响应归一化:增强大的抑制小的,增强泛化能力
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # 卷积层2
    with tf.variable_scope('conv2') as scope:
        # weight不进行L2正则化
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        # biases初始化为0.1
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        # 卷积层2的结果由ReLu激活
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # 汇总
        _activation_summary(conv2)
    # lrn层2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # 池化层2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # 全连接层3
    with tf.variable_scope('local3') as scope:
        # 将样本转换为一维向量
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        # 维数
        dim = reshape.get_shape()[1].value
        # 添加L2正则化约束,防止过拟合
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        # biases初始化为0.1
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        # ReLu激活
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # 全连接层4
    with tf.variable_scope('local4') as scope:
        # 添加L2正则化约束,防止过拟合
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        # biases初始化为0.1
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        # ReLu激活
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # 线性层
    # (WX+b)不使用softmax,因为tf.nn.sparse_softmax_cross_entropy_with_logits接受未缩放的logits并在内部执行softmax以提高效率
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=None)
        # biases初始化为0
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        # (WX+b) 进行线性变换以输出 logits
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        # 汇总
        _activation_summary(softmax_linear)
    return softmax_linear
# -------------------------------------------------------------------


# -------------------------模型训练-----------------------------------
# 将L2损失添加到所有可训练变量
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    # 计算logits和labels之间的交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
    # 计算整个批次的平均交叉熵损失
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # 把变量放入一个集合
    tf.add_to_collection('losses', cross_entropy_mean)
    # 总损失定义为交叉熵损失加上所有的权重衰减项(L2损失)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# 添加损失的summary;计算所有单个损失的移动均值和总损失
def _add_loss_summaries(total_loss):
    # 指数移动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    # 将指数移动平均应用于单个损失
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # 单个损失损失和全部损失的标量summary
    for l in losses + [total_loss]:
        # 将每个损失命名为raw,并将损失的移动平均命名为原始损失
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

# 训练CIFAR-10模型
# 创建一个优化器并应用于所有可训练变量,为所有可训练变量添加移动均值(全部损失,训练步数)
def train(total_loss, global_step):
    # 影响学习率的变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # 指数衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    # 对总损失进行移动平均
    loss_averages_op = _add_loss_summaries(total_loss)
    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # 应用处理过后的梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # 为可训练变量添加直方图
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    # 为梯度添加直方图
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    # 跟踪所有可训练变量的移动均值
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 使用默认图形的包装器
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]): train_op = tf.no_op(name='train')
    return train_op
# -------------------------------------------------------------------

# 下载并解压数据
def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)