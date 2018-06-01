#!/usr/bin/python3
# coding:utf-8

# 评估CIFAR-10模型的预测性能

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf

from cifar10 import cifar_10

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
# 单次评估
def eval_once(saver, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        # checkpoint文件会记录保存信息,通过它可以定位最新保存的模型
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 从检查点恢复
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 假设model_checkpoint_path为/my-favorite-path/cifar10_train/model.ckpt-0从中提取global_step
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        # 启动队列协调器
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            # 统计正确预测的数量
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            # 检查是否被请求停止
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            # 计算准确度　precision@1
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        # pylint: disable=broad-except
        except Exception as e:
            coord.request_stop(e)
        # 请求线程结束
        coord.request_stop()
        # 等待线程终止
        coord.join(threads, stop_grace_period_secs=10)

# 评估CIFAR-10
def evaluate():
    with tf.Graph().as_default() as g:
        # 获取CIFAR-10的图像和标签
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar_10.inputs(eval_data=eval_data)
        # 构建一个图表,用于计算推理模型中的logits预测
        logits = cifar_10.inference(images)
        # 计算预测
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        # 为eval恢复学习变量的移动平均
        variable_averages = tf.train.ExponentialMovingAverage(cifar_10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        # 创建一个saver对象,用于保存参数到文件中
        saver = tf.train.Saver(variables_to_restore)
        # 根据摘要TF集合构建摘要操作
        summary_op = tf.summary.merge_all()
        # 将Summary protocol buffers写入事件文件
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

# pylint: disable=unused-argument
def main(argv=None):
    cifar_10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()