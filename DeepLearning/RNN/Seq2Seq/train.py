# coding: utf-8
from data import MakeSrcTrgDataset
from model import NMTModel
import tensorflow as tf

# 假设输入数据已经用9.2.1小节中的方法转换成了单词编号的格式。
SRC_TRAIN_DATA = "./../tmp/en-zh.en.out"  # 源语言输入文件。
TRG_TRAIN_DATA = "./../tmp/en-zh.zh.out"  # 目标语言输入文件。
CHECKPOINT_PATH = "./../model/seq2seq_ckpt"  # checkpoint保存路径。
TENSORBOARD_PATH = "./../log/Seq2Seq/"

HIDDEN_SIZE = 1024  # LSTM的隐藏层规模。
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数。
SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小。
TRG_VOCAB_SIZE = 4000  # 目标语言词汇表大小。
BATCH_SIZE = 100  # 训练数据batch的大小。
NUM_EPOCH = 5  # 使用训练数据的轮数。
KEEP_PROB = 0.8  # 节点不被dropout的概率。
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数。

MAX_LEN = 50  # 限定句子的最大单词数量。
SOS_ID = 1  # 目标语言词汇表中<sos>的ID。


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    tf.reset_default_graph()

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        train_model = NMTModel()

    # 定义输入数据。
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    # 训练模型。
    saver = tf.train.Saver()
    # 将当前的计算图输出到tensorboard日志文件
    writer = tf.summary.FileWriter(TENSORBOARD_PATH, graph=tf.get_default_graph())
    step = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer, options=run_options, run_metadata=run_metadata)
            while True:
                try:
                    # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供。
                    cost, _ = sess.run([cost_op, train_op], options=run_options, run_metadata=run_metadata)
                    summary = sess.run(train_model.merged)
                    if step % 10 == 0:
                        print("After %d steps, per token cost is %.3f" % (step, cost))
                    # 每200步保存一个checkpoint。
                    if step % 200 == 0:
                        saver.save(sess, CHECKPOINT_PATH, global_step=step)
                    writer.add_summary(summary, step)
                    writer.add_run_metadata(run_metadata=run_metadata, tag='step%03d' % step, global_step=step)
                    step += 1
                except tf.errors.OutOfRangeError:
                    break
    writer.close()


if __name__ == "__main__":
    main()
