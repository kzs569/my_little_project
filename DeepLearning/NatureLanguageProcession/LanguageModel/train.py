# coding: utf-8
from preprocess import preprocess
import configuration
from PTBModel import PTBModel
from data_read import make_batches, read_data
import numpy as np
import tensorflow as tf
import time

preprocess(configuration.TRAIN_SETS_PATH, configuration.TRAIN_VOCAB_OUTPUT, configuration.TRAIN_DATA_OUTPUT)
preprocess(configuration.TEST_SETS_PATH, configuration.TEST_VOCAB_OUTPUT, configuration.TEST_DATA_OUTPUT)
preprocess(configuration.VALID_SETS_PATH, configuration.VALID_VOCAB_OUTPUT, configuration.VALID_DATA_OUTPUT)

TRAIN_DATA = configuration.TRAIN_DATA_OUTPUT  # 训练数据路径。
EVAL_DATA = configuration.VALID_DATA_OUTPUT  # 验证数据路径。
TEST_DATA = configuration.TEST_DATA_OUTPUT  # 测试数据路径。


# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值。
def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均perplexity的辅助变量。
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # 训练一个epoch。
    for x, y in batches:
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单
        # 词为给定单词的概率。
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y,
             model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        # 只有在训练时输出日志。
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                step, np.exp(total_costs / iters)))
        step += 1

    # 返回给定模型在给定数据上的perplexity值。
    return step, np.exp(total_costs / iters)


def main():
    tf.reset_default_graph()
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, configuration.TRAIN_BATCH_SIZE, configuration.TRAIN_NUM_STEP)

    # with tf.variable_scope("language_model", reuse=True, initializer=initializer):
    #     valid_model = PTBModel(is_training=False,
    #                                batch_size=configuration.EVAL_BATCH_SIZE,
    #                                num_steps=configuration.EVAL_NUM_STEP)
    #
    # with tf.name_scope("Test"):
    #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #         test_model = PTBModel(is_training=False,
    #                                   batch_size=configuration.EVAL_BATCH_SIZE,
    #                                   num_steps=configuration.EVAL_NUM_STEP)

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(configuration.TENSORBOARD_PATH, tf.get_default_graph())

    # sv = tf.train.Supervisor(logdir=configuration.TENSORBOARD_PATH,
    #                          init_op=configuration.CHECKPOINT_PATH)
    # saver = sv.saver

    # 训练模型。
    with tf.Session() as session:

        tf.global_variables_initializer().run()

        train_batches = make_batches(
            read_data(TRAIN_DATA),
            configuration.TRAIN_BATCH_SIZE,
            configuration.TRAIN_NUM_STEP)

        print('train data info:', len(train_batches))

        eval_batches = make_batches(
            read_data(EVAL_DATA), configuration.EVAL_BATCH_SIZE, configuration.EVAL_NUM_STEP)
        test_batches = make_batches(
            read_data(TEST_DATA), configuration.EVAL_BATCH_SIZE, configuration.EVAL_NUM_STEP)

        step = 0
        for i in range(configuration.NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            output_log = True
            # 计算平均perplexity的辅助变量。
            total_costs = 0.0
            iters = 0

            state = session.run(train_model.initial_state)
            # 训练一个epoch。
            for x, y in train_batches:
                # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单
                # 词为给定单词的概率。
                cost, state, _ = session.run(
                    [train_model.cost, train_model.final_state, train_model.train_op],
                    {train_model.input_data: x, train_model.targets: y, train_model.initial_state: state})
                merged = session.run(train_model.merged)
                total_costs += cost
                iters += train_model.num_steps

                writer.add_summary(merged, step)
                # 只有在训练时输出日志。
                if output_log and step % 100 == 0:
                    print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
                    if (step + 1) % 2000 == 0:
                        print("---------Saver saving----------")
                        saver.save(session, configuration.CHECKPOINT_PATH, global_step=step)
                step += 1

            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, np.exp(total_costs / iters)))

        #     _, eval_pplx = run_epoch(session, valid_model, eval_batches,
        #                              tf.no_op(), False, 0)
        #     print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))
        #
        # _, test_pplx = run_epoch(session, test_model, test_batches,
        #                          tf.no_op(), False, 0)
        # print("Test Perplexity: %.3f" % test_pplx)

    writer.close()


if __name__ == "__main__":
    main()
