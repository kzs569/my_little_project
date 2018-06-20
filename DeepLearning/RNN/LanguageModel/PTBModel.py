import tensorflow as tf
import configuration


# 通过一个PTBModel类来描述模型，这样方便维护循环神经网络中的状态。
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度。
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义每一步的输入和预期输出。两者的维度都是[batch_size, num_steps]。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        self.is_training = is_training

        # 将输入单词转化为词向量。
        with tf.variable_scope('embedding'):
            # 定义单词的词向量矩阵。
            embedding = tf.get_variable("embedding", [configuration.VOCAB_SIZE, configuration.HIDDEN_SIZE])

            # 将输入单词转化为词向量。
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

            # 只在训练时使用dropout。
            if is_training:
                inputs = tf.nn.dropout(inputs, configuration.EMBEDDING_KEEP_PROB)

        output, state = self.build_rnn_graph_lstm(inputs)

        # Softmax层：将RNN在每个位置上的输出转化为各个单词的logits。
        with tf.variable_scope('softmax'):
            if configuration.SHARE_EMB_AND_SOFTMAX:
                weight = tf.transpose(embedding)
            else:
                weight = tf.get_variable("weight", [configuration.HIDDEN_SIZE, configuration.VOCAB_SIZE])
            bias = tf.get_variable("bias", [configuration.VOCAB_SIZE])

            logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数和平均损失。
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits)
        tf.summary.scalar('loss', tf.reduce_sum(loss))

        self.cost = tf.reduce_sum(loss) / self.batch_size
        tf.summary.scalar('cost', self.cost)

        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not self.is_training:
            return

        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), configuration.MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        self.merged_summary = tf.summary.merge_all()

    def build_rnn_graph_lstm(self, inputs):
        def make_cell():

            dropout_keep_prob = configuration.LSTM_KEEP_PROB if self.is_training else 1.0

            lstm_cells = [
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.BasicLSTMCell(configuration.HIDDEN_SIZE),
                    output_keep_prob=dropout_keep_prob)
                for _ in range(configuration.NUM_LAYERS)]

            cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(configuration.NUM_LAYERS)], state_is_tuple=True)

        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        state = self.initial_state
        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs,
        #                                   initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, configuration.HIDDEN_SIZE])
        return output, state
