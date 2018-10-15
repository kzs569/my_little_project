import numpy as np
import configuration
# 从文件中读取数据，并返回包含单词编号的数组。
def read_data(file_path):
    with open(file_path, "r") as fin:
        # 将整个文档读进一个长字符串。
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转为整数
    return id_list


def make_batches(id_list, batch_size, num_step):
    # 计算总的batch数量。每个batch包含的单词数量是batch_size * num_step。
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    # 如9-4图所示，将数据整理成一个维度为[batch_size, num_batches * num_step]
    # 的二维数组。
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分成num_batches个batch，存入一个数组。
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位。这里得到的是RNN每一步输出所需要预测的
    # 下一个单词。
    label = np.array(id_list[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)
    # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵。
    return list(zip(data_batches, label_batches))


if __name__ == '__main__':
    print(make_batches(read_data(configuration.TRAIN_DATA_OUTPUT),
                       configuration.TRAIN_BATCH_SIZE,
                       configuration.TRAIN_NUM_STEP))
