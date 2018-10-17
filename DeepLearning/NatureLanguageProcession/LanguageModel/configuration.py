# 数据相关参数
TRAIN_SETS_PATH = "/mnt/datasets/ptb/simple-examples/data/ptb.train.txt"
TEST_SETS_PATH = "/mnt/datasets/ptb/simple-examples/data/ptb.test.txt"
VALID_SETS_PATH = "/mnt/datasets/ptb/simple-examples/data/ptb.valid.txt"

TRAIN_VOCAB_OUTPUT = './../tmp/ptb.train.vocab'
TEST_VOCAB_OUTPUT = './../tmp/ptb.test.vocab'
VALID_VOCAB_OUTPUT = './../tmp/ptb.valid.vocab'

TRAIN_DATA_OUTPUT = './../tmp/ptb.train'
TEST_DATA_OUTPUT = './../tmp/ptb.test'
VALID_DATA_OUTPUT = './../tmp/ptb.valid'

# 模型相关参数
HIDDEN_SIZE = 300  # 隐藏层规模
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数
VOCAB_SIZE = 10000  # 词典规模
TRAIN_BATCH_SIZE = 20  # 训练数据batch的大小
TRAIN_NUM_STEP = 35  # 训练数据截断长度

# train相关参数
EVAL_BATCH_SIZE = 1  # 测试数据batch的大小
EVAL_NUM_STEP = 1  # 测试数据截断长度
NUM_EPOCH = 50  # 使用训练数据的轮数
LSTM_KEEP_PROB = 0.9  # LSTM节点不被dropout的概率
EMBEDDING_KEEP_PROB = 0.9  # 词向量不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数

# 模型存储地址
CHECKPOINT_PATH = './model/lmodel_cpkt'

# 相对应tensorboard存储地址
TENSORBOARD_PATH = '/home/robinkong/Documents/JupyterNotebook/log'
