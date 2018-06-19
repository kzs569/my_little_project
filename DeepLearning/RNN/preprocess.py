from moses import MosesTokenizer
import jieba
import codecs
import collections
from operator import itemgetter

SRC_TRAIN_DATA = '/mnt/datasets/en-zh/train.tags.en-zh.en'
TRG_TRAIN_DATA = '/mnt/datasets/en-zh/train.tags.en-zh.zh'

SRC_VOCAB_PATH = './tmp/en-zh.en.vocab'
TRG_VOCAB_PATH = './tmp/en-zh.zh.vocab'

SRC_OUTPUT_PATH = './tmp/en-zh.en.out'
TRG_OUTPUT_PATH = './tmp/en-zh.zh.out'

CHECKPOINT_PATH = './model/seq2seq_ckpt'


# type : zh-jieba, en-moses
def preprocess(source, vocab, output, type):
    counter = collections.Counter()
    with codecs.open(source, "r", "utf-8") as f:
        for line in f:
            if type == 'zh':
                for word in jieba.lcut(line):
                    counter[word] += 1
            elif type == 'en':
                m = MosesTokenizer()
                for word in m.tokenize(line):
                    counter[word] += 1
            else:
                print('segment is failed!')
                return

    # 按词频顺序对单词进行排序。
    sorted_word_to_cnt = sorted(
        counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    sorted_words = ["<eos>", "<sos>", "<unk>"] + sorted_words

    if type == 'en':
        if len(sorted_words) > 10000:
            sorted_words = sorted_words[:10000]
    elif type == 'zh':
        if len(sorted_words) > 4000:
            sorted_words = sorted_words[:4000]

    with codecs.open(vocab, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")

    # 读取词汇表，并建立词汇到单词编号的映射。
    with codecs.open(vocab, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了不在词汇表内的低频词，则替换为"unk"。
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(source, "r", "utf-8")
    fout = codecs.open(output, 'w', 'utf-8')
    for line in fin:
        words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
        # 将每个单词替换为词汇表中的编号
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()
    print(type, "preprocession is completed!!!")


if __name__ == '__main__':
    preprocess(SRC_TRAIN_DATA, SRC_VOCAB_PATH, SRC_OUTPUT_PATH, 'en')
    preprocess(TRG_TRAIN_DATA, TRG_VOCAB_PATH, TRG_OUTPUT_PATH, 'zh')
