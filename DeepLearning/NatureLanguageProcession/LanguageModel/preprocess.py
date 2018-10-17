import codecs
import collections
from operator import itemgetter


def preprocess(source, vocab, output):
    counter = collections.Counter()
    with codecs.open(source, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按词频顺序对单词进行排序。
    sorted_word_to_cnt = sorted(
        counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    sorted_words = ["<eos>"] + sorted_words

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
