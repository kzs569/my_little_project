## ELMo
- 数据集

    [1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/)

    [paper](http://arxiv.org/abs/1312.3005) | [code](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [data](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz) | [output probabilities](http://www.statmt.org/lm-benchmark/output.tar)

    介绍

    在这个版本中，我们开源了一个在十亿字基准上训练的模型（http://arxiv.org/abs/1312.3005)，这是一种英文的大语言语料库，于2013年发布。该数据集包含约十亿字，并且具有大约800K字的词汇大小。它主要包含新闻数据。由于训练集中的句子被洗牌，模型可以忽略上下文并集中于句子级语言建模。

    在原始版本和后续工作中，人们使用相同的测试集来训练该数据集上的模型，作为语言建模的标准基准。最近，我们写了一篇文章（http://arxiv.org/abs/1602.02410)，描述了字符CNN，一个大而深的LSTM和一个特定的Softmax架构之间的模型混合，使我们能够在这个数据集上训练最好的模型远远超过了其他人以前获得的最好的困惑。

    代码发布

    开源组件包括：

    TensorFlow GraphDef原始缓冲区文本文件。

    TensorFlow预培训的检查点碎片。

    用于评估预训练模型的代码。

    词汇文件。

    测试仪从LM-1B评估。

    代码支持4种评估模式：

    1. 给定提供的数据集，计算模型的困惑。
    2. 给定一个前缀句子，预测下一个单词。
    3. 转储softmax嵌入，字符级CNN字嵌入。
    4. 给一个句子，从LSTM状态转储嵌入。

- Background
1. Character-Aware Neural Language Models

    pdf : [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)

    Github : [tf-lstm-char-cnn](https://github.com/mkroutikov/tf-lstm-char-cnn)

    [how-to-use-elmo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)

    [Character-Aware Neural Language Models](https://zhuanlan.zhihu.com/p/21242454)

    [CS224n研究热点10 Character-Aware神经网络语言模型](http://www.hankcs.com/nlp/cs224n-character-aware-neural-language-models.html)



- 调用流程

commands/elmo.py : https://github.com/allenai/allennlp/blob/master/allennlp/commands/elmo.py

``` python
from allennlp.commands.elmo import ElmoEmbedder
import scipy

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = ElmoEmbedder(options_file, weight_file)
context_tokens = [['I', 'love', 'you', '.'], ['Sorry', ',', 'I', 'don', "'t", 'love', 'you', '.']]
elmo_embedding, elmo_mask = elmo.batch_to_embeddings(context_tokens)

print(elmo_embedding)
print(elmo_mask)

tokens = ["I", "ate", "an", "apple", "for", "breakfast"]

# embed_sentence(),embed_batch(),embed_sentences(),embed_file()都是调用batch_to_embeddings()

vectors = elmo.embed_sentence(tokens)

print(len(vectors), len(vectors[0]))

vectors2 = elmo.embed_sentence(["I", "ate", "a", "carrot", "for", "breakfast"])

scipy.spatial.distance.cosine(vectors[2][3],vectors2[2][3])
```

``` python
    """
    ElmoEmbedder的核心function(),对比后可以发现，就是调用allennlp.modules.elmo中的内容。
    """
    def batch_to_embeddings(self, batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        Returns
        -------
            A tuple of tensors, the first representing activations (batch_size, 3, num_timesteps, 1024) and
        the second a mask (batch_size, num_timesteps).
        """
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # without_bos_eos is a 3 element list of (activation, mask) tensor pairs,
        # each with size (batch_size, num_timesteps, dim and (batch_size, num_timesteps)
        # respectively.
        without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
                           for layer in layer_activations]
        # Converts a list of pairs (activation, mask) tensors to a single tensor of activations.
        activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
        # The mask is the same for each ELMo vector, so just take the first.
        mask = without_bos_eos[0][1]

        return activations, mask
```

modules/elmo.py https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L27

``` python
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

print(character_ids)

embeddings = elmo(character_ids)

print(embeddings)
```

``` python
def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
    """
    Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
    (len(batch), max sentence length, max word length).
    Parameters
    ----------
    batch : ``List[List[str]]``, required
        A list of tokenized sentences.
    Returns
    -------
        A tensor of padded character ids.
    """
    instances = []
    indexer = ELMoTokenCharactersIndexer()
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens,
                          {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']
```

上述两种调用方式本质是相同的，都是将sentences[[]]首先通过batch_to_ids()转变为character_ids,然后从character_ids构造elmo,得到输出。



- Inference

    [ELMo代码详解(一)：数据准备](https://blog.csdn.net/jeryjeryjery/article/details/80839291)

    [ELMo代码详解(二):模型代码](https://blog.csdn.net/jeryjeryjery/article/details/81183433)

    [流水账︱Elmo词向量中文训练过程杂记](https://blog.csdn.net/sinat_26917383/article/details/81913790)

    [在中文语料中，ELMo比word2vec好很多么？](https://www.zhihu.com/question/288565744)

    [Deep Contextualized Word Representations with ELMo](https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/)

    [NAACL2018 一种新的embedding方法Deep contextualized word representations ELMo原理与用法](https://zhuanlan.zhihu.com/p/37915351)

    [Document-QA](https://github.com/allenai/document-qa)
<meta http-equiv="refresh" content="0.1">