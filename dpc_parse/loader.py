# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

import torch
from torch.utils.data import TensorDataset, DataLoader

import mylib.word_embed as lw
import mylib.utils as lu
from dpc_parse.args import arguments, logger
from dpc_parse.corpus import CorpusPropValIter
from dpc_parse.api import DpcKinds


class DependencyEmbeddingAgent:

    def __init__(self, word_embed_size, max_word_count, max_char_count, bert_vocab_file, word_vector_file=None):
        self.SOS = True
        self.max_word_count = max_word_count
        self.max_char_count = max_char_count
        self.word_embed_tool = lw.WordEmbeddingTool(word_embed_size, logger)
        if word_vector_file:
            self.word_embed_tool.load_w2v_model(word_vector_file)
        self.bert_embed_tool = lw.BertEmbeddingTool(bert_vocab_file, 'char', max_word_chars=200)

    def load(self, model_path):
        self.word_embed_tool.load_vocab_table(model_path)

    def save(self, model_path):
        self.word_embed_tool.save_vocab_table(model_path)

    @property
    def vocab_built(self):
        return self.word_embed_tool.vocab_built

    def stat_vocab(self, corpus, min_word_freq=0):
        self.word_embed_tool.logger = logger
        self.word_embed_tool.stat_vocab(min_word_freq, 0, CorpusPropValIter(corpus, "FORM"))

    def embed(self, corpus, show_first_sample=True):
        corpus.filter(self.max_word_count, self.max_char_count)
        word_ids_list = self.word_embed_tool.embed_corpus(CorpusPropValIter(corpus, "FORM"),
                                                          self.max_word_count, sos=self.SOS)
        if self.word_embed_tool.max_sequence_length > self.max_word_count > 0:
            logger.warning("最大词数：{}，实际：{}".format(self.max_word_count, self.word_embed_tool.max_sequence_length))
        else:
            if self.max_word_count == 0:
                self.max_word_count = self.word_embed_tool.max_sequence_length
            logger.info("最大词数：{}，实际：{}".format(self.max_word_count, self.word_embed_tool.max_sequence_length))
        word_lens_list = self.get_word_lens(CorpusPropValIter(corpus, "FORM"), self.max_word_count)
        char_ids_list, trunc_count = self.bert_embed_tool.embed(CorpusPropValIter(corpus, "FORM"),
                                                                self.max_char_count, sos=self.SOS)
        if trunc_count > 0:
            logger.warning("{}句最大字符数超长：{}".format(trunc_count, self.max_char_count))
        if show_first_sample:
            logger.info("word_ids_list: " + str(word_ids_list[0]))
            logger.info("word_lens_list: " + str(word_lens_list[0]))
            logger.info("char_ids_list: " + str(char_ids_list[0]))

        return word_ids_list, char_ids_list, word_lens_list

    def get_word_lens(self, text_words_iter, sequence_length, pad=True):
        word_lens_list = []
        for text_words in text_words_iter:
            word_lens = []
            if self.SOS:
                word_lens.append(1)
            for word in text_words:
                word_lens.append(len(word))
            word_lens_list.append(word_lens)
        if pad:
            for word_lens in word_lens_list:
                pad_count = sequence_length - len(word_lens)
                for _ in range(pad_count):
                    word_lens.append(0)
        return word_lens_list

    def embed_fast(self, tagged_sentence_list):
        word_ids_list = self.word_embed_tool.embed_corpus(TaggedSentenceIter(tagged_sentence_list),
                                                          self.max_word_count, sos=self.SOS)
        word_lens_list = self.get_word_lens(TaggedSentenceIter(tagged_sentence_list), self.max_word_count)
        char_ids_list, trunc_count = self.bert_embed_tool.embed(TaggedSentenceIter(tagged_sentence_list),
                                                                self.max_char_count, sos=self.SOS)
        char_counts1 = [sum(len(w) for w, t in wts) for wts in tagged_sentence_list]
        char_counts2 = [sum((1 if i != 0 else 0) for i in ids) for ids in char_ids_list]
        for i, c in enumerate(char_counts1):
            if char_counts2[i] != c + 1:
                print("\t".join(''.join(w for w, t in tagged_sentence_list[i])))
                print("\t".join(str(i) for i in char_ids_list[i]))
        return word_ids_list, char_ids_list, word_lens_list

    def embed_dependency(self, corpus, pad=True, show_first_sample=True):
        if self.max_word_count == 0:
            self.max_word_count = self.word_embed_tool.max_sequence_length

        dpc_arcs_list, dpc_kinds_list = [], []
        for sentence in corpus:
            dpc_arcs, dpc_kinds = [], []
            if self.SOS:
                dpc_arcs.append(0)
                dpc_kinds.append(DpcKinds.UNK)
            for conll in sentence:
                dpc_arcs.append(conll.HEAD)
                dpc_kinds.append(DpcKinds.parse(conll.DEPREL))
                if dpc_kinds[-1] == DpcKinds.ERR:
                    logger.error(conll)
            if pad:
                pad_count = self.max_word_count - len(dpc_arcs)
                for _ in range(pad_count):
                    dpc_arcs.append(0)
                    dpc_kinds.append(DpcKinds.UNK)
            dpc_arcs_list.append(dpc_arcs)
            dpc_kinds_list.append(dpc_kinds)
        if show_first_sample:
            logger.info("dpc_arcs_list: " + str(dpc_arcs_list[0]))
            logger.info("dpc_kinds_list: " + str(dpc_kinds_list[0]))
        return dpc_arcs_list, dpc_kinds_list


class TaggedWordsIter:

    def __init__(self, word_tag_list):
        self.word_tag_list = word_tag_list

    def __iter__(self):
        for word, tag in self.word_tag_list:
            yield word


class TaggedSentenceIter:

    def __init__(self, tagged_sentence_list):
        self.tagged_sentence_list = tagged_sentence_list

    def __iter__(self):
        for tagged_sentence in self.tagged_sentence_list:
            yield TaggedWordsIter(tagged_sentence)


def create_embed_agent(args, load_vector, existing_path=None):
    if not existing_path:
        bert_vocab_path = args['bert']['vocab_file']
    else:
        bert_vocab_path = lu.path_join(existing_path, 'vocab.txt')

    embed_tool = DependencyEmbeddingAgent(
        args['model']['word_embed_size'],
        args['corpus']['max_word_count'],
        args['corpus']['max_char_count'],
        bert_vocab_path,
        args['corpus'].get('vector_model', None) if load_vector else None
    )
    if existing_path:
        embed_tool.load(existing_path)
    return embed_tool


def embed_pack_corpus(corpus, embed_tool, is_train, is_eval):
    batch_size = arguments['train' if is_train else 'predict']['batch_size']
    word_ids_list, char_ids_list, word_lens_list = embed_tool.embed(corpus)
    logger.info("{}语料包含{}条".format('Train' if is_train else ('Eval' if is_eval else ''), len(word_ids_list)))

    if is_train or is_eval:
        dpc_arcs_list, dpc_kinds_list = embed_tool.embed_dependency(corpus)
        # 创建数据集
        dataset = TensorDataset(torch.tensor(word_ids_list, dtype=torch.long),
                                torch.tensor(char_ids_list, dtype=torch.long),
                                torch.tensor(word_lens_list, dtype=torch.long),
                                torch.tensor(dpc_arcs_list, dtype=torch.long),
                                torch.tensor(dpc_kinds_list, dtype=torch.long))
        # 分布式训练 if local_rank >= 0 DistributedSampler
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    else:
        # 创建数据集
        dataset = TensorDataset(torch.tensor(word_ids_list, dtype=torch.long),
                                torch.tensor(char_ids_list, dtype=torch.long),
                                torch.tensor(word_lens_list, dtype=torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
