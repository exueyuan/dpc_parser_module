# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

from typing import *
from collections import namedtuple

import mylib.utils as lu
import mylib.txt_lib as lt
from dpc_parse.args import arguments, logger
from dpc_parse.api import DpcKinds


ConllFields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL']  # , 'PHEAD', 'PDEPREL'
ConllFieldCount = len(ConllFields)
ConllField2Idx = {fn: idx for idx, fn in enumerate(ConllFields)}
CoNLL = namedtuple(typename='CoNLL',
                   field_names=ConllFields)


class Sentence(List[CoNLL]):

    @classmethod
    def parse_conll(cls, line):
        pv_list = line.split('\t')
        if len(pv_list) < ConllFieldCount:
            return None
        pv_list = [pv.strip() if pv != "_" else None for pv in pv_list]
        pv_list = pv_list[:ConllFieldCount]
        pv_list[0] = int(pv_list[0])
        pv_list[6] = int(pv_list[6])
        return CoNLL(*pv_list)

    @classmethod
    def format_conll(cls, conll):
        pv_list = [str(pv) if pv is not None else "_" for pv in conll]
        return '\t'.join(pv_list)

    def to_tagged_text(self, concat_by=' '):
        items = []
        oi, wi, ti, ai, di = 0, 1, 4, 6, 7
        for conll in self:
            word = conll[wi]
            word = word.replace(' ', '')
            word = word.replace(lt.ColumnSeparator, '|')
            dpc_desc = DpcKinds.get(conll[di]).Title
            item = "{}│{}〖{}-{}-{}〗".format(word, conll[ti], conll[oi], dpc_desc, conll[ai])
            items.append(item)
        return concat_by.join(items) + '〗'  # 表示重新开始编号


class Corpus(List[Sentence]):

    def __repr__(self):
        return f"{len(self)} CoNLL items"

    def filter(self, max_word_count, max_char_count):
        check_len = max_char_count // 4
        old_len = len(self)
        i = old_len - 1
        while i >= 0:
            sentence = self[i]
            if len(sentence) >= max_word_count:
                self.pop(i)
            elif len(sentence) >= check_len:
                char_len = sum(len(c[1]) for c in sentence)
                if char_len >= max_char_count:
                    self.pop(i)
            i -= 1
        logger.warning(f"已过滤超长语料{old_len - len(self)}句，保留{len(self)}句")

    @classmethod
    def load_from(cls, conll_txt_file):
        logger.info("LOAD FROM {} ...".format(conll_txt_file))

        result = Corpus()
        sentence = Sentence()
        for line in lt.load_lines(conll_txt_file):
            if not line:
                if len(sentence) > 0:
                    result.append(sentence)
                    sentence = Sentence()
            else:
                conll = Sentence.parse_conll(line)
                if conll is not None:
                    sentence.append(conll)
                else:
                    logger.error("CoNLL ERROR: " + line)
        if len(sentence) > 0:
            result.append(sentence)
        return result

    def save_to(self, conll_txt_file):
        logger.info("SAVE TO {} ...".format(conll_txt_file))

        with lt.open_file(conll_txt_file, 'w') as writer:
            for sentence in self:
                for conll in sentence:
                    line = Sentence.format_conll(conll)
                    writer.write_line(line)
                writer.write_line('')

    def replace_words_tag(self):
        import fool
        wi = ConllField2Idx['FORM']
        batch_size = 100
        start, count = 0, len(self)
        while start < count:
            batch = self[start: start+batch_size]
            word_list_list = []
            for sentence in batch:
                word_list = list(conll[wi] for conll in sentence)
                word_list_list.append(word_list)
            tag_list_list = fool.LEXICAL_ANALYSER.pos(word_list_list)
            for idx, (sentence, tag_list) in enumerate(zip(batch, tag_list_list)):
                self[start+idx] = Sentence(self.replace_tag(conll, tag) for conll, tag in zip(sentence, tag_list))
            start += batch_size
            print(f"已处理{start}")

    @classmethod
    def replace_tag(cls, conll, tag):
        ti = ConllField2Idx['POS']
        pv_list = list(conll)
        pv_list[ti] = tag
        pv_list[ti - 1] = tag[:1]
        return CoNLL(*pv_list)

    @classmethod
    def parse_tagged_text(cls, tagged_words_text):
        sentence_list = []
        sentence = Sentence()
        order = 0
        for item in tagged_words_text.split(' '):
            order += 1
            word, tags = item.split(lt.ColumnSeparator)
            if len(word) == 0:
                word = " "
            tag = tags[:tags.index('〖')]
            dependency = tags[tags.index('〖') + 1: -1] 
            idx, dpc, arc = dependency.split('-')
            assert int(idx) == order
            if arc.endswith('〗'):
                arc = arc[:-1]
                sentence_end = True
            else:
                sentence_end = False
            conll = CoNLL(order, word, None, tag[0], tag, None, int(arc), dpc)
            sentence.append(conll)
            if sentence_end:
                sentence_list.append(sentence)
                order = 0
                sentence = Sentence()

        return sentence_list


def load_corpus(corpus_type):
    if corpus_type == "train":
        data_file = arguments['corpus']['train_file']
    elif corpus_type in lu.TrainOrEvalCorpus:
        data_file = arguments['corpus']['eval_file']
    elif corpus_type == "test":
        data_file = arguments['corpus']['test_file']
    elif corpus_type == "predict":
        data_file = arguments['predict']['data_file']
    else:
        raise ValueError("Invalid mode %s" % corpus_type)

    return Corpus.load_from(data_file)


class SentencePropValIter:

    def __init__(self):
        self.pv_idx = 0
        self.sentence = None
        self.iterating = False

    def reset(self, pv_idx, sentence):
        assert not self.iterating, "SentencePropValIter使用中，不能重置"
        self.pv_idx = pv_idx
        self.sentence = sentence

    def __iter__(self):
        self.iterating = True
        for i in range(len(self.sentence)):
            yield self.sentence[i][self.pv_idx]
        self.iterating = False


class CorpusPropValIter:

    def __init__(self, sentence_list, prop_name):
        prop_name = prop_name.upper()
        if prop_name == "WORD":
            self.pv_idx = 1
        else:
            self.pv_idx = ConllField2Idx[prop_name]
        self.sentence_list = sentence_list
        self.iter_cache = SentencePropValIter()

    def __iter__(self):
        for si in range(len(self.sentence_list)):
            self.iter_cache.reset(self.pv_idx, self.sentence_list[si])
            yield self.iter_cache


def convert_corpus_tags():
    for name in ['eval_file', 'train_file', 'test_file']:
        data_file = arguments['corpus'][name]
        corpus = Corpus.load_from(data_file)
        corpus.replace_words_tag()
        corpus.save_to(data_file.replace('.conllx', '.txt'))


if __name__ == '__main__':
    convert_corpus_tags()
