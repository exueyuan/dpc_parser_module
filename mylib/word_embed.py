# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

"""
嵌入及分词工具
"""

import re
import os
import sys
import math
import numpy as np
import collections
import jieba
import jieba.posseg as psg
import mylib.txt_lib as lt
from mylib import utils as lu


MaxCh16 = chr(65535)
CHAR_TRANSLATE = [i for i in range(ord(MaxCh16) + 1)]
STOP_WORDS = {'', ' ', '.', ',', ';', '!', '、', '，', '；', '。', '！', '？', '│',
              '[', ']', '（', '）', '“', '”', '《', '》', '：', '-', '·', '.', '(', ')'
              '我', '我们', '你', '你们', '他', '他们', '她', '她们', '它', '它们',
              '这', '那', '其', '其他', '其它',
              '的', '地', '得', '着', '了', '过', '等', '吗', '吧', '之',
              '是', '为', '有', '给', '来', '去', '到', '上', '中', '下', '前', '后',
              '也', '还', '都', '将', '要', '请', '指',
              '在', '对', '于', '让', '以', '向', '由', '被', '所', '从', '即', '可',
              '和', '与', '或', '并', '及', '就', '而', '又', '因', '但',
              '个', '人',  # '年', '月', '日', '时', '分', '秒', '最', '大', '小',
              '至今', '现在',
              }


def load_char_translate(filename):
    global CHAR_TRANSLATE
    for trans_pair in lt.load_lines(filename, False, True, False):
        if len(trans_pair) == 3:
            CHAR_TRANSLATE[ord(trans_pair[0])] = ord(trans_pair[2])
    return CHAR_TRANSLATE


def translate(text):
    valid_chars = [chr(CHAR_TRANSLATE[ord(ch)]) if ch <= MaxCh16 else ch for ch in text]
    return ''.join(valid_chars)


def load_stopwords(filename):
    global STOP_WORDS
    for word in lt.load_lines(filename, False, True, False):
        STOP_WORDS.add(word)
        if word != word.lower():
            STOP_WORDS.add(word.lower())
        if word != word.upper():
            STOP_WORDS.add(word.upper())
    return STOP_WORDS


def clean_stopwords(text):
    valid_chars = [ch for ch in text if ch not in STOP_WORDS]
    return ''.join(valid_chars)


CLS = "[CLS]"
SEP = "[SEP]"
PAD = "[PAD]"
UNK = "[UNK]"
SOS = "[SOS]"
EOS = "[EOS]"
_basic_tokenizer = None


def bert_clean(text, remove_space_between_chinese=False):
    """参考pytorch_pretrained_bert.tokenization.BasicTokenizer._clean"""
    def is_redundant_space(ch):
        return lt.is_punctuation(ch) or (lt.is_cjk_char(ch) and remove_space_between_chinese)
    output = []
    prev_need_space, has_space = False, False
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or lt.is_control(char):
            continue
        if lt.is_whitespace(char):
            has_space = True
        else:
            need_space = not is_redundant_space(char)
            if prev_need_space and need_space and has_space:
                output.append(" ")
            has_space = False
            output.append(char)
            prev_need_space = need_space

    return "".join(output)


def bert_trans(text):
    text = text.lower()
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('…', '.')
    text = text.replace('―', '-')
    text = bert_clean(text)
    return text


def basic_tokenize(text, trans=True):
    global _basic_tokenizer
    # from pytorch_pretrained_bert.tokenization import BasicTokenizer
    from transformers import BasicTokenizer
    if _basic_tokenizer is None:
        _basic_tokenizer = BasicTokenizer(do_lower_case=False)

    if trans:
        text = bert_trans(text)
    return _basic_tokenizer.tokenize(text)


def jieba_tokenize(text, trans=True):
    if trans:
        text = bert_trans(text)
    else:
        text = bert_clean(text)
    words = jieba_cut(text)
    return [w for w in words if w != ' ']


def calc_char_len(token_list, from_idx=0, to_idx=-1):
    if to_idx < 0:
        to_idx = len(token_list)
    result = 0
    for i in range(from_idx, to_idx):
        result += len(token_list[i])
    return result


def get_tokens_text(embedded_tokens, start=0, stop=-1):
    tokens = []
    for segment in embedded_tokens[start:stop]:
        if segment.startswith('##'):
            segment = segment[2:]
        if segment == UNK:
            segment = "◆"
        tokens.append(segment)
    return bert_clean(' '.join(tokens), True)


class BertEmbeddingTool:
    """BERT嵌入工具类，sentence > segment > token """

    def __init__(self, bert_vocab_file, seg_method="default", unk_token=UNK,
                 max_word_chars=40, force_lower=True):
        self.unk_tokens = tuple([unk_token])
        self.tokens_list = lt.load_lines(bert_vocab_file, False, False)
        self.tokens2id = {token: idx for idx, token in enumerate(self.tokens_list)}

        self.parts_cache = {}
        self.force_lower = force_lower
        self.max_word_chars = max_word_chars
        seg_method = seg_method.lower()
        if seg_method == "default":
            self.split_mode = 0
        elif seg_method == "char":
            self.split_mode = 1
            self.parts_cache[" "] = ("[unused1]",)
        elif seg_method == "jieba":
            self.split_mode = 2
        else:
            assert not seg_method, "支持的分词方式"

    def segment(self, text, trans=True):
        if self.split_mode == 2:
            return jieba_tokenize(text, trans)
        if self.split_mode == 0:
            return basic_tokenize(text, trans)
        return list(text)

    def get_tokens(self, segment):
        result = self.parts_cache.get(segment, None)
        if result is None:
            result = self._split_tokens(segment)
            self.parts_cache[segment] = result
        return result

    def _split_tokens(self, segment):
        chars = list(segment)
        if self.force_lower:
            chars = [ch.lower() for ch in chars]
        if len(chars) > self.max_word_chars:
            return self.unk_tokens

        start, count = 0, len(chars)
        piece_list = []
        if self.split_mode == 1:
            while start < count:
                token = chars[start]
                if token in self.tokens2id:
                    piece_list.append(token)
                elif "##" + token in self.tokens2id:
                    piece_list.append("##" + token)
                else:
                    piece_list.append(UNK)
                start += 1
        elif self.split_mode == 2:
            while start < count:
                token = chars[start]
                if start > 0 and "##" + token in self.tokens2id:
                    piece_list.append("##" + token)
                elif token in self.tokens2id:
                    piece_list.append(token)
                else:
                    piece_list.append(UNK)
                start += 1
        else:
            # FROM Bert WordpieceTokenizer.tokenize
            while start < count:
                end = count
                word_piece = None
                while start < end:
                    token = "".join(chars[start:end])
                    if start > 0:
                        token = "##" + token
                    if token in self.tokens2id:
                        word_piece = token
                        break
                    end -= 1
                if word_piece is None:
                    word_piece = UNK
                piece_list.append(word_piece)
                start = end
        return tuple(piece_list)

    def to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.tokens2id[token])
        return ids

    def ids_to_text(self, token_ids, join_str=' '):
        tokens = []
        for tid in token_ids:
            segment = self.tokens_list[tid]
            if segment.startswith('##'):
                segment = segment[2:]
            elif segment == UNK:
                segment = "◆"
            elif segment == CLS or segment == SEP:  # 忽略 0 0
                segment = ""
            elif segment == PAD:
                segment = "◇"  # ◈
            tokens.append(segment)
        return join_str.join(tokens)

    def tokenize(self, article_segments, trace_start=-2, trace_end=-2, seg_separator=None):
        if isinstance(article_segments, str):
            if seg_separator is not None:
                article_segments = article_segments.split(seg_separator)
            else:
                article_segments = self.segment(article_segments)
        return self.tokenize_(article_segments, trace_start, trace_end)

    def tokenize_(self, segments, trace_start=-2, trace_end=-2):
        trace_start_piece = -1
        trace_end_piece = -1
        result_pieces = []
        for idx, segment in enumerate(segments):
            if idx == trace_start:
                trace_start_piece = len(result_pieces)
            result_pieces.extend(self.get_tokens(segment))
            if idx == trace_end:
                trace_end_piece = len(result_pieces) - 1
        if trace_start < -1 and trace_end < -1:
            return result_pieces
        return result_pieces, trace_start_piece, trace_end_piece

    def embed(self, article_list, max_sequence_length, seg_separator=None,
              pad=True, sos=False, eos=False):
        trunc_count = 0
        max_token_count = max_sequence_length - (1 if sos else 0) - (1 if eos else 0)
        article_ids_list = []
        for article in article_list:
            if isinstance(article, str):
                if seg_separator is not None:
                    segments = article.split(seg_separator)
                else:
                    segments = self.segment(article)
            else:
                segments = article
            token_list = self.tokenize(segments)
            if len(token_list) > max_token_count:
                trunc_count += 1
                token_list = token_list[:max_token_count]
            if sos:
                token_list = [CLS] + token_list  # Bert词汇表中没有SOS，用CLS代替
            if eos:
                token_list = token_list + [SEP]  # Bert词汇表中没有EOS，用SEP代替

            token_ids = self.to_ids(token_list)
            if pad:
                while len(token_ids) < max_sequence_length:
                    token_ids.append(0)
            article_ids_list.append(token_ids)
        return article_ids_list, trunc_count


re_number_or_e = re.compile(r"[0-9]+([eE]?)([0-9]?)")
number_units = [
    '十', '拾', '百', '佰', '千', '仟', '万', '十万', '拾万', '百万', '佰万', '千万', '仟万',
    '亿', '十亿', '拾亿', '百亿', '佰亿', '千亿', '仟亿', '万亿', '兆',
]
measure_units = [
    '元', '美元', '欧元', '日元', '澳元', '英镑', '法郎', '港币',
    '年', '月', '天', '小时', '分钟', '秒', '公斤',  '克',  '千克', '斤',
    '尺', '米', '千米', '公里', '里', '英里', '海里', '平方尺', '平方米', '平方千米', '平方公里', '立方米', '方',
    '股', '手', '人', '人次', '片', '吨', '个', '台', '只', '条', '号', '张', '倍', '局', '批', '套', '件', '封', '头',
    '户', '家', '辆', '份', '盎司', '支', '宗', '桶',  '块', '亩',  '篇', '页', '盒', '瓦', '千瓦', '度', '艘', '颗'
]
combined_units = {}
for nu in number_units:
    combined_units[nu] = 0
    for mu in measure_units:
        combined_units[nu + mu] = len(mu)
combined_units.pop('百度')


def jieba_cut(text, cn_units_to_num=False):
    tokenizer = jieba.cut(text, False, HMM=False)
    word_list = []

    number = None
    number_start = sys.maxsize
    need_int = False
    concat_dot = False
    concat_exp = 0
    concat_comma = -1

    for word in tokenizer:
        if cn_units_to_num and number is not None and not need_int:
            ul = combined_units.get(word, -1)
            if ul == 0:
                number += word
                word_list[-1] = number
                continue
            if ul > 0 and word not in measure_units:
                word_list[-1] = number + word[:-ul]
                word = word[-ul:]
        word_list.append(word)
        if concat_dot and word == '.':
            number += word
            concat_dot = False
            concat_comma = -1
            need_int = True
            continue
        elif concat_comma >= 0 and word == ',':
            number += word
            concat_comma += 1
            need_int = True
            continue
        elif concat_exp > 0 and (word == '+' or word == '-'):
            number += word
            concat_exp = 2
            need_int = True
            continue
        nsr = re_number_or_e.search(word)
        has_e = nsr is not None and len(nsr.group(1)) > 0
        has_exp = nsr is not None and len(nsr.group(2)) > 0
        if need_int and nsr is not None and (not has_e or concat_comma <= 0 and concat_exp < 2) and \
                (concat_comma <= 0 or len(word) == 3):
            if number == '+' or number == "-":
                concat_dot = not has_e
                concat_exp = 1 if has_e and not has_exp else 0
                concat_comma = 0 if (not has_e) and len(word) <= 3 else -1
            number += word
            need_int = False
            while len(word_list) > number_start:
                word_list.pop()
            word_list[number_start - 1] = number
            if has_e and not has_exp:
                concat_exp = 1
            continue
        if nsr is not None:
            number = word
            number_start = len(word_list)
            need_int = False
            concat_dot = not has_e
            concat_exp = 1 if has_e and not has_exp else 0
            concat_comma = 0 if (not has_e) and len(word) <= 3 else -1
        elif word == '+' or word == '-':
            number = word
            number_start = len(word_list)
            need_int = True
            concat_dot = False
            concat_exp = 0
            concat_comma = -1
        else:
            if (word == '%' or word == '‰') and number is not None and \
                    not need_int and concat_exp == 0 and concat_comma <= 0:
                number += word
                word_list.pop()
                word_list[-1] = number
            number = None
            number_start = sys.maxsize
            need_int = False
            concat_dot = False
            concat_exp = False
            concat_comma = -1

    return word_list


def split_words(text, seg_method='jieba', words_concat_by=None):
    if seg_method == 'jieba':
        words = jieba_cut(text)
    elif seg_method == 'char':
        words = [ch for ch in text]
    elif seg_method == 'space':
        words = text.split()
    elif seg_method == 'tab':
        words = text.split('\t')
    else:
        raise RuntimeError('分词方法【{}】不支持'.format(seg_method))

    if words_concat_by:
        words = words_concat_by.join(words)
    return words


def jieba_pos_tag(text, ner=False):
    tagged_words = psg.lcut(text, HMM=ner)
    for i, (word, tag) in enumerate(tagged_words):
        if tag is None or len(tag) == 0 or tag == "eng":
            if lt.is_decimal_str(word):
                tag = 'm'
            elif lt.contains_punc_or_symbol(word):
                tag = 'w'
            else:
                tag = 'x'
        tagged_words[i] = (word, tag)
    return tagged_words


def filter_word(word, stopwords=True, punc_symbols=False, whole_numbers=False,
                any_numbers=False, any_letters=False, only_chinese=False):
    if word is None or len(word) == 0:
        return True
    if stopwords and word in STOP_WORDS:
        return True
    has_chinese = False
    has_number = False
    has_letter = False
    has_punc_symbol = False
    for ch in word:
        if '0' <= ch <= '9':
            has_number = True
        elif ch == '.' or ch == '+' or ch == '-':
            has_punc_symbol = True  # 不是has_number
        elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch == '_':
            has_letter = True
        elif lt.is_chinese(ch):
            has_chinese = True
        else:
            has_punc_symbol = True
    if punc_symbols and has_punc_symbol:
        return True
    if any_numbers and has_number:
        return True
    if any_letters and has_letter:
        return True
    if only_chinese and (has_punc_symbol or has_letter or has_number):
        return True
    if whole_numbers and has_number and not (has_letter or has_chinese):  # has_punc_symbol
        return True
    return False


def filter_words(words_or_items, stopwords=True, punc_symbols=False, whole_numbers=False,
                 any_numbers=False, any_letters=False, only_chinese=False):
    filtered = []
    for word_item in words_or_items:
        word = word_item if isinstance(word_item, str) else word_item[0]
        if not filter_word(word, stopwords, punc_symbols, whole_numbers,
                           any_numbers, any_letters, only_chinese):
            filtered.append(word_item)
    return filtered


def cosine(a, b):
    return np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calc_sentence_cosine(w2v_model, sentence1, sentence2, seg_method='jieba'):
    sv1 = [w2v_model[word] for word in split_words(sentence1, seg_method) if word in w2v_model.vocab]
    sv2 = [w2v_model[word] for word in split_words(sentence2, seg_method) if word in w2v_model.vocab]
    if len(sv1) > 0:
        sv1 = np.mean(sv1, axis=0)
    else:
        return 0.0  # sv1 = np.zeros([w2v_model.vector_size])
    if len(sv2) > 0:
        sv2 = np.mean(sv2, axis=0)
    else:
        return 0.0  # sv2 = np.zeros([w2v_model.vector_size])
    return cosine(sv1, sv2)


def articles_to_sentences(article_list, sentence_end_chars=lt.SentenceEndChars, strip=True, align_length=0):
    sentence_list = []
    sentence_article_idx_list = []
    if not isinstance(sentence_end_chars, set):
        sentence_end_chars = set(sentence_end_chars)
    for article_idx, article in enumerate(article_list):
        if align_length > 0:
            max_sen_len = align_length
        else:
            max_sen_len = 2048
        sentences = lt.split_sentences(article, sentence_end_chars, max_sen_len)
        last_len, last_sentence = 0, None
        for sentence in sentences:
            if strip:
                sentence = sentence.strip()
            sl = len(sentence)
            if sl <= 0:
                continue
            if last_len == 0:
                last_len, last_sentence = sl, sentence
            elif sl + last_len > align_length:
                sentence_list.append(last_sentence)
                sentence_article_idx_list.append(article_idx)
                last_len, last_sentence = sl, sentence
            else:
                last_len += sl
                last_sentence += sentence
        if last_len > 0:
            sentence_list.append(last_sentence)
            sentence_article_idx_list.append(article_idx)

    return sentence_list, sentence_article_idx_list


def cut_words(word_tag_list, max_word_count, max_char_count, broken_words="；，：;、, :·"):
    parts = []
    si, ei = 0, 0
    ccc, ecc = 0, 0
    for wi, (word, tag) in enumerate(word_tag_list):
        ccc += len(word)
        if word in broken_words and wi - si < max_word_count and ccc <= max_char_count:
            ei = wi + 1
            ecc = ccc

        if wi - si >= max_word_count or ccc > max_char_count:
            if ei <= 0:
                ei = wi
                ecc = ccc - len(word)
            parts.append(word_tag_list[si:ei])
            si = ei
            ccc -= ecc
            ei, ecc = 0, 0
    if si < len(word_tag_list):
        parts.append(word_tag_list[si:])
    return parts


def split_sentences(tagged_words, max_word_count, max_char_count, broken_words="。。！？ !?"):
    sentence_list = []
    in_quote = False
    in_brace, brace_pos = 0, 0
    idx_last = len(tagged_words) - 1
    start, char_count = 0, 0
    for wi, (word, tag) in enumerate(tagged_words):
        if word == '“' or word == '"':
            in_quote = not in_quote  # 不用True，存在中文引号不匹配的的情形，“沪伦通“
        elif word == "”":
            in_quote = False
        elif word in '([{（［【':
            in_brace += 1
            brace_pos = wi  # 避免括号丢一半
        elif word in ')]}）］】' and in_brace > 0:
            in_brace -= 1
        char_count += len(word)
        if wi == idx_last or (word in broken_words or word[-1] == '。') and \
                not in_quote and (brace_pos == 0 or wi - brace_pos > 15):
            if word == ' ' and 0 < wi < idx_last:
                if not lt.is_chinese(tagged_words[wi - 1][0][-1]) or not lt.is_chinese(tagged_words[wi + 1][0][0]):
                    continue
            sentence = tagged_words[start:wi+1]
            if wi - start < max_word_count and char_count <= max_char_count:
                sentence_list.append(sentence)
            else:
                sentence_list.extend(cut_words(sentence, max_word_count, max_char_count))
            start, char_count = wi + 1, 0
    return sentence_list


class WordVectorTool:

    def __init__(self, w2v_model):
        from gensim.models import KeyedVectors
        if isinstance(w2v_model, str):
            self.w2v_model = KeyedVectors.load_word2vec_format(w2v_model, datatype=np.float32, binary=True)
        else:
            self.w2v_model = w2v_model
        self.vocab = self.w2v_model.vocab
        self.embedding_size = self.w2v_model.vector_size

    def get_word_vec(self, word, zero_oov=False, random_oov=False):
        if self.w2v_model and word in self.w2v_model.vocab:
            return self.w2v_model[word]
        if zero_oov:
            return np.zeros(self.embedding_size, np.float32)
        if random_oov:
            return np.random.uniform(-1.0, 1.0, self.embedding_size).astype(np.float32)
        return None

    def calc_similarity(self, word1, word2):
        vec1 = self.get_word_vec(word1)
        vec2 = self.get_word_vec(word2)
        if vec1 is None or vec2 is None:
            return 0.0
        return cosine(vec1, vec2)

    def get_phrase_vec(self, phrase, seg_method='jieba'):
        if phrase in self.w2v_model.vocab:
            return self.w2v_model[phrase]
        return self.get_words_vec(split_words(phrase, seg_method))

    def get_words_vec(self, word_list):
        vec_list = []
        for word in word_list:
            if word in self.w2v_model.vocab:
                vec = self.w2v_model[word]
            else:
                cvs = [self.w2v_model[ch] for ch in word if ch in self.w2v_model.vocab]  # 忽略稀有字符
                if len(cvs) <= 0:
                    continue
                vec = np.array(np.mean(cvs, axis=0))
            vec_list.append(vec)
        return np.array(np.mean(vec_list, axis=0)) if len(vec_list) > 0 else None

    def calc_phrase_similarity(self, phrase1, phrase2, seg_method='jieba'):
        vec1 = self.get_phrase_vec(phrase1, seg_method)
        vec2 = self.get_phrase_vec(phrase2, seg_method)
        return cosine(vec1, vec2) if vec1 is not None and vec2 is not None else 0.0

    def find_similar(self, phrase, top_count=5, seg_method='jieba'):
        if phrase in self.w2v_model.vocab:
            return self.w2v_model.most_similar(phrase, topn=top_count)
        words = []
        for word in split_words(phrase, seg_method):
            if word in self.w2v_model.vocab:
                words.append(word)
            else:
                for ch in word:
                    if ch in self.w2v_model.vocab:
                        words.append(ch)
        return self.w2v_model.most_similar(positive=words, topn=top_count)


class WordEmbeddingTool:

    def __init__(self, embedding_size=100, oov_as_unk=True, logger=lu.NullLogger):
        self.embedding_size = max(10, embedding_size)
        self.oov_as_unk = oov_as_unk
        self.vocabulary = []
        self.words2index = collections.Counter()
        self.number_skip_freq = 2  # todo 设置单独的词
        self.symbol_skip_freq = 2
        self._embedding_table = []
        self.logger = logger
        self.sequence_length = 0
        self.max_sequence_length = 0
        self.vec_tool = None
        self.embedding_mode = 'random'

    def load_w2v_model(self, word_vector_file):
        if word_vector_file:
            self.vec_tool: WordVectorTool = WordVectorTool(word_vector_file)
            self.embedding_size = self.vec_tool.embedding_size
            _, self.embedding_mode = os.path.split(word_vector_file)
        else:
            self.vec_tool = None
            self.embedding_mode = 'random'

    @property
    def vocab_built(self):
        return len(self.words2index) > 4

    def _reset_vocab(self):
        self.oov_count = 0
        self.vocabulary.clear()
        self.words2index.clear()
        self._embedding_table.clear()
        self.append_word(PAD, np.zeros(self.embedding_size, dtype=np.float32))
        self.append_word(UNK, np.random.uniform(-0.3, 0.3, self.embedding_size).astype(np.float32))
        self.append_word(SOS, np.random.uniform(-0.1, 0.1, self.embedding_size).astype(np.float32))
        self.append_word(EOS, np.random.uniform(-0.1, 0.1, self.embedding_size).astype(np.float32))
        self._loc_reserved_words()

    def _loc_reserved_words(self):
        self.pad = self.words2index[PAD]
        self.unk = self.words2index[UNK]
        self.sos = self.words2index[SOS]
        self.eos = self.words2index[EOS]
        assert self.pad == 0
        self.oov_count = 0
        self.random_count = 0

    def _get_vec(self, word):
        if self.vec_tool and word in self.vec_tool.vocab:
            return self.vec_tool.get_word_vec(word)
        elif self.vec_tool:
            self.random_count += 1
            self.logger.debug('"%s"不在词向量库中，用合成向量代替' % word)
            vector = self.vec_tool.get_phrase_vec(word)
            if vector is not None:
                return vector
        return np.random.uniform(-1.0, 1.0, self.embedding_size).astype(np.float32)

    _NoUseVec = (0.0, 0.0)

    def append_word(self, word, vector):
        if word in self.words2index:
            raise RuntimeError('"{}"重复'.format(word))
        word_id = len(self.vocabulary)
        self.vocabulary.append(word)
        self.words2index[word] = word_id
        # if vector != self._NoUseVec and len(vector) != self.embedding_size:
        #     raise RuntimeError('"{}"向量无效{}'.format(word, str(vector)))
        self._embedding_table.append(vector)

    def stat_vocab(self, min_frequency=0, max_vocab_size=0, *text_words_iter_list):
        self._reset_vocab()

        words_counter = collections.Counter()
        for text_words_iter in text_words_iter_list:
            for text_words in text_words_iter:
                words_counter.update(text_words)

        word_count_list = sorted(words_counter.items(), key=lambda pair: pair[1], reverse=True)
        # 按词频降序生成词汇表
        for index, (word, count) in enumerate(word_count_list):
            if count == min_frequency and self.vec_tool is not None:
                if word not in self.vec_tool.vocab:
                    self.oov_count += 1
                    self.logger.debug("{}频次低且不在词向量中被忽略".format(word))
                    continue
            elif count < self.number_skip_freq and lt.is_decimal_str(word) or \
                    count < self.symbol_skip_freq and lt.contains_punc_or_symbol(word):
                # if self.vec_tool is not None and word not in self.vec_tool.vocab:
                self.oov_count += 1
                self.logger.debug("{}频次低被忽略".format(word))
                continue

            if count < min_frequency:
                self.oov_count += len(words_counter) - index
                self.logger.warning("{}个频次小于{}的词被忽略".format(len(word_count_list) - index, min_frequency))
                break
            if index >= max_vocab_size > 0:
                self.oov_count += len(word_count_list) - index
                self.logger.warning("{}个词超出词汇表长度{}被忽略".format(len(word_count_list) - index, max_vocab_size))
                break
            self.append_word(word, self._get_vec(word))

    def save_vocab_table(self, file_or_path_name, only_path=True):
        if only_path:
            file_or_path_name = os.path.join(file_or_path_name, 'words.txt')
        with lt.open_file(file_or_path_name, 'w') as writer:
            writer.write("__\tembedding_size\t{}\n".format(self.embedding_size))
            writer.write("__\tembedding_mode\t{}\n".format(self.embedding_mode))
            writer.write("__\tsequence_length\t{}\n".format(self.sequence_length))
            writer.write("__\tmax_sequence_length\t{}\n".format(self.max_sequence_length))
            writer.write("__\tvocab_size\t{}\n".format(len(self.vocabulary)))
            writer.write("__\trandom_count\t{}\n".format(self.random_count))
            writer.write("__\toov_count\t{}\n".format(self.oov_count))
            for word in self.vocabulary:
                writer.write(word)
                writer.write('\n')
        writer.close()

    def load_vocab_table(self, file_or_path_name, only_path=True):
        if only_path:
            file_or_path_name = os.path.join(file_or_path_name, 'words.txt')
        self._reset_vocab()
        with lt.open_file(file_or_path_name) as reader:
            for line in reader:
                line = line.rstrip('\n\r')
                if line.startswith("__\t"):
                    if self.sequence_length <= 0 and "sequence_length" in line:
                        self.sequence_length = int(line.split('\t')[-1])
                    continue
                if line not in self.words2index:
                    self.append_word(line, self._NoUseVec)  # self._get_vec(line)值无用

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    @property
    def embedding_table(self):
        return np.array(self._embedding_table)

    def embed_text(self, text_words, eos=False, sos=False):
        word_ids = []
        if sos:
            word_ids.append(self.sos)
        for word in text_words:
            word_id = self.words2index.get(word, -1)
            if word_id < 0:
                if self.oov_as_unk:
                    word_id = self.unk
                else:
                    continue
            word_ids.append(word_id)
        if eos:
            word_ids.append(self.eos)
        return word_ids

    def embed_corpus(self, text_words_iter, sequence_length=0, pad=True, eos=False, sos=False):
        # self.max_sequence_length = 0
        word_ids_list = []
        for text_words in text_words_iter:
            word_ids = self.embed_text(text_words, eos, sos)
            word_count = len(word_ids)
            if word_count > self.max_sequence_length:
                self.max_sequence_length = word_count
            if word_count > sequence_length > 0:
                word_ids = word_ids[:sequence_length]
            word_ids_list.append(word_ids)
        if sequence_length > 0:
            self.sequence_length = sequence_length
        else:
            self.sequence_length = self.max_sequence_length
        if pad:
            for word_ids in word_ids_list:
                pad_count = self.sequence_length - len(word_ids)
                for _ in range(pad_count):
                    word_ids.append(self.pad)
        return word_ids_list

    def filter_words(self, sentence_words):
        filtered_words = []
        for word in sentence_words:
            word_id = self.words2index.get(word, -1)
            if word_id >= 0:
                filtered_words.append(word)
        return filtered_words


class WordsDictionary:

    IdfBase = 1.0
    DocCountKey = "__TotalDocCount__"

    def __init__(self, idf_limit=10.0):
        self.words_table = {}
        self.idf_limit = max(2 * self.IdfBase, idf_limit)
        self._auto_zoom(1000)
        self.max_idf_weight = self.IdfBase
        self.words_table[' '] = ('w', self.IdfBase)
        self.words_table['\t'] = ('w', self.IdfBase)
        self.words_table['\n'] = ('w', self.IdfBase)
        self.words_table['\r'] = ('w', self.IdfBase)
        self.words_table['\n\r'] = ('w', self.IdfBase)
        self.words_table['\r\n'] = ('w', self.IdfBase)

    def _auto_zoom(self, real_doc_count):
        if real_doc_count <= 1000:
            self.doc_count_zoom = max(100, real_doc_count)
        else:
            self.doc_count_zoom = real_doc_count // (np.log10(real_doc_count) - 2)
        self.idf_zoom = (self.idf_limit - self.IdfBase) / np.log(self.doc_count_zoom / 2)
        self.max_idf_weight = self.IdfBase

    def calc_idf(self, doc_freq: int):
        if doc_freq >= self.doc_count_zoom:
            idf = self.IdfBase
        elif doc_freq > 0:
            idf = self.idf_zoom * np.log(self.doc_count_zoom / (doc_freq + 1)) + self.IdfBase
        else:
            idf = self.idf_limit
        return idf

    def load(self, table_file, logger=None):
        first_line = True
        skip_count = 0
        for line in lt.load_lines(table_file, strip=False, remove_empty=True):
            # 会导致#被忽略 if line.startswith('#') or line.startswith('//') or line.startswith(';'):
            #     continue
            columns = line.split('\t')
            if len(columns) < 3:
                if logger is not None:
                    logger.error(line)
                continue
            word, tag, doc_freq = columns[0], columns[1], int(columns[2])
            if first_line:
                if word == self.DocCountKey:
                    self._auto_zoom(doc_freq)
                elif logger is not None:
                    logger.error("首行应该为总文档数{}".format(self.DocCountKey))
            first_line = False
            idf_weight = self.calc_idf(doc_freq)
            if len(columns) > 3:
                idf_weight = min(self.idf_limit, float(columns[3]) * idf_weight)
            if idf_weight > self.max_idf_weight:
                self.max_idf_weight = idf_weight
            self.words_table[word] = (tag, idf_weight)
        if logger is not None:
            logger.info("共有词条{}，忽略数字时间{}".format(len(self.words_table), skip_count))
        return self

    @staticmethod
    def skip(word, tag):
        if tag.find('t') >= 0 and lt.contains_number(word):
            return True
        if tag.find('m') >= 0 and lt.contains_number(word) and lt.contains_any(word, '.%‰万亿'):
            return True
        return False

    def oov(self, word):
        return word not in self.words_table

    def get_tag(self, word, only_one=True):
        tw = self.words_table.get(word, None)
        tag = tw[0] if tw is not None else ' '
        if only_one and ';' in tag:
            tag = tag[:tag.find(';')]
        return tag

    def get_tag_char(self, word):
        tag = self.get_tag(word)
        return tag[0]

    def get_idf_weight(self, word):
        tw = self.words_table.get(word, None)
        return tw[1] if tw is not None else self.max_idf_weight

    def get_idf_sqrt(self, word):
        return math.sqrt(self.get_idf_weight(word))

    def might_be_feature(self, word_or_weight):
        if isinstance(word_or_weight, str):
            word_or_weight = self.get_idf_weight(word_or_weight)
        word_or_weight -= self.IdfBase
        value = self.idf_limit - self.IdfBase
        low, high = value * 0.2, value * 0.8
        return low <= word_or_weight  # < high
