# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>
"""
结巴分词，Fool词性标注、实体识别
"""
import os
import queue
import jieba
import fool
import multiprocessing as mp

import mylib.word_embed as lw


StableNameTypes = frozenset([
    'person',
    'org',
    'company',
    'job'
])
AllNameTypes = frozenset([
    'person',
    'org',
    'company',
    'job',
    'time',
    'number',
    'location',
])
FoolNameTypes2Tag = {
    'time': 't',
    'person': 'nr',
    'org': 'nt',
    'company': 'nt',
    'job': 'nn',
    'number': 'mq',
    'location': 'ns',
}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LexicalAnalyzer:

    def close(self):
        pass

    def cut(self, text):
        pass

    def tag(self, text):
        pass

    def cut_and_ner(self, text):
        pass

    def batch_cut_and_ner(self, text_list, batch_sentence_count=0, max_sentence_len=0):
        pass


class MixedLexicalAnalyzer(LexicalAnalyzer):

    def __init__(self, want_entity_types=None):
        self.entity_types = frozenset(AllNameTypes if want_entity_types is None else want_entity_types)
        self.entity_filter = None

    def cut(self, text):
        return lw.jieba_cut(text)

    def tag(self, text):
        word_list = lw.jieba_cut(text)
        tag_list = fool.LEXICAL_ANALYSER.pos([word_list])
        if isinstance(tag_list[0], list):
            tag_list = tag_list[0]
        tagged_words_list = [(word, tag) for word, tag in zip(word_list, tag_list)]
        return tagged_words_list

    def cut_and_ner(self, text):
        word_list = lw.jieba_cut(text)
        tag_list = fool.LEXICAL_ANALYSER.pos([word_list])
        if isinstance(tag_list, list) and isinstance(tag_list[0], list):
            tag_list = tag_list[0]
        tagged_words_list = [(word, tag) for word, tag in zip(word_list, tag_list)]
        name_entities = fool.LEXICAL_ANALYSER.ner([text])
        if isinstance(name_entities, list) and isinstance(name_entities[0], list):
            name_entities = name_entities[0]
        word_or_entity_tags = self.merge_name_entities(text, tagged_words_list, name_entities)
        return word_or_entity_tags

    def cut_and_ner_one_batch(self, text_list):
        word_list_list = [lw.jieba_cut(text) for text in text_list]
        tag_list_list = fool.LEXICAL_ANALYSER.pos(word_list_list)
        name_entities_list = fool.LEXICAL_ANALYSER.ner(text_list)

        word_or_entity_tags_list = []
        for index, text in enumerate(text_list):
            tagged_words = [(word, tag) for word, tag in zip(word_list_list[index], tag_list_list[index])]
            word_or_entity_tags = self.merge_name_entities(text, tagged_words, name_entities_list[index])
            word_or_entity_tags_list.append(word_or_entity_tags)

        return word_or_entity_tags_list

    def schedule_cut_and_ner(self, aligned_sentence_list, char_count_per_batch):
        word_or_entity_tags_list = []
        sentences_in_batch, max_len_in_batch = [], 0
        index, count = 0, len(aligned_sentence_list)
        while index < count:
            sentence = aligned_sentence_list[index]
            max_len = max(len(sentence), max_len_in_batch)
            if len(sentences_in_batch) > 0 and max_len * (len(sentences_in_batch) + 1) > char_count_per_batch:
                word_or_entity_tags_list.extend(self.cut_and_ner_one_batch(sentences_in_batch))
                sentences_in_batch = []
                max_len = len(sentence)
            sentences_in_batch.append(sentence)
            max_len_in_batch = max_len
            index += 1
        word_or_entity_tags_list.extend(self.cut_and_ner_one_batch(sentences_in_batch))

        return word_or_entity_tags_list

    def batch_cut_and_ner(self, article_list, batch_sentence_count=64, max_sentence_len=128):
        if isinstance(article_list, str):
            article_list = [article_list]
        if batch_sentence_count == 0:
            batch_sentence_count = 64
        if max_sentence_len == 0:
            max_sentence_len = 128

        # 文章断句
        all_article_sentence_list, article_idx_list = \
            lw.articles_to_sentences(article_list, align_length=max_sentence_len)
        # 自动调度处理
        word_or_entity_tags_list = self.schedule_cut_and_ner(
            all_article_sentence_list, max_sentence_len * batch_sentence_count)
        # 重组恢复文章结构
        article_done = 0
        article_words_tags_list = []
        for index, sentence in enumerate(all_article_sentence_list):
            article_idx = article_idx_list[index]
            word_or_entity_tags = word_or_entity_tags_list[index]
            while article_idx > article_done:
                # 文章正文为空，被忽略
                article_done += 1
                article_words_tags_list.append([])
            if article_idx == article_done:
                article_done += 1
                article_words_tags_list.append(word_or_entity_tags)
            elif article_idx == article_done - 1:
                article_words_tags_list[article_idx].extend(word_or_entity_tags)
            else:
                raise RuntimeError("文章序号错误：{}/{}/{}".format(article_idx, article_done, len(article_list)))

        while article_done < len(article_list):
            # 文章正文为空，被忽略
            article_done += 1
            article_words_tags_list.append([])
        return article_words_tags_list

    def merge_name_entities(self, sentence, words_tags, name_entities):
        word_or_entity_tag_list = []
        w_start, s_len = 0, len(sentence)
        index, count = 0, len(words_tags)
        for e_start, e_stop, e_type, e_name in name_entities:
            if e_type not in self.entity_types:
                continue
            if self.entity_filter and self.entity_filter(e_name):
                continue
            if e_start + len(e_name) == e_stop - 1:
                e_stop -= 1
            if e_start + len(e_name) != e_stop or e_start >= s_len or sentence[e_start] != e_name[0] or e_stop > s_len:
                pos = sentence.find(e_name, w_start)
                if pos >= w_start:
                    e_start, e_stop = pos, pos + len(e_name)
                else:
                    # # print("实体位置错误：({}, {}, {})，原文：{}".format(e_start, e_stop, e_name, sentence))
                    continue

            while w_start < e_start:
                tw = words_tags[index]
                index += 1
                w_start += len(tw[0])
                word_or_entity_tag_list.append(tw)
            if w_start == e_start:
                wi = index
                w_stop = w_start
                while w_stop < e_stop:
                    tw = words_tags[wi]
                    wi += 1
                    w_stop += len(tw[0])
                if w_stop == e_stop:
                    word_or_entity_tag_list.append((e_name, FoolNameTypes2Tag[e_type]))
                    index = wi
                    w_start = e_stop
                # 断词位置不一致会出错 elif e_name[-1] in "十百千万亿":
                #     word_or_entity_tag_list.append((e_name, FoolNameTypes2Tag[e_type]))
                #     index = wi
                #     w_start = e_stop

            while w_start < e_stop:
                tw = words_tags[index]
                index += 1
                w_start += len(tw[0])
                word_or_entity_tag_list.append(tw)

        if index == 0:
            return words_tags
        elif index < count:
            word_or_entity_tag_list.extend(words_tags[index:])
        return word_or_entity_tag_list

    def merge_ner_and_user_words(self, sentence_list, words_tags_list, name_entities_list):
        word_or_entity_tags_list = []
        for index, sentence in enumerate(sentence_list):
            words_tags = words_tags_list[index]
            name_entities = [] if name_entities_list is None else name_entities_list[index]
            try:
                word_or_entity_tags = self.merge_name_entities(sentence, words_tags, name_entities)
                word_or_entity_tags_list.append(word_or_entity_tags)
            except IndexError:
                word_or_entity_tags_list.append(words_tags)
                print("无效字符：" + sentence)
        return word_or_entity_tags_list


class MixedLexicalAnalyzerMp(MixedLexicalAnalyzer):

    def __init__(self, want_entity_types=None, user_dict_files=None):
        MixedLexicalAnalyzer.__init__(self, want_entity_types)
        self._sync_params = queue.Queue()
        self._sen_queue1 = mp.Queue(16)
        self._sen_queue2 = mp.Queue(16)
        self._words_queue = mp.Queue(16)
        self._results_queue = mp.Queue()
        self._seg_process = mp.Process(target=mp_jieba_seg, args=(user_dict_files, self._sen_queue1, self._words_queue))
        self._pos_ner_process = mp.Process(target=mp_fool_pos_ner,
                                           args=(self._sen_queue2, self._words_queue, self._results_queue))
        self._seg_process.start()
        self._pos_ner_process.start()

    def close(self):
        self._seg_process.terminate()
        self._pos_ner_process.terminate()

    def _send_mp_task(self, sentence_list):
        # print("{}\t发送 {} lines".format(mp.current_process().name, len(sentence_list)))
        self._sync_params.put(sentence_list)
        self._sen_queue1.put(sentence_list)
        self._sen_queue2.put(sentence_list)

    def schedule_cut_and_ner(self, aligned_sentence_list, char_count_per_batch):
        # 发送所有批次
        sentences_in_batch, max_len_in_batch = [], 0
        index, count = 0, len(aligned_sentence_list)
        while index < count:
            sentence = aligned_sentence_list[index]
            max_len = max(len(sentence), max_len_in_batch)
            if len(sentences_in_batch) > 0 and max_len * (len(sentences_in_batch) + 1) > char_count_per_batch:
                self._send_mp_task(sentences_in_batch)
                sentences_in_batch = []
                max_len = len(sentence)
            sentences_in_batch.append(sentence)
            max_len_in_batch = max_len
            index += 1
        if len(sentences_in_batch) > 0:
            self._send_mp_task(sentences_in_batch)

        # 等待后台NER并返回
        word_or_entity_tags_list = []
        while not self._sync_params.empty():
            words_tags_list, name_entities_list = self._results_queue.get()
            sentence_list = self._sync_params.get()
            word_or_entity_tags_list.extend(self.merge_ner_and_user_words(
                sentence_list, words_tags_list, name_entities_list))
        return word_or_entity_tags_list


def mp_jieba_seg(user_dict_files, sentence_queue: mp.Queue, words_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if user_dict_files is not None:
        for user_dict_file in user_dict_files:
            jieba.load_userdict(user_dict_file)
    while True:
        sentence_list = sentence_queue.get()
        words_list = [lw.jieba_cut(sentence) for sentence in sentence_list]
        words_queue.put(words_list)


def mp_fool_pos_ner(sentences_queue: mp.Queue, words_queue: mp.Queue, results_queue: mp.Queue):
    analyzer = fool.lexical.LexicalAnalyzer()
    while True:
        sentence_list = sentences_queue.get()
        entities_list = analyzer.ner(sentence_list)

        words_list = words_queue.get()
        tags_list = analyzer.pos(words_list)
        words_tags = [list(zip(words, tags)) for words, tags in zip(words_list, tags_list)]
        results_queue.put((words_tags, entities_list))


if __name__ == "__main__":
    tt1 = '医保7600亿元浪费93.3亿元占比13.5%，总发电量由2017年的约211.53万兆瓦时增至约319.263万兆瓦时，增幅约51%。'
    tt2 = "8.香港出现比特币和以太坊ATM机据generalbytes报道，近日，在香港的上环文咸东街的NakedHub裸心社安装了比特币和以太坊ATM机。"
    print(lw.jieba_cut(tt1))
    _analyzer = MixedLexicalAnalyzer()
    print(_analyzer.batch_cut_and_ner([tt1, tt2]))
    _analyzer.close()
