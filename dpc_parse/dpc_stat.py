# -*- coding: utf-8 -*-

import os
import collections

from mylib import txt_lib as lt
from mylib import word_embed as lw
from mylib import utils as lu
from dpc_parse import api


class DependencyStore:

    def __init__(self, w2v_model):
        self.forward_search = {}
        self.backward_search = {}
        self.forward_counter = collections.Counter()
        self.backward_counter = collections.Counter()
        self.kinds_estimates = collections.defaultdict(float)
        self.w2v_model = w2v_model

    @classmethod
    def _add_search(cls, search, from_word, to_word, dpc_kind):
        word_links = search.get(from_word, None)
        if word_links is None:
            word_links = {}
            search[from_word] = word_links
        kinds_count = word_links.get(to_word, None)
        if kinds_count is None:
            kinds_count = collections.Counter()
            word_links[to_word] = kinds_count
        kinds_count[dpc_kind] += 1

    def stat(self, from_word, to_word, dpc_kind, from_tag=None, to_tag=None):
        self._add_search(self.forward_search, from_word, to_word, dpc_kind)
        self._add_search(self.backward_search, to_word, from_word, dpc_kind)
        if from_tag:
            self._add_search(self.forward_search, from_tag, to_word, dpc_kind)
        if to_tag:
            self._add_search(self.backward_search, to_tag, from_word, dpc_kind)

    def clean_noise(self, min_occur=2):
        def clean_and_count(link_details):
            words_counter = {w: sum(kcs) for w, kcs in link_details.items()}
            for w, c in words_counter.items():
                if c < min_occur:
                    link_details.pop(w)
            return sum(c for c in words_counter.values() if c >= min_occur)

        for word, links in self.forward_search.items():
            total = clean_and_count(links)
            if total > 0:
                self.forward_counter[word] = total
        for word, links in self.backward_search.items():
            total = clean_and_count(links)
            if total > 0:
                self.backward_counter[word] = total

    def _stat_kinds(self, kinds_counter, factor=1.0):
        for k, c in kinds_counter.items():
            self.kinds_estimates[k] += c * factor

    LinkDecay = 0.66
    MinSimilarity = 0.54

    def _simple_stat(self, link_details, link_word, factor=0.03):
        # 词性与次的映射，因为聚集了同一类的所有词，所以系数较小
        kinds_count = link_details.get(link_word, None)
        if kinds_count:
            self._stat_kinds(kinds_count, factor)
            return sum(kinds_count.values()) * factor
        else:
            return 0.0

    def _sampling_stat(self, link_details, link_word):
        similar_words = self.w2v_model.find_similar(link_word, 8)
        similar_words = [(w, s * self.LinkDecay) for w, s in similar_words if s >= self.MinSimilarity]
        similar_words.append((link_word, 1.0))

        sampling_data = lu.TopTracker(4)
        for word, factor in similar_words:
            kinds_count = link_details.get(word, None)
            if kinds_count:
                sampling_data.add(kinds_count, sum(kinds_count.values())*factor)
        total_weight = 0.0
        for kinds_count, weight in sampling_data:
            total_weight += weight
            self._stat_kinds(kinds_count)
        return total_weight

    def estimate(self, from_word, to_word, from_tag=None, to_tag=None):
        w1, w2, w3, w4 = 0.0, 0.0, 0.0, 0.0
        self.kinds_estimates.clear()
        if from_word in self.forward_counter:
            w1 = self._sampling_stat(self.forward_search[from_word], to_word) / self.forward_counter[from_word]
        if to_word in self.backward_counter:
            w2 = self._sampling_stat(self.backward_search[to_word], from_word) / self.backward_counter[to_word]
        if from_tag and from_tag in self.forward_counter:
            w3 = self._simple_stat(self.forward_search[from_tag], to_word) / self.forward_counter[from_tag]
        if to_tag and to_tag in self.backward_counter:
            w4 = self._simple_stat(self.backward_search[to_tag], from_word) / self.backward_counter[to_tag]
        w = w1 + w2 + w3 + w4
        print(f"{from_word}\t{to_word}\t{w:.4f}\t{w1:.4f}\t{w2:.4f}\t{w3:.4f}\t{w4:.4f}")
        return w


def stat_dpc_file(file_name, dpc_store):
    article_count = 0
    for line in lt.open_file(file_name):
        line = line.strip()
        if line:
            for sentence in api.parse_tagged_text(line):
                for word, tag, link_to, link_kind in sentence[1:]:
                    dpc_store.stat(word, sentence[link_to][0], link_kind)
        else:
            article_count += 1
            if article_count % 1000 == 0:
                logger.info(f"\t{article_count}")


def stat_dpc_files(path, dpc_store):
    for name in os.listdir(path):
        file_name = os.path.join(path, name)
        if os.path.isdir(file_name) or not name.endswith('.dp.txt'):
            continue
        logger.info("文件：" + file_name)
        stat_dpc_file(file_name, dpc_store)


def learn():
    dpc_store = DependencyStore(None)
    stat_dpc_files(work_path, dpc_store)
    dpc_store.clean_noise()
    lt.save_pickle(dpc_store, work_path + r"\dpc_db.pkl")


def verify():
    data = "金字火腿│ns〖1-主谓-4〗 连续│d〖2-状中-4〗 三│m〖3-状中-4〗 涨停│v〖4-连谓-5〗 " \
           "收│v〖5-核心-0〗 深交所│nt〖6-定中-8〗 关注│n〖7-定中-8〗 函│nz〖8-动宾-5〗 。│wj〖9-标点-5〗〗"

    dpc_store = lt.load_pickle(work_path + r"\dpc_db.pkl")
    logger.info("模型已加载！")
    dpc_store.w2v_model = lw.WordVectorTool(r"F:\Python\stock_search\_data\dict\news_vec.s256.bin")
    sentence = api.parse_tagged_text(data)[0]
    for idx1 in range(1, len(sentence)):
        tag = sentence[idx1][1]
        word = sentence[idx1][0]
        for idx2 in range(len(sentence)):
            if idx1 != idx2:
                dpc_store.estimate(word, sentence[idx2][0], tag, sentence[idx2][1])


if __name__ == "__main__":
    logger = lu.get_logger()
    work_path = r"G:\Data\dpc"
    # learn()
    verify()
