# -*- coding:utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>
"""
纯Python实现的DoubleArrayTrie
存储用numpy数组，按需分配/回收，内存消耗小。
支持Searcher，高速搜索文本内包含的所有字典条目。
"""
import numpy as np


class MatchKind:
    Null = 0
    Prefix = 1
    Full = 2
    NeedMode = 3


class _Searcher:
    def __init__(self, text, from_index=0, only_chinese=False):
        if only_chinese:
            chars = [ch for ch in text if '\u4e00' <= ch <= '\u9fa5']
            text = "".join(chars)
        self.text = text
        self.cur_start = from_index - 1
        self.cur_stop = len(text)
        self.cur_pid = 0
        self.cur_entry = 0


class _WordsSearcher:

    def __init__(self, words_list, start_idx=0, stop_idx=0):
        self.words_list = words_list
        self.w_idx = start_idx - 1
        self.w_len = len(words_list) if stop_idx <= 0 else stop_idx
        self.word = ""
        self.c_idx = 0
        self.c_last = 0
        self.save_w_idx = -2
        self.save_c_idx = 0

    def _move(self, w_idx):
        self.c_idx = 0
        self.w_idx = w_idx
        if 0 <= w_idx < self.w_len:
            self.word = self.words_list[w_idx]
            if isinstance(self.word, tuple):
                self.word = self.word[0]
            self.c_last = len(self.word) - 1
            assert self.c_last >= 0
        else:
            self.word = ""
            self.c_last = -1

    def next_word(self):
        self._move(self.w_idx + 1)
        if self.c_last >= 0:
            return self.word[0]
        else:
            return None

    def next_char(self):
        self.c_idx += 1
        if self.c_idx <= self.c_last:
            return self.word[self.c_idx]
        else:
            return self.next_word()

    def backup_pos(self):
        self.save_w_idx = self.w_idx
        self.save_c_idx = self.c_idx

    def restore_pos(self):
        assert self.save_w_idx > -1
        self._move(self.save_w_idx)
        self.c_idx = self.save_c_idx


class DoubleArrayTrie(object):
    RootId = 1
    ValueCode = -1

    def __init__(self, key_list=None, value_list=None):
        self._slots = None
        self._parents = None
        self._alloc_size = 0
        self._used_size = 0
        self._next_empty_id = 0
        self._key_list = []
        self._value_list = []
        if key_list is not None:
            self.build(key_list, value_list)

    def __len__(self):
        return len(self._key_list)

    def keys(self):
        return self._key_list

    def values(self):
        return self._value_list

    def get(self, key):
        index = self._exact_search(key)
        return self.get_value(index) if index >= 0 else None

    def get_key(self, index):
        return self._key_list[index]

    def get_value(self, index):
        if self._value_list is not None:
            return self._value_list[index] if index >= 0 else None
        return index

    def _resize(self, new_size):
        # print("resize to: " + str(new_size))
        if new_size < 0 or new_size == self._alloc_size:
            return
        if self._used_size > 0:
            self._slots.resize(new_size, refcheck=False)
            self._parents.resize(new_size, refcheck=False)
        else:
            self._slots = np.zeros(new_size, dtype=np.int32)
            self._parents = np.zeros(new_size, dtype=np.int32)
        self._alloc_size = new_size

    def build(self, key_list, value_list):
        self._key_list = key_list
        self._value_list = value_list
        estimate_size = 4096
        for key in key_list:
            estimate_size += len(key)
        estimate_size = estimate_size * 4 // 3
        # print("estimate_size: " + str(estimate_size))

        self._used_size = 0
        self._resize(estimate_size)
        self._parents[DoubleArrayTrie.RootId] = -1
        self._used_size = DoubleArrayTrie.RootId + 1
        self._next_empty_id = self._used_size

        if len(key_list) > 0:
            root = DoubleArrayTrie.Node(code=0, depth=0, left=0, right=len(key_list))
            children = self._fetch(root)
            self._slots[DoubleArrayTrie.RootId] = self._insert(children, DoubleArrayTrie.RootId)

        # print("used_size: " + str(self._used_size))
        if self._alloc_size - self._used_size > 3000:
            self._resize(self._used_size)
        return self

    def _fetch(self, parent):
        children = []
        depth = parent.depth

        last = None
        pc = 0
        for i in range(parent.left, parent.right):
            key = self._key_list[i]
            kl = len(key)
            if kl < depth:
                continue  # 已经存储好了
            elif kl == depth:
                if last is not None:
                    raise RuntimeError("键重复：" + key)
                last = DoubleArrayTrie.Node(
                    code=DoubleArrayTrie.ValueCode, depth=depth + 1, left=i, right=i + 1)
                children.append(last)
            else:
                cc = ord(key[depth])
                if pc > cc:
                    raise RuntimeError("未排序：" + key)
                pc = cc
                if last is None or last.code != cc:
                    last = DoubleArrayTrie.Node(code=cc, depth=depth + 1, left=i, right=i + 1)
                    children.append(last)
                else:  # last.code == cc
                    last.right += 1
        return children

    def _insert(self, siblings, pid):
        min_code = siblings[0].code
        max_code = siblings[-1].code

        first_empty = True
        occupy_count = 0  # 非零统计
        eid = self._next_empty_id - 1  # 因为进循环先+1
        eid = max(min_code, eid)   # 确保child_entry为正

        while True:
            eid += 1
            max_id = eid - min_code + max_code
            if self._alloc_size <= max_id:
                self._resize(max_id * 3 // 2)
                
            if self._parents[eid] != 0:
                occupy_count += 1  # 已被使用
                continue
            elif first_empty:
                self._next_empty_id = eid  # 第一个可以使用的位置，记录？
                first_empty = False

            child_entry = eid - min_code  # 对应的 data_offset
            can_use = True
            for sibling in siblings[1:]:
                if self._parents[child_entry + sibling.code] != 0:
                    can_use = False
                    break
            if can_use:
                break

        # 得到合适的begin
        self._used_size = max(self._used_size, child_entry + max_code + 1)
        if 10 * occupy_count > (eid - self._next_empty_id + 1) * 9:
            self._next_empty_id = eid  # 从_next_empty_id 到 eid，已占用的空间在90%以上，

        for sibling in siblings:
            self._parents[child_entry + sibling.code] = pid

        for sibling in siblings:
            cid = child_entry + sibling.code
            if sibling.code == DoubleArrayTrie.ValueCode:
                self._slots[cid] = sibling.left
            else:
                children = self._fetch(sibling)
                self._slots[cid] = self._insert(children, cid)

        return child_entry

    def _exact_search(self, key, from_index=0, to_index=0):
        if to_index <= 0:
            to_index = len(key)
        if from_index > to_index:
            return -1

        pid = DoubleArrayTrie.RootId
        entry = self._slots[pid]
        for i in range(from_index, to_index):
            cid = entry + ord(key[i])
            if cid < self._alloc_size and pid == self._parents[cid]:
                pid = cid
                entry = self._slots[pid]
            else:
                return -1

        value_index = entry + DoubleArrayTrie.ValueCode
        if pid == self._parents[value_index]:
            return self._slots[value_index]
        else:
            return -1

    def lookup(self, sentence, at_pos=0, stop_pos=0):
        if stop_pos <= 0:
            stop_pos = len(sentence)
        if at_pos > stop_pos:
            return -1

        ret_index = -1
        pid = DoubleArrayTrie.RootId
        entry = self._slots[pid]
        for i in range(at_pos, stop_pos):
            cid = entry + ord(sentence[i])
            if cid < self._alloc_size and pid == self._parents[cid]:
                pid = cid
                entry = self._slots[pid]
                value_index = entry + DoubleArrayTrie.ValueCode
                if pid == self._parents[value_index]:
                    ret_index = self._slots[value_index]
            else:
                return ret_index

        value_index = entry + DoubleArrayTrie.ValueCode
        if pid == self._parents[value_index]:
            ret_index = self._slots[value_index]

        return ret_index

    def lookup_all(self, sentence, start_pos=0, stop_pos=0):
        founds = []
        if stop_pos <= 0:
            stop_pos = len(sentence)
        if start_pos > stop_pos:
            return founds

        pid = DoubleArrayTrie.RootId
        entry = self._slots[pid]
        for i in range(start_pos, stop_pos):
            cid = entry + ord(sentence[i])
            if cid < self._alloc_size and pid == self._parents[cid]:
                pid = cid
                entry = self._slots[pid]
                value_index = entry + DoubleArrayTrie.ValueCode
                if pid == self._parents[value_index]:
                    founds.append(self._slots[value_index])
            else:
                return founds

        value_index = entry + DoubleArrayTrie.ValueCode
        if pid == self._parents[value_index]:
            founds.append(self._slots[value_index])

        return founds

    def match_longest(self, sentence, at_pos=0, stop_pos=0):
        if stop_pos <= 0:
            stop_pos = len(sentence)
        ret_index, match_kind = -1, MatchKind.Null

        pid = DoubleArrayTrie.RootId
        entry = self._slots[pid]
        for i in range(at_pos, stop_pos):
            cid = entry + ord(sentence[i])
            if cid < self._alloc_size and pid == self._parents[cid]:
                pid = cid
                entry = self._slots[pid]
                value_index = entry + DoubleArrayTrie.ValueCode
                if pid == self._parents[value_index]:
                    ret_index = self._slots[value_index]
                    match_kind = MatchKind.Prefix
            else:
                return ret_index, match_kind

        value_index = entry + DoubleArrayTrie.ValueCode
        if pid == self._parents[value_index]:
            ret_index = self._slots[value_index]
            match_kind = MatchKind.Full
        else:
            match_kind = MatchKind.NeedMode

        return ret_index, match_kind

    class Node:

        def __init__(self, code, depth, left, right):
            self.code = code
            self.depth = depth
            self.left = left
            self.right = right

    def search_next(self, searcher):
        n_len = len(searcher.text)
        while True:
            searcher.cur_stop += 1
            if searcher.cur_stop >= n_len:
                # 指针到头了，将起点往前挪一个，重新开始，状态归零
                searcher.cur_start += 1
                if searcher.cur_start >= n_len:
                    return None, None, -1
                searcher.cur_stop = searcher.cur_start
                searcher.cur_pid = DoubleArrayTrie.RootId
                searcher.cur_entry = self._slots[searcher.cur_pid]

            cid = searcher.cur_entry + ord(searcher.text[searcher.cur_stop])
            if cid < self._alloc_size and searcher.cur_pid == self._parents[cid]:
                searcher.cur_pid = cid
                searcher.cur_entry = self._slots[searcher.cur_pid]
                value_index = searcher.cur_entry + DoubleArrayTrie.ValueCode
                if searcher.cur_pid == self._parents[value_index]:
                    index = self._slots[value_index]
                    return self._key_list[index], self.get_value(index), searcher.cur_stop + 1
            else:
                # 转移失败，也将起点往前挪一个，重新开始，状态归零
                searcher.cur_stop = n_len
                continue

    def search_all(self, text):
        results = []
        searcher = _Searcher(text)
        sr = self.search_next(searcher)
        while sr[0] is not None:
            results.append(sr)
            sr = self.search_next(searcher)
        return results

    def search_in_words(self, words_list, allow_overlap=False, allow_part_word=True, from_idx=0):
        searcher = _WordsSearcher(words_list, from_idx)

        found_list = []
        last_overlap = False
        last_w, last_c = 0, 0
        ch = searcher.next_char()

        searcher.backup_pos()
        cur_pid = DoubleArrayTrie.RootId
        cur_entry = self._slots[cur_pid]
        while ch is not None:
            cid = cur_entry + ord(ch)
            if cid < self._alloc_size and cur_pid == self._parents[cid]:
                cur_pid = cid
                cur_entry = self._slots[cur_pid]
                value_index = cur_entry + DoubleArrayTrie.ValueCode
                if cur_pid == self._parents[value_index]:
                    full_word = searcher.save_c_idx == 0 and searcher.c_idx == searcher.c_last
                    if allow_part_word or full_word:
                        if last_overlap:
                            found_list.pop(-1)
                        if searcher.w_idx > last_w or searcher.w_idx == last_w and searcher.c_idx > last_c:
                            if not allow_overlap:
                                last_overlap = True
                                last_w, last_c = searcher.w_idx, searcher.c_idx
                            index = self._slots[value_index]
                            word, value = self._key_list[index], self.get_value(index)
                            found_list.append((word, value, searcher.save_w_idx, searcher.w_idx + 1, full_word))
                ch = searcher.next_char()
            else:
                # 转移失败，也将起点往前挪一个，重新开始，状态归零
                searcher.restore_pos()
                if allow_part_word:
                    ch = searcher.next_char()
                else:
                    ch = searcher.next_word()
                last_overlap = False
                searcher.backup_pos()
                cur_pid = DoubleArrayTrie.RootId
                cur_entry = self._slots[cur_pid]
        return found_list

    def search_words_first(self, words_list, full_word=True, from_idx=0):
        fw, fv, fi = None, None, -1
        searcher = _WordsSearcher(words_list, from_idx)
        ch = searcher.next_char()
        cur_pid = DoubleArrayTrie.RootId
        cur_entry = self._slots[cur_pid]
        while ch is not None:
            cid = cur_entry + ord(ch)
            if cid < self._alloc_size and cur_pid == self._parents[cid]:
                cur_pid = cid
                cur_entry = self._slots[cur_pid]
                value_index = cur_entry + DoubleArrayTrie.ValueCode
                if cur_pid == self._parents[value_index]:
                    if not full_word or searcher.c_idx == searcher.c_last:
                        index = self._slots[value_index]
                        fw = self._key_list[index]
                        fv = self.get_value(index)
                        fi = searcher.w_idx + 1
                ch = searcher.next_char()
            else:
                break
        return fw, fv, fi


if __name__ == "__main__":
    words = ["", "北", "北京", "京东", "北京东", "东方", "京东方", "东方电机", "东方电机厂"]
    words = sorted(words)
    dat = DoubleArrayTrie(words, words)

    for idx, wd in enumerate(words):
        if dat.get(wd) != wd:
            print("MISSING\t" + wd)
    print("get('北京')\t" + str(dat.get('北京')))
    print("get('北京西')\t" + str(dat.get('北京西')))
    print("lookup_all('北京西')\t" + dat.get_value(dat.lookup('北京西')))

    print(dat.search_all("北京东方电机厂"))

    print(dat.search_in_words("北京 东方 电机厂".split(), False))
    print(dat.search_in_words("北京 东方 电机厂".split(), True, False))
    print(dat.search_in_words("北京 东方 电机厂".split(), True, True))

    print(dat.search_words_first("北京 东方 电机厂".split(), False))
    print(dat.search_words_first("北京 东方 电机厂".split(), True))

