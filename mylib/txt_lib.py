# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

"""
文本处理工具
"""

import re
import json
import unicodedata
import collections
import numpy as np
import pickle


ColumnSeparator = '│'
re_spaces = re.compile(r'\s+')
re_numbers = re.compile(r"[+\u002d]?[0-9]+")
re_decimals = re.compile(r"[+\u002d]?[0-9,]+(.[0-9]+)?")
re_scientific_decimals = re.compile(r"[+\u002d]?\d+.?\d*[Ee][+\u002d]?\d+")
re_sentence_ends = re.compile(r"[。！？!?；]")
re_line_break_chars = re.compile(r'[\u000d\u000a]+')
SentenceEndChars = "。！？!?"


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
            (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_cjk_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((0x4E00 <= cp <= 0x9FFF) or  #
            (0x3400 <= cp <= 0x4DBF) or  #
            (0x20000 <= cp <= 0x2A6DF) or  #
            (0x2A700 <= cp <= 0x2B73F) or  #
            (0x2B740 <= cp <= 0x2B81F) or  #
            (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or  #
            (0x2F800 <= cp <= 0x2FA1F)):  #
        return True

    return False


def is_number_str(str_data):
    return str_data is not None and re_numbers.fullmatch(str_data)


def contains_number(text):
    for ch in text:
        if '0' <= ch <= '9':  # or ch == '.' or ch == '+' or ch == '-'
            return True
    return False


def is_decimal_str(str_data):
    return re_decimals.fullmatch(str_data) or re_scientific_decimals.fullmatch(str_data)


def clean_any_space(text):
    text, _ = re_spaces.subn("", text)
    return text


def clean_line_breaks(text, replace_to=""):
    text, _ = re_line_break_chars.subn(replace_to, text)
    return text


def is_all_ascii(text):
    for ch in text:
        if ord(ch) >= 128:
            return False
    return True


def clean_not_ascii(text, except_chars='', clean_spaces=True):
    result = []
    for ch in text:
        if ch in except_chars:
            result.append(ch)
            continue
        if ord(ch) >= 128:
            continue
        if clean_spaces and ch <= ' ':
            continue
        result.append(ch)
    return "".join(result)


def all_letter_or_number(text, ignore_chars=""):
    for ch in text:
        if ignore_chars.find(ch) >= 0:
            continue
        if '0' <= ch <= '9':  # or ch == '.' or ch == '+' or ch == '-'
            continue
        elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch == '_':
            continue
        else:
            return False
    return True


def contains_any(text: str, search_words):
    if text is None or len(text) <= 0:
        return False
    for word in search_words:
        if text.find(word) >= 0:
            return True
    return False


def count_any(text: str, words_list_or_chars_str):
    if text is None or len(text) <= 0:
        return 0
    result, t_len = 0, len(text)
    for sub in words_list_or_chars_str:
        start = 0
        while start < t_len:
            idx = text.find(sub, start)
            if idx < 0:
                break
            result += 1
            start = idx + len(sub)
    return result


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fa5'


def contains_chinese(text):
    for ch in text:
        if '\u4e00' <= ch <= '\u9fa5':
            return True
    return False


def is_all_chinese(text):
    for ch in text:
        if ch < '\u4e00' or ch > '\u9fa5':
            return False
    return len(text) > 0


def clean_not_chinese(text, letters=False, numbers=False):
    result = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fa5':
            result.append(ch)
        elif letters and 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
            result.append(ch)
        elif numbers and '0' <= ch <= '9':
            result.append(ch)
    return "".join(result)


def is_punc_or_symbol(ch):
    if '0' <= ch <= '9':  # or ch == '.' or ch == '+' or ch == '-'
        return False
    elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch == '_':
        return False
    elif '\u4e00' <= ch <= '\u9fa5':
        return False
    return True


def all_punc_or_symbol(text, ignore_chars=""):
    for ch in text:
        if ignore_chars.find(ch) >= 0:
            continue
        if '0' <= ch <= '9':  # or ch == '.' or ch == '+' or ch == '-'
            return False
        elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch == '_':
            return False
        elif '\u4e00' <= ch <= '\u9fa5':
            return False
    return True


def contains_punc_or_symbol(text, ignore_chars=""):
    for ch in text:
        if ignore_chars.find(ch) >= 0:
            continue
        if '0' <= ch <= '9':  # or ch == '.' or ch == '+' or ch == '-'
            pass
        elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch == '_':
            pass
        elif '\u4e00' <= ch <= '\u9fa5':
            pass
        else:
            return True
    return False


def clean_punc_or_symbol(text: str, ignore_chars=None, allow_dot_num=True):
    result = []
    last_i = len(text) - 1
    prev_is_num = False
    for i, ch in enumerate(text):
        if ignore_chars is not None and ch in ignore_chars:
            result.append(ch)
        elif '0' <= ch <= '9':
            result.append(ch)
        elif allow_dot_num and (ch == '.' or ch == '+' or ch == '-'):
            if (ch != '.' or prev_is_num) and (i < last_i and '0' <= text[i+1] <= '9'):
                result.append(ch)
        elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch == '_':
            result.append(ch)
        elif '\u4e00' <= ch <= '\u9fa5':
            result.append(ch)
        prev_is_num = '0' <= ch <= '9'
    return "".join(result)


def open_file(filename: str, mode: str = 'rt', encoding=None, buffering=-1):
    if encoding is None or len(encoding) == 0:
        mode = mode.lower()
        if ('w' in mode or 'a' in mode) and 'b' not in mode:
            encoding = TestFileEncodings[0]
        elif 'r' in mode and 'b' not in mode:
            encoding = judge_file_encoding(filename)
    file = open(filename, mode, encoding=encoding, buffering=buffering)
    if mode == 'w' or mode == 'wt' or mode == 'a' or mode == 'at':
        file.write_line = Writer(file).write_line
    return file


class Writer:

    def __init__(self, file):
        self.file = file

    def write_line(self, line):
        if not isinstance(line, str):
            line = str(line)
        self.file.write(line)
        self.file.write('\n')


def load_lines(filename, strip=True, remove_empty=False, to_lower_case=False, encoding=None):
    file = open_file(filename, encoding=encoding)
    lines = file.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    else:
        lines = [line.rstrip('\n\r') for line in lines]
    if remove_empty:
        lines = [line for line in lines if len(line) > 0]
    if to_lower_case:
        lines = [line.lower() for line in lines]
    file.close()
    return lines


def save_lines(filename, lines, append=False):
    file = open_file(filename, 'a' if append else 'w')
    for line in lines:
        file.write(line)
        file.write('\n')
    file.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='utf-8')


def save_pickle(data_obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data_obj, f)


def load_json(filename):
    with open_file(filename, 'r') as f:
        return json.load(f)


def save_json(data_obj, filename, indent=None):
    with open_file(filename, 'w') as f:
        json.dump(data_obj, f, ensure_ascii=False, indent=indent)


# 说明：UTF兼容ISO8859-1和ASCII，GB18030兼容GBK，GBK兼容GB2312，GB2312兼容ASCII
TestFileEncodings = ['UTF-8-SIG', 'UTF-16-LE', 'UTF-16-BE', 'UTF-8', 'GB18030', 'BIG5', 'ISO-8859-1']
TestFileEncodingsBOM = [b'\xef\xbb\xbf', b'\xff\xfe', b'\xfe\xff']
re_TestDecodeErrorPos = re.compile(r"in position (\d+)[-: ]")


def judge_file_encoding(filename: str, test_decode_byte_count=1024, max_encode_bytes_per_char=3):
    with open(filename, 'rb') as f:
        test_bytes = f.read(test_decode_byte_count)
    for index, encoding in enumerate(TestFileEncodings):
        if index >= len(TestFileEncodingsBOM):
            try:
                test_bytes.decode(encoding)
                return encoding
            except UnicodeDecodeError as ex:
                if len(test_bytes) == test_decode_byte_count:  # 可能被截断
                    spr = re_TestDecodeErrorPos.search(str(ex))
                    if spr is not None:
                        pos = int(spr.group(1))
                        if pos >= len(test_bytes) - max_encode_bytes_per_char:
                            return encoding
        elif test_bytes.startswith(TestFileEncodingsBOM[index]):
            return encoding
    raise UnicodeError("文件编码检测失败，可以尝试[errors='ignore']方式打开文件")


def cut_long_sentence(sentence, cut_length, cut_chars="；，：;、,:·"):
    parts = []
    s_len = len(sentence)
    while s_len > cut_length:
        cut_pos = cut_length - 1
        for i in reversed(range(cut_length // 2, cut_pos)):
            ch = sentence[i]
            if ch in cut_chars:
                cut_pos = i + 1
                break
        parts.append(sentence[:cut_pos])
        sentence = sentence[cut_pos:]
        s_len -= cut_pos

    if s_len > 0:
        parts.append(sentence)
    return parts


def split_sentences(article, split_after_chars=SentenceEndChars, max_sentence_len=256, skip_start=0, skip_stop=0):
    sentence_list = []
    if not isinstance(split_after_chars, set):
        split_after_chars = set(split_after_chars)

    in_quote = False
    in_brace, brace_pos = 0, 0
    start, n_last = 0, len(article) - 1
    for i, ch in enumerate(article):
        if ch == '“' or ch == '"':
            in_quote = not in_quote  # 不用True，存在中文引号不匹配的的情形，“沪伦通“
        elif ch == "”":
            in_quote = False
        elif ch in '([{（［【':
            in_brace += 1
            brace_pos = i  # 避免括号丢一半
        elif ch in ')]}）］】' and in_brace > 0:
            in_brace -= 1
        if i == n_last or ch in split_after_chars and not in_quote and (brace_pos == 0 or i - brace_pos > 8) \
                and (i < skip_start or i >= skip_stop - 1):
            sentence = article[start:i+1]
            if max_sentence_len > 0 and len(sentence) <= max_sentence_len:
                sentence_list.append(sentence)
            else:
                sentence_list.extend(cut_long_sentence(sentence, max_sentence_len))
            start = i + 1
    return sentence_list


def split_paragraphs(article, split_after_chars=SentenceEndChars, max_sentence_len=256, strip=True):
    paragraph_list = []
    sentence_list = split_sentences(article, split_after_chars, max_sentence_len)
    last_len, last_sentence = 0, None
    for sentence in sentence_list:
        if strip:
            sentence = sentence.strip()
        s_len = len(sentence)
        if s_len <= 0:
            continue
        if last_len == 0:
            last_len, last_sentence = s_len, sentence
        elif s_len + last_len > max_sentence_len:
            paragraph_list.append(last_sentence)
            last_len, last_sentence = s_len, sentence
        else:
            last_len += s_len
            last_sentence += sentence
    if last_len > 0:
        paragraph_list.append(last_sentence)
    return paragraph_list


def calc_bow_similarity(words_counter1, words_counter2, smoothing=0):
    wcs1 = sorted(words_counter1.items(), key=lambda pair: pair[0])
    wcs2 = sorted(words_counter2.items(), key=lambda pair: pair[0])

    i1, i2 = 0, 0
    l1, l2 = len(wcs1), len(wcs2)
    if l1 <= 0 or l2 <= 0:
        return 0.0

    total, matched = 0, 0
    wc1, wc2 = wcs1[i1], wcs2[i2]
    while True:
        key1, key2 = wc1[0], wc2[0]
        if key1 > key2:
            total += wc2[1]
            i2 += 1
            if i2 >= l2:
                break
            wc2 = wcs2[i2]
        elif key1 < key2:
            total += wc1[1]
            i1 += 1
            if i1 >= l1:
                break
            wc1 = wcs1[i1]
        else:
            w1, w2 = wc1[1], wc2[1]
            total += max(w1, w2)
            matched += min(w1, w2)
            i1 += 1
            i2 += 1
            if i1 >= l1:
                break
            wc1 = wcs1[i1]
            if i2 >= l2:
                break
            wc2 = wcs2[i2]

    while i1 < l1:
        total += wcs1[i1][1]
        i1 += 1
    while i2 < l2:
        total += wcs2[i2][1]
        i2 += 1

    return (matched + smoothing) / (total + smoothing) if total > 0 else 0.0


def calc_bow_coverage(full_bow_counter, sub_bow_counter, smoothing=0):
    full_wcs = sorted(full_bow_counter.items(), key=lambda pair: pair[0])
    sub_wcs = sorted(sub_bow_counter.items(), key=lambda pair: pair[0])

    i1, i2 = 0, 0
    l1, l2 = len(full_wcs), len(sub_wcs)
    if l1 <= 0 or l2 <= 0:
        return 0.0

    total, matched = 0, 0
    wc1, wc2 = full_wcs[i1], sub_wcs[i2]
    while True:
        key1, key2 = wc1[0], wc2[0]
        if key1 > key2:
            total += wc2[1]
            i2 += 1
            if i2 >= l2:
                break
            wc2 = sub_wcs[i2]
        elif key1 < key2:
            # total += wc1[1]
            i1 += 1
            if i1 >= l1:
                break
            wc1 = full_wcs[i1]
        else:
            w1, w2 = wc1[1], wc2[1]
            total += w2  # + w2
            matched += min(w1, w2)  # * 2
            i1 += 1
            i2 += 1
            if i1 >= l1:
                break
            wc1 = full_wcs[i1]
            if i2 >= l2:
                break
            wc2 = sub_wcs[i2]

    # while i1 < l1:
    #     total += full_wcs[i1][1]
    #     i1 += 1
    while i2 < l2:
        total += sub_wcs[i2][1]
        i2 += 1

    return (matched + smoothing) / (total + smoothing) if total > 0 else 0.0


def get_str_grams(str_data, gram_len, extra_head_gram=0, extra_tail_gram=0):
    grams = []
    if str_data is None:
        return grams
    s_len = len(str_data)
    if 0 < s_len <= gram_len:
        grams.append(str_data)
    else:
        for i in range(s_len - gram_len):
            gram = str_data[i:i + gram_len]
            grams.append(gram)

    if gram_len > extra_head_gram > 0 and s_len >= extra_head_gram:
        grams.append("［" + str_data[:extra_head_gram])
    if gram_len > extra_tail_gram > 0 and s_len >= extra_head_gram + extra_tail_gram:
        grams.append(str_data[s_len - extra_tail_gram:] + "］")
    return grams


def calc_str_similarity(s1, s2, smoothing=0):
    gcs1 = collections.Counter()
    gcs1.update(get_str_grams(s1, 1))
    gcs1.update(get_str_grams(s1, 2))
    gcs2 = collections.Counter()
    gcs2.update(get_str_grams(s2, 1))
    gcs2.update(get_str_grams(s2, 2))
    return calc_bow_similarity(gcs1, gcs2, smoothing)


def calc_str_coverage(full_str, sub_str, smoothing=0):
    gcs1 = collections.Counter()
    gcs1.update(get_str_grams(full_str, 1))
    gcs1.update(get_str_grams(full_str, 2))
    gcs2 = collections.Counter()
    gcs2.update(get_str_grams(sub_str, 1))
    gcs2.update(get_str_grams(sub_str, 2))
    return calc_bow_coverage(gcs1, gcs2, smoothing)


def calc_edit_distance(s1, s2):
    # 算法：levenshtein2
    if len(s1) < len(s2):
        return calc_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # j+1 instead of j since previous and current are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def longest_common_sub_len(s1: str, s2: str) -> int:
    """计算最长公共子序列"""
    l1 = len(s1)
    l2 = len(s2)

    common_char_counts = np.zeros([l1 + 1, l2 + 1], dtype=np.int)
    for i in range(l1):
        for j in range(l2):
            if s1[i] == s2[j]:
                common_char_counts[i+1][j+1] = common_char_counts[i][j] + 1
            else:
                common_char_counts[i+1][j+1] = max(common_char_counts[i][j+1], common_char_counts[i+1][j])
    return common_char_counts[l1, l2]


def search_pattern(search_text, start_sign, end_sign, eof_end=False):
    found_patterns = []
    while True:
        si = search_text.find(start_sign)
        if si < 0:
            break
        ei = search_text.find(end_sign, si + 1)
        if ei <= si:
            if eof_end:
                ei = len(search_text)+1
            else:
                break
        found_patterns.append(search_text[si+len(start_sign):ei].strip())
        if ei > len(search_text):
            search_text = search_text[:si]
        else:
            search_text = search_text[:si] + " " + search_text[ei+len(end_sign):]

    return search_text, found_patterns
