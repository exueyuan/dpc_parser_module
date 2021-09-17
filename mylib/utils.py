# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

"""
辅助工具
"""

import os
import collections
import numpy as np
import datetime
import dateutil
import logging


TrainOrEvalCorpus = {"train", "dev", "eval", "valid", "validation"}


def cur_time_str(time_format="%m%d_%H%M"):
    return datetime.datetime.now().strftime(time_format)


def str_to_datetime(datetime_str):
    return dateutil.parser.parse(datetime_str)


def to_datetime(date_value, allow_none=False):
    if isinstance(date_value, str):
        date_value = dateutil.parser.parse(date_value)
    elif isinstance(date_value, float) or isinstance(date_value, int):
        date_value = datetime.datetime.fromtimestamp(date_value)
    if date_value is None and not allow_none or \
            date_value is not None and not isinstance(date_value, datetime.datetime):
        raise ValueError("参数应该为datetime类型：" + str(date_value))
    return date_value


def datetime_add(date_value, days=1, hours=0):
    return date_value + datetime.timedelta(days=days, hours=hours)


def split_data(data_list, dev_ratio=0.1, test_ratio=0.2, shuffle=True, random_seed=None, extra_data_list=None):
    data_size = len(data_list)
    if shuffle:
        if random_seed:
            np.random.seed(random_seed)  # 相同语料，相同种子，相同切分，方便模型加载后继续训练
        shuffle_indices = np.random.permutation(np.arange(data_size))
    else:
        shuffle_indices = range(data_size)

    data_parts = [[], [], []]
    extra_parts = [[], [], []]

    dev_count = int(data_size * dev_ratio)
    test_count = int(data_size * (dev_ratio + test_ratio))
    for i, index in enumerate(shuffle_indices):
        if i < dev_count:
            part_idx = 1
        elif i < test_count:
            part_idx = 2
        else:
            part_idx = 0
        data_parts[part_idx].append(data_list[index])
        if extra_data_list:
            extra_parts[part_idx].append(extra_data_list[index])

    del shuffle_indices
    if extra_data_list:
        return data_parts, extra_parts
    else:
        return data_parts


def split_data_by(data_list, data_kinds, dev_ratio=0.1, test_ratio=0.2, shuffle=True, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)  # 相同语料，相同种子，相同切分，方便模型加载后继续训练

    data_list_by_kinds = collections.defaultdict(list)
    for data, kind in zip(data_list, data_kinds):
        data_list_by_kinds[kind].append(data)

    data_parts = [[], [], []]
    for kind_data_list in data_list_by_kinds.values():
        data_size = len(kind_data_list)
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
        else:
            shuffle_indices = range(data_size)

        dev_count = int(data_size * dev_ratio)
        test_count = int(data_size * (dev_ratio + test_ratio))
        for i, index in enumerate(shuffle_indices):
            if i < dev_count:
                part_idx = 1
            elif i < test_count:
                part_idx = 2
            else:
                part_idx = 0
            data_parts[part_idx].append(kind_data_list[index])
        del shuffle_indices
        kind_data_list.clear()
    del data_list_by_kinds

    return data_parts


def find_work_root(existing_name, from_path=None):
    if not from_path:
        from_path = os.path.abspath(os.path.curdir)
    search_dir = from_path
    for parent_level in range(6):
        if os.path.exists(os.path.join(search_dir, existing_name)):
            return search_dir
        search_dir, _ = os.path.split(search_dir)
    raise RuntimeError("无法从【{}】找到{}".format(from_path, existing_name))


path_join = os.path.join


def check_path(config_path: str, path_is_file=False):
    if config_path.find('{}') >= 0:
        config_path = config_path.format(cur_time_str())
    config_path = os.path.abspath(config_path)
    if path_is_file:
        parent_dir, _ = os.path.split(config_path)
        os.makedirs(parent_dir, exist_ok=True)
    else:
        os.makedirs(config_path, exist_ok=True)
    return config_path


LOG_TIME_FORMAT = '%m-%d %H:%M:%S'
LOG_LINE_FORMAT = '%(asctime)s\t%(levelname)s\t%(message)s'
NullLogger = logging.Logger("NULL")
NullLogger.handlers = tuple()  # 不能添加Handler


def get_logger(level=logging.DEBUG, time_format=LOG_TIME_FORMAT, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_LINE_FORMAT, datefmt=time_format))
        logger.addHandler(console_handler)
    return logger


class Logger(logging.Logger):

    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def __init__(self, level=NOTSET, log_to_console=False):
        super(Logger, self).__init__("NoName", level)
        if log_to_console:
            self.log_to_console(level)

    def log_to_console(self, level=DEBUG, log_format=LOG_LINE_FORMAT, time_format=LOG_TIME_FORMAT):
        console_handler = None
        for handler in self.handlers:
            if isinstance(handler, logging.StreamHandler):
                console_handler = handler
                break
        if console_handler is None:
            console_handler = logging.StreamHandler()
            self.addHandler(console_handler)
        console_handler.setLevel(max(level, self.level))
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=time_format))

    def log_to_file(self, file_name, level=INFO, log_format=LOG_LINE_FORMAT, time_format=LOG_TIME_FORMAT,
                    append_mode=True):
        handler = logging.FileHandler(file_name, 'a' if append_mode else 'w', encoding='utf-8')
        handler.setFormatter(logging.Formatter(log_format, datefmt=time_format))
        handler.setLevel(max(level, self.level))
        self.addHandler(handler)


def get_file_logger(file_name, append_mode=True, level=logging.DEBUG, out_to_console=True):
    logger = Logger(level, out_to_console)
    logger.log_to_file(file_name, append_mode=append_mode)
    return logger


class TopTracker:

    def __init__(self, top_count=10):
        assert 0 < top_count < 100000
        self.top_count = top_count
        self._len = 0
        self._items = [None] * top_count
        self._weights = [None] * top_count

    def clear(self):
        self._len = 0

    def add(self, item, weight):
        if self._len == self.top_count and weight <= self._weights[-1]:
            return
        ins_pos = 0
        while ins_pos < self._len and self._weights[ins_pos] >= weight:
            ins_pos += 1
        move_pos = self._len
        if move_pos >= self.top_count:
            move_pos = self.top_count - 1
        else:
            self._len += 1
        while move_pos > ins_pos:
            self._items[move_pos] = self._items[move_pos - 1]
            self._weights[move_pos] = self._weights[move_pos - 1]
            move_pos -= 1
        self._items[ins_pos] = item
        self._weights[ins_pos] = weight

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if 0 <= index < self._len:
            return self._items[index], self._weights[index]
        raise IndexError(index)

    @property
    def top(self):
        return self._items[0] if self._len > 0 else None

    @property
    def items(self):
        if self._len == self.top_count:
            return self._items
        return self._items[:self._len]

    @property
    def weights(self):
        if self._len == self.top_count:
            return self._weights
        return self._weights[:self._len]

    def get_item(self, index):
        return self._items[index]

    def get_weight(self, index):
        return self._weights[index]


def print_counter(items_counter, min_count=3, file=None):
    item_count_list = sorted(items_counter.items(), key=lambda pair: pair[1], reverse=True)
    for item, count in item_count_list:
        if count < min_count:
            break
        print(f"{item}\t{count:.1f}", file=file)
