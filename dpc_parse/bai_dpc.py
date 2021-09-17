# -*- coding: utf-8 -*-

import os
import json

from mylib import txt_lib as lt
from mylib.baidu_api import BaiduDependency


def dependency_parse_file(parser, file_name, out_file_name):
    with lt.open_file(out_file_name, 'w') as writer:
        paragraphs = lt.load_lines(file_name)
        for paragraph in paragraphs:
            sentences = lt.split_sentences(paragraph, max_sentence_len=parser.MaxSentenceLength)
            for sentence0 in sentences:
                sentence = sentence0.encode('gbk', 'ignore').decode('gbk', 'ignore')
                parser.logger.info("\t语句：" + sentence)
                if sentence != sentence0:
                    parser.logger.warning("\t原句：" + sentence)
                if sentence:
                    makeups = parser.process(sentence)
                    output = {"sentence": sentence, "dp_items": makeups}
                    writer.write_line(json.dumps(output, ensure_ascii=False))


def dependency_parse_files(parser, path):
    for name in os.listdir(path):
        file_name = os.path.join(path, name)
        if os.path.isdir(file_name) or not name.endswith('.txt') or name.endswith('.dp.txt'):
            continue
        parser.logger.info("文件：" + file_name)
        fn, ext = os.path.splitext(file_name)
        out_file_name = f"{fn}.dp{ext}"
        if not os.path.exists(out_file_name):
            dependency_parse_file(parser, file_name, out_file_name)


if __name__ == "__main__":
    d_parser = BaiduDependency()
    d_parser.logger.setLevel(20)
    dependency_parse_files(d_parser, r"E:\Corpus\金融图谱\yanbao_txt")
