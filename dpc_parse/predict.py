# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

import os
import torch
from mylib import txt_lib as lt
from mylib import word_embed as lw

from ner.data import parse_tagged_words
from dpc_parse.api import DpcKinds
from dpc_parse.loader import create_embed_agent
from dpc_parse.model import DependencyModel, decode_dependency


class DependencyParser:

    def __init__(self, model_path, gpu_ids, batch_size, num_threads=6):
        torch.set_num_threads(num_threads)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_ids else "cpu")
        self.model = DependencyModel.load_model(model_path, self.device)
        self.model.to(self.device)
        self.model.eval()
        self.embed_tool = create_embed_agent(self.model.config, False, model_path)

    def parse(self, tagged_article_words):
        tagged_sentence_list = lw.split_sentences(tagged_article_words,
                                                  self.embed_tool.max_word_count - 1,
                                                  self.embed_tool.max_char_count - 1)
        parsed_sentence_list = []
        start, count = 0, len(tagged_sentence_list)
        while start < count:
            tagged_sentence_batch = tagged_sentence_list[start: start + self.batch_size]
            batch_data = self.embed_tool.embed_fast(tagged_sentence_batch)
            batch_data = [torch.tensor(data, dtype=torch.long, device=self.device)
                          for data in batch_data]
            arc_scores, kind_scores, easier_mask = self.model(*batch_data)
            # easier_mask[:, 0] = 0
            predict_arcs, predict_kinds = decode_dependency(arc_scores, kind_scores, None)  # no easier_mask
            predict_arcs = predict_arcs.tolist()
            predict_kinds = predict_kinds.tolist()
            for tagged_sentence, dpc_arcs, dpc_kinds in zip(tagged_sentence_batch, predict_arcs, predict_kinds):
                sentence_words = []
                for idx, (word, tag) in enumerate(tagged_sentence):
                    idx += 1  # SOS
                    sentence_words.append((word, tag, dpc_arcs[idx], DpcKinds.get(dpc_kinds[idx]).Title))
                parsed_sentence_list.append(sentence_words)

            start += self.batch_size
        return parsed_sentence_list


def dependency_parse_file(file_name, out_file_name, concat_by='\u00a0'):
    done_count = 0
    with lt.open_file(out_file_name, 'w') as writer:
        for article in lt.open_file(file_name):
            article = article.strip()
            if not article:
                writer.write_line(article)
                continue
            done_count += 1
            tagged_article_words = parse_tagged_words(article)
            parsed_sentence_list = dpc_parser.parse(tagged_article_words)
            for parsed_sentence in parsed_sentence_list:
                parts = []
                for idx, (word, tag, arc, kind) in enumerate(parsed_sentence):
                    part = "{}│{}〖{}-{}-{}〗".format(word, tag, idx + 1, kind, arc)
                    parts.append(part.replace(concat_by, ''))
                writer.write_line(concat_by.join(parts) + '〗')
            if done_count % 300 == 0:
                logger.info("\t{}".format(done_count))
        logger.info("\t{}".format(done_count))


def parse_data_files(data_dir):
    for name in os.listdir(data_dir):
        file_name = os.path.join(data_dir, name)
        if os.path.isdir(file_name) or not name.endswith('.tag.txt') or name.endswith('dp.txt'):
            continue
        out_file_name = file_name.replace('.tag.', '.dp.')
        if not os.path.exists(out_file_name):
            logger.info("分析文件：" + file_name)
            dependency_parse_file(file_name, out_file_name)


if __name__ == '__main__':
    path = r"F:\Python\stock_search\_data\models\dpc_l2_nt"
    dpc_parser = DependencyParser(path, [0], 256)

    from dpc_parse.args import logger
    logger.log_to_console()
    parse_data_files(r"G:\Data\ner")
