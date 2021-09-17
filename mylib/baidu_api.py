# -*- coding: utf-8 -*-

import time
from aip import AipNlp  # pip install baidu-aip

import mylib.txt_lib as lt
import mylib.utils as lu


class BaiduClient:
    """
    百度AI在线Api封装
    """
    APP_ID = '10413677'
    API_KEY = 'Dci3SHexD5RaoCLI29nfX42I'
    SECRET_KEY = 'H8RVtSGgFkCUDCeGGafgk8KrZWG7AS8T'
    logger = lu.get_logger()

    def __init__(self):
        self.client = AipNlp(self.APP_ID, self.API_KEY, self.SECRET_KEY)

    def safe_remote_call(self, sentence):
        """
        避免频繁调用导致的失败
        :param sentence:
        :return:
        """
        last_error = "lexer调用失败"
        for try_index in range(6):  # 存在失败可能，多尝试几次
            reply_data = self.remote_call(sentence)
            error_msg = reply_data.get("error_msg", "")
            if error_msg.find("request limit") >= 0:
                pass
            elif len(error_msg) > 0:
                self.logger.error(str(reply_data))
                raise RuntimeError(error_msg)
            else:
                reply_items = reply_data.get("items", None)
                if reply_items and len(reply_items) >= 0:
                    return reply_items
            last_error = str(reply_data)
            time.sleep(0.03 + try_index * 0.06)  # 避免频繁访问限制
        self.logger.error(last_error)
        return None

    def remote_call(self, sentence):
        return None

    def process(self, sentence):
        return self.safe_remote_call(sentence)

    def process_file(self, input_file, output_file):
        self.logger.info("加载数据...")
        with lt.open_file(output_file, 'w') as f:
            for line in lt.load_lines(input_file, True):
                if len(line) <= 0:
                    print(line, file=output_file)
                else:
                    result = self.process(line)
                    print(result, file=f)


class BaiduNer(BaiduClient):
    """
    专名识别
    """
    TagsMap = {'org': 'nt', 'per': 'nr', 'loc': 'ns'}

    def remote_call(self, sentence):
        return self.client.lexer(sentence)

    def parse_ner_from_items(self, items):
        found_entities = []
        if items is None:
            return found_entities
        for token in items:
            tag = token.get("ne", "").lower()
            tag = self.TagsMap.get(tag, "")
            name = token.get("item", None)
            if name:
                found_entities.append((name, tag))

        return found_entities

    def process(self, sentence):
        items = self.safe_remote_call(sentence)
        return self.parse_ner_from_items(items)


class BaiduSentiment(BaiduClient):
    """
    情感分析
    """
    def remote_call(self, sentence):
        return self.client.sentimentClassify(sentence)

    def process(self, sentence):
        items = self.safe_remote_call(sentence)
        reply = items[0]
        sentiment = reply.get("sentiment", 1) - 1  # 表示情感极性分类结果
        confidence = reply.get("confidence", 0.0)  # 表示分类的置信度
        positive_prob = reply.get("positive_prob", 0.0)  # 表示属于积极类别的概率
        negative_prob = reply.get("negative_prob", 0.0)  # 表示属于消极类别的概率
        print("{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            sentence, sentiment, confidence, positive_prob, negative_prob))

        return sentiment


class BaiduDependency(BaiduClient):
    """
    依存句法分析
    """
    MaxSentenceLength = 128  # gbk 256

    def remote_call(self, sentence):
        return self.client.depParser(sentence)

    def process(self, sentence):
        items = self.safe_remote_call(sentence)
        return items


if __name__ == "__main__":
    # evaluator = BaiduSentiment()
    # evaluator.process_file("G:/XFJ/情感分析.txt")

    evaluator = BaiduDependency()
    evaluator.process_file(r'F:\Python\stock_search\_data\features\stock-products-phrase.txt', 'g:/stock_dependency.txt')

