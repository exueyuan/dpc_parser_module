# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

import time
import torch

import mylib.utils as lu
import mylib.torch_lib as tl
from dpc_parse.args import arguments, logger
from dpc_parse.corpus import load_corpus
from dpc_parse.api import DpcKinds
from dpc_parse.loader import embed_pack_corpus, create_embed_agent
from dpc_parse.model import DependencyModel, DependencyMetric, decode_dependency


tl.set_seeds(arguments['train'].get('seed', 666))
model_save_dir = arguments['train']['model_dir']
logger.log_to_console()
logger.log_to_file(lu.check_path(lu.path_join(model_save_dir, '{}.log'), True), level=lu.logging.INFO)


class TrainAssistant(tl.TrainAider):

    DefaultGPUIds = tuple([0])

    def __init__(self, model):
        super(TrainAssistant, self).__init__(model, model_save_dir, False, sleep_between_step=0.1)
        self.logger = logger

    def do_train_(self, step, batch_data, accumulate_only, log_steps, show_steps):
        word_ids_tensor, char_ids_tensor, word_lens_tensor, dpc_arcs_tensor, dpc_kinds_tensor = batch_data
        arc_scores, kind_scores, mask = self.model_t(word_ids_tensor, char_ids_tensor, word_lens_tensor)

        mask[:, 0] = 0
        train_loss = self.model.get_loss(arc_scores, kind_scores, dpc_arcs_tensor, dpc_kinds_tensor, mask)
        if self.n_gpu > 1:
            train_loss = train_loss.mean()  # 多GPU时需要
        step_loss = train_loss.item()
        if self.gradient_accumulation_steps > 1:
            train_loss = train_loss / self.gradient_accumulation_steps
        if self.fp16:
            self.optimizer.backward(train_loss)
        else:
            train_loss.backward()
        if accumulate_only:  # 只计算回传误差并累积，不更新
            return

        self.global_step += 1
        self.optimizer.step()
        self.optimizer.zero_grad()

        metrics = DependencyMetric()
        predict_arcs, predict_kinds = decode_dependency(arc_scores, kind_scores, None)  # no easier
        metrics(predict_arcs, predict_kinds, dpc_arcs_tensor, dpc_kinds_tensor, mask)
        self.his_data.trace_step(step_loss, metrics.uas, metrics.las)

        if show_steps > 0 and self.global_step % show_steps == 0:
            print("\tStep {}, Loss {:.4f}, {}".format(step, step_loss, metrics))

        if log_steps > 0 and self.global_step % log_steps == 0:
            steps_avg = self.his_data.calc_steps_avg(log_steps, True)
            done_ratio = self.global_step / self.total_steps
            logger.info("  Iter {}: {:.1%} done, LR {:.6f}, Avg Loss {:.4f}, Avg UAS {:.4f}, Avg LAS {:.4f}".format(
                self.global_step, done_ratio, self.warmup_schedule.last_lr, *steps_avg))

    EvalAccName = 'UAS'

    def do_eval_(self, eval_data_loader):
        self.model_t.eval()
        metrics = DependencyMetric()
        eval_loss, step_count = 0.0, 0
        with torch.no_grad():
            for step, batch_data in enumerate(eval_data_loader):
                if self.n_gpu == 1:  # multi-gpu does scattering it-self
                    batch_data = tuple(t.to(self.device) for t in batch_data)
                word_ids_tensor, char_ids_tensor, word_lens_tensor, dpc_arcs_tensor, dpc_kinds_tensor = batch_data
                arc_scores, kind_scores, mask = self.model_t(word_ids_tensor, char_ids_tensor, word_lens_tensor)

                mask[:, 0] = 0
                step_loss = self.model.get_loss(arc_scores, kind_scores, dpc_arcs_tensor, dpc_kinds_tensor, mask)

                if self.n_gpu > 1:
                    step_loss = step_loss.mean()
                eval_loss += step_loss.item()
                step_count += 1
                predict_arcs, predict_kinds = decode_dependency(arc_scores, kind_scores, None)  # no easier
                metrics(predict_arcs, predict_kinds, dpc_arcs_tensor, dpc_kinds_tensor, mask)
                time.sleep(self.sleep_between_step / 2)
        eval_scores = {
            'Loss': eval_loss / step_count,
            'UAS': metrics.uas,
            'LAS': metrics.las,
        }
        return eval_scores, eval_scores['LAS']


def train_model():
    # 加载语料
    train_corpus = load_corpus("train")
    eval_corpus = load_corpus("eval")
    # 语料编码向量化
    model_exists = not arguments['train'].get('zero_start', False) and DependencyModel.exist_model(model_save_dir)
    if model_exists:
        embed_tool = create_embed_agent(False)
        embed_tool.load(model_save_dir)
    else:
        embed_tool = create_embed_agent(True)
        embed_tool.stat_vocab(train_corpus, arguments['corpus'].get('min_word_freq', 0))
        embed_tool.save(model_save_dir)
        arguments['model']['vocab_size'] = embed_tool.word_embed_tool.vocab_size
        arguments['model']['dpc_kind_count'] = DpcKinds.Count
    train_data_iter = embed_pack_corpus(train_corpus, embed_tool, True, False)
    eval_data_iter = embed_pack_corpus(eval_corpus, embed_tool, False, True)

    # 创建或者加载模型
    if model_exists:
        logger.info('加载最佳模型' + model_save_dir)
        model = DependencyModel.load_model(model_save_dir)
    else:
        bert_path = arguments['bert']['path']
        logger.info('加载预训练Bert模型' + bert_path)
        model = DependencyModel(arguments, bert_path=bert_path,
                                embedding_matrix=embed_tool.word_embed_tool.embedding_table)

    train_epochs = arguments['train']['train_epochs']
    assistant = TrainAssistant(model)
    assistant.calc_total_steps(len(train_data_iter.dataset),
                               train_epochs,
                               train_data_iter.batch_size,
                               arguments['train'].get('gradient_accumulation_steps', 1))
    logger.info("总训练步数{}".format(assistant.total_steps))

    from torch.optim.adamw import AdamW
    assistant.build_optimizer(arguments['train']['learning_rate'],
                              arguments['train']['weight_decay'],
                              arguments['train']['warmup_proportion'],
                              AdamW)
    assistant.to_device(arguments['gpu_ids'])
    assistant.train(train_data_iter, eval_data_iter, train_epochs,
                    show_steps=10, eval_steps=arguments['train']['eval_steps'])


if __name__ == '__main__':
    try:
        train_model()
    finally:
        if arguments['gpu_ids']:  # 释放显存
            torch.cuda.empty_cache()
