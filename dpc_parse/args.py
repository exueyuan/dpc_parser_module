# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

import mylib.utils as lu

bert_model_dir = 'pretrained_roberta'
work_dir = lu.find_work_root(bert_model_dir)
bert_model_dir = lu.path_join(work_dir, bert_model_dir + '/')

data_dir = lu.path_join(work_dir, 'data/dpc_parse/')
model_dir = lu.path_join(work_dir, 'models/dpc_/')

logger = lu.Logger()

arguments = {
    'gpu_ids': [0],  # 参与训练/预测的GPU编号，为空表示CPU
    'task': 'DependencyModel',

    'corpus': {
        'train_file': data_dir + 'stanford.train.txt',
        'eval_file': data_dir + 'stanford.dev.txt',
        'test_file': data_dir + 'stanford.test.txt',
        'min_word_freq': 1,  # 1代表在词向量中，2会刷掉2万多词
        'max_word_count': 80,
        'max_char_count': 160,
        'vector_model': r'F:\Python\stock_search\_data\dict\news_vec.s256.bin',
    },
    "model": {
        'vocab_size': -1,
        'dpc_kind_count': -1,
        'word_embed_size': 256,
        'bert_projection': 384,
        'encode_hidden': 384,
        'num_layers': 2,
        'output_hidden': 280,
        'input_dropout': 0.36,
        'encode_dropout': 0.33,
        'output_dropout': 0.33,
    },
    "bert": {
        'path': bert_model_dir,
        'vocab_file': bert_model_dir + 'vocab.txt',
    },
    'train': {
        'model_dir': model_dir,
        'bert_tuning': False,
        'learning_rate': 0.001,
        'train_epochs': 80,
        'batch_size': 128,
        'warmup_proportion': 0.1,
        'eval_steps': 0,
        'weight_decay': 0.01,
        'train_chart': model_dir + 'loss_acc.png',
    },
    'predict': {
        'batch_size': 256,
    }
}
