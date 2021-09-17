# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

import os
import time
import collections
import torch
from torch import nn
from transformers import BertModel, BertConfig

from mylib import txt_lib as lt
from mylib import utils as lu

Epsilon = 1e-6


class ModelBase(nn.Module):

    def __init__(self, config: dict = None, **kwargs):
        super(ModelBase, self).__init__()
        self.config = collections.OrderedDict()
        copy_dict(config, self.config)
        copy_dict(kwargs, self.config)
        self.build_()

    def configure_(self, config: dict = None, **kwargs):
        copy_dict(config, self.config)
        copy_dict(kwargs, self.config)

    def get_config_(self, key, default_value=None):
        return self.config.get(key, default_value)

    def get_config(self, group, key, default_value=None):
        group = self.config.get(group, self.EmptyMap)
        return group.get(key, default_value)

    def build_(self):
        pass

    EmptyMap = {}
    ConfigFile = "model.json"
    ModelFileName = "torch_model"
    BestModelTag = '.best'
    FinalModelTag = '.final'

    def save_model(self, model_path, tag=BestModelTag, save_config=True):
        if save_config:
            lt.save_json(self.config, os.path.join(model_path, self.ConfigFile), indent=4)
        model_file = os.path.join(model_path, self.ModelFileName + tag)
        torch.save(self.state_dict(), model_file)

    @classmethod
    def exist_model(cls, model_path, tag=BestModelTag):
        model_file = os.path.join(model_path, cls.ModelFileName + tag)
        return os.path.exists(model_file)

    @classmethod
    def load_model(cls, model_path, device=None, tag=BestModelTag, **kwargs):
        model_file = os.path.join(model_path, cls.ModelFileName + tag)
        if device is None:
            model_states = torch.load(model_file)
        else:
            model_states = torch.load(model_file, map_location=torch.device(device))

        config = None
        if os.path.exists(os.path.join(model_path, cls.ConfigFile)):
            config = lt.load_json(os.path.join(model_path, cls.ConfigFile))

        model = cls(config, **kwargs)
        model.load_state_dict(model_states)
        return model


class BertModelBase(ModelBase):

    def build_(self):
        bert_path = self.get_config_('bert_path')
        if bert_path:
            self.bert = BertModel.from_pretrained(bert_path)
        else:
            self.bert = BertModel(self.get_config_('bert_config'))
        self.bert_tuning = self.get_config('train', 'bert_tuning', False)

    # noinspection PyUnresolvedReferences
    @property
    def bert_tuning(self):
        for p in self.bert.parameters():
            return p.requires_grad
        return False

    # noinspection PyUnresolvedReferences
    @bert_tuning.setter
    def bert_tuning(self, trainable):
        for p in self.bert.parameters():
            p.requires_grad = trainable

    BertConfigFile = "config.json"

    @classmethod
    def load_model(cls, model_path, device=None, tag=ModelBase.BestModelTag, **kwargs):
        if kwargs and "bert_config" in kwargs:
            bert_config = kwargs['bert_config']
        else:
            if kwargs and "bert_path" in kwargs:
                bert_config_file = kwargs['bert_path']
            else:
                bert_config_file = model_path
            bert_config = BertConfig.from_pretrained(bert_config_file)
        return super(BertModelBase, cls).load_model(model_path, device, tag, bert_config=bert_config, **kwargs)

    # noinspection PyUnresolvedReferences
    def save_model(self, model_path, tag=ModelBase.BestModelTag, save_config=True):
        if save_config:
            self.config.pop('bert_path', None)
            self.config.pop('bert_config', None)
            self.bert.config.to_json_file(os.path.join(model_path, self.BertConfigFile))
        return super(BertModelBase, self).save_model(model_path, tag, save_config)


class DoubleLinear(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0,
                 activator="leakyrelu", bias=True):
        super(DoubleLinear, self).__init__()

        self.hidden_linear = nn.Linear(in_features, hidden_features, bias)
        if activator.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        elif activator.lower() == "relu":
            self.activation = nn.ReLU()
        elif activator.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None
        self.output_liner = nn.Linear(hidden_features, out_features, bias)
        if 0.0 < dropout < 1.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.hidden_linear.weight)
        # 其他已初始化

    def forward(self, x):
        x = self.hidden_linear(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.output_liner(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class AttentionWeightedPool(nn.Module):

    PoolMethods = {"sum", "min", "max", "max_p", "detail"}

    def __init__(self, hidden_units, pool_method="sum"):
        super(AttentionWeightedPool, self).__init__()
        assert pool_method in self.PoolMethods
        self.pool_method = pool_method
        self.batch_offsets = None
        self.attention = nn.Linear(hidden_units, 1, bias=False)

    def forward(self, input_states, attention_mask):
        if self.batch_offsets is None:
            batch_size = input_states.size(0)
            self.batch_offsets = torch.tensor(range(batch_size), dtype=torch.long,
                                              device=input_states.device, requires_grad=False)
            self.batch_offsets *= batch_size

        # input_states (batch_size, sequence_len, hidden_units)
        pos_weights = self.attention(input_states).squeeze(-1)  # (batch_size, sequence_len)
        pos_weights = torch.exp(pos_weights - torch.max(pos_weights, dim=-1, keepdim=True)[0])
        
        if attention_mask is not None:
            pos_weights = pos_weights * attention_mask.float()
        pos_weights = pos_weights / (torch.sum(pos_weights, dim=1, keepdim=True) + Epsilon)
        weighted_states = input_states * pos_weights.unsqueeze(-1)
        if self.pool_method == "max":
            return torch.max(weighted_states, dim=1, keepdim=False)[0]
        if self.pool_method == "sum":
            return torch.sum(weighted_states, dim=1, keepdim=False)
        if self.pool_method == "min":
            return torch.min(weighted_states, dim=1, keepdim=False)[0]
        if self.pool_method == "max_p":
            pos_max = pos_weights.argmax(1, False) + self.batch_offsets
            # max_state = weighted_states.select(1, pos_max)
            # print(max_state.shape)
            weighted_states = weighted_states.flatten(0, 1)
            max_state = weighted_states.index_select(0, pos_max)
            return max_state
        return weighted_states


def warmup_linear(x, warmup=0.06):
    if x < warmup:
        return x/warmup
    return (1.0 - x) / (1.000001 - warmup)


class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `1 - warmup` steps.
    """

    # noinspection PyTypeChecker
    def __init__(self, optimizer, total_step_count, warmup_proportion=0.06):
        self.total_step_count = total_step_count
        self.warmup_proportion = warmup_proportion
        super(WarmupLinearSchedule, self).__init__(optimizer, self.calc_warmup)

    def calc_warmup(self, step):
        return warmup_linear(step / self.total_step_count, self.warmup_proportion)

    # noinspection PyUnresolvedReferences
    @property
    def last_lr(self):
        last_lrs = self.get_last_lr()
        return last_lrs[0]


# noinspection PyUnresolvedReferences
class BertAdam(torch.optim.Optimizer):

    """Implements BERT version of Adam algorithm with momentum, weight decay fix and max_grad_norm."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, max_grad_norm=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                next_m, next_v = state['next_m'], state['next_v']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['eps'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                lr_scheduled = group['lr']
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # lr_scheduled = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_(-lr_scheduled * update)
                # state['step'] += 1

        return loss


class TrainAider:

    DefaultGPUIds = tuple([0])
    NoDecayNames = ['bias', 'LayerNorm']

    def __init__(self, model: ModelBase, model_save_dir, fp16: bool = False, sleep_between_step=0.0):
        self.model: ModelBase = model
        self.model_t = model
        self.model_save_dir = model_save_dir
        self.logger = lu.NullLogger
        self.his_data = TrainHistoryData()

        self.fp16 = fp16
        self.device = 'cpu'
        self.n_gpu = 0
        self.global_step = 0
        self.total_steps = -1
        self.warmup_schedule = None
        self.gradient_accumulation_steps = 1
        self.sleep_between_step = sleep_between_step

    def calc_total_steps(self, train_data_size: int, train_epochs: int, batch_size: int,
                         gradient_accumulation_steps: int = 1):
        steps_per_epoch = int(train_data_size + batch_size - 1) // batch_size
        if gradient_accumulation_steps > 1:
            steps_per_epoch //= gradient_accumulation_steps

        self.total_steps = steps_per_epoch * train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def build_optimizer(self, learning_rate: float, weight_decay: float = 0.01, warmup_proportion: float = 0.06,
                        optimizer_class=BertAdam):
        self.learning_rate = learning_rate
 
        train_params = list(self.model.named_parameters())
        train_params = [(n, p) for n, p in train_params if p.requires_grad]
        parameter_groups = self.group_parameters(train_params, weight_decay)
        self.optimizer = self.create_optimizer_(parameter_groups, optimizer_class)
        self.warmup_schedule = WarmupLinearSchedule(self.optimizer, self.total_steps, warmup_proportion)

    def group_parameters(self, train_params, weight_decay, no_decay_names=NoDecayNames):
        train_params = [(n, p) for n, p in train_params if "pooler" not in n]
        # for name, param in train_params:
        #     self.logger.info("训练参数{}\t{}".format(name, param.shape))
        return [
            {'params': [p for n, p in train_params if not any(nd in n for nd in no_decay_names)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in train_params if any(nd in n for nd in no_decay_names)],
             'weight_decay': 0.0}]

    def create_optimizer_(self, parameter_groups, optimizer_class):
        if self.fp16:  # GPU半精度fp16
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(parameter_groups,
                                  lr=self.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)  # static_loss_scale=loss_scale
        else:
            optimizer = optimizer_class(parameter_groups, lr=self.learning_rate)
        return optimizer

    def to_device(self, gpu_ids=DefaultGPUIds, distributed_rank=-1):
        # ---------------------模型初始化----------------------
        if self.fp16:
            self.model.half()
        # ------------------判断CUDA模式----------------------
        if distributed_rank == -1:
            self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_ids else "cpu")
            self.n_gpu = 0 if self.device == 'cpu' else len(gpu_ids)
        else:
            torch.cuda.set_device(distributed_rank)
            self.device = torch.device("cuda", distributed_rank)
            self.n_gpu = 1

        self.model.to(self.device)
        if distributed_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel
            except ImportError:
                raise ImportError("Please install apex to use distributed and fp16 training.")
            self.model_t = DistributedDataParallel(self.model)
        elif self.n_gpu > 1:
            self.model_t = nn.DataParallel(self.model, device_ids=gpu_ids)
        self.logger.info("模型适配完成：{} {} {}".format(self.device, self.n_gpu, self.model_t))

    def do_train_(self, step, batch_data, accumulate_only, log_steps, show_steps):
        raise NotImplementedError()

    def do_eval_(self, eval_data_loader):
        # 必须返回所有评估分{Loss: xx, Acc: xx...}，以及模型综合评分（分值越大，模型越优）
        raise NotImplementedError()

    def train(self, train_data_iter, eval_data_iter, train_epochs: int,
              log_steps: int = 100, show_steps: int = 1, eval_steps: int = 0,
              train_chart_file='train_chart.png'):
        epoch = 0
        while self.global_step < self.total_steps and epoch < train_epochs:
            epoch += 1
            self.logger.info('Epoch {}'.format(epoch))
            for step, batch_data in enumerate(train_data_iter):
                if self.sleep_between_step > 0:
                    time.sleep(self.sleep_between_step)
                if self.n_gpu == 1:  # multi-gpu does scattering it-self
                    batch_data = tuple(t.to(self.device) for t in batch_data)

                self.model_t.train()
                global_step0 = self.global_step
                accumulate_only = (step + 1) % self.gradient_accumulation_steps != 0
                self.do_train_(step + 1, batch_data, accumulate_only, log_steps, show_steps)
                if not accumulate_only:
                    if self.global_step == global_step0:
                        self.global_step += 1
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    if eval_steps > 0 and self.global_step % eval_steps == 0:
                        self._eval_and_save(eval_data_iter, train_chart_file)
                    self.warmup_schedule.step()
            if eval_steps <= 0:
                self._eval_and_save(eval_data_iter, train_chart_file)
        if eval_steps > 0 and self.global_step % eval_steps >= eval_steps // 2:
            self._eval_and_save(eval_data_iter, train_chart_file)

    EvalAccName = 'Acc'

    def _eval_and_save(self, eval_data_iter, train_chart_file):
        eval_scores, eval_score = self.do_eval_(eval_data_iter)
        self.logger.info("EVAL: " + ", ".join('{} {:.4f}'.format(k, v) for k, v in eval_scores.items()))
        # 保存模型
        is_best = self.his_data.trace_eval(eval_scores['Loss'], eval_scores.get(self.EvalAccName, 0.0), eval_score)
        if is_best:
            self.model.save_model(self.model_save_dir, save_config=self.his_data.eval_step_count == 1)
            self.logger.info('Best Model saved to {}\n'.format(self.model_save_dir))
        else:
            self.model.save_model(self.model_save_dir, '.final', save_config=False)
            self.logger.info('Final Model saved to {}, Previous best {:.4f}\n'.format(
                self.model_save_dir, self.his_data.best_score))
        if train_chart_file:
            self.his_data.plot(os.path.join(self.model_save_dir, train_chart_file))


class TrainHistoryData:

    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.train_score = []
        self.eval_loss = []
        self.eval_acc = []
        self.eval_score = []
        self.best_index = -1
        self.loss_sum, self.acc_sum, self.score_sum = 0.0, 0.0, 0.0
        self.loss_avg, self.acc_avg, self.score_avg = 0.0, 0.0, 0.0
        self.terminate_after_n_descend_eval = 10

    @property
    def best_score(self):
        return self.eval_score[self.best_index] if self.best_index >= 0 else 0.0

    @property
    def steps_avg(self):
        return self.loss_avg, self.acc_avg, self.score_avg

    def trace_step(self, loss, acc, score):
        self.loss_sum += loss
        self.acc_sum += acc
        self.score_sum += score

    def calc_steps_avg(self, step_count, reset_step=False):
        self.loss_avg = self.loss_sum / step_count
        self.acc_avg = self.acc_sum / step_count
        self.score_avg = self.score_sum / step_count
        if reset_step:
            self.loss_sum, self.acc_sum, self.score_sum = 0.0, 0.0, 0.0
        return self.loss_avg, self.acc_avg, self.score_avg

    @property
    def eval_step_count(self):
        return len(self.eval_loss)

    def trace_eval(self, eval_loss, eval_acc, eval_score):
        self.train_loss.append(self.loss_avg)
        self.train_acc.append(self.acc_avg)
        self.train_score.append(self.score_avg)
        self.eval_loss.append(eval_loss)
        self.eval_acc.append(eval_acc)
        self.eval_score.append(eval_score)
        if eval_score > self.best_score:
            self.best_index = len(self.eval_score) - 1
            return True
        return False

    def plot(self, file_name):
        if self.best_index < 0:
            return

        import matplotlib.pyplot as plt
        # 无图形界面需要加，否则plt报错
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(16, 10))
        epochs = range(1, len(self.train_loss) + 1)

        fig.add_subplot(3, 1, 1)
        plt.title('losses')
        plt.plot(epochs, self.train_loss)
        plt.plot(epochs, self.eval_loss)
        plt.legend(['train_loss', 'eval_loss'])

        fig.add_subplot(3, 1, 2)
        plt.title('accuracy')
        plt.plot(epochs, self.train_acc)
        plt.plot(epochs, self.eval_acc)
        val, eps = max((val, -eps) for eps, val in zip(epochs, self.train_acc))
        plt.plot(-eps, val, marker='^', color='b')
        plt.text(-eps, val, f"{val:.4f}", ha='center', va='bottom', color='b')
        val, eps = max((val, -eps) for eps, val in zip(epochs, self.eval_acc))
        plt.plot(-eps, val, marker='v', color='r')
        plt.text(-eps, val, f"{val:.4f}", ha='center', va='top', color='r')
        plt.legend(['train_acc', 'eval_acc'])

        fig.add_subplot(3, 1, 3)
        plt.title('score')
        plt.plot(epochs, self.train_score)
        plt.plot(epochs, self.eval_score)
        val, eps = max((val, -eps) for eps, val in zip(epochs, self.train_score))
        plt.plot(-eps, val, marker='^', color='b')
        plt.text(-eps, val, f"{val:.4f}", ha='center', va='bottom', color='b')
        val, eps = max((val, -eps) for eps, val in zip(epochs, self.eval_score))
        plt.plot(-eps, val, marker='v', color='r')
        plt.text(-eps, val, f"{val:.4f}", ha='center', va='top', color='r')
        plt.legend(['train_score', 'eval_score'])

        plt.savefig(file_name)
        if len(self.train_loss) + eps >= self.terminate_after_n_descend_eval > 0:
            raise InterruptedError(f"{self.terminate_after_n_descend_eval}轮评估得分无提升，提前终止训练")


# noinspection PyUnresolvedReferences
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def copy_dict(from_dict, to_dict, overwrite=True):
    if from_dict is None:
        return
    for key, nv in from_dict.items():
        if isinstance(nv, dict):
            ov = to_dict.get(key, None)
            if ov is dict:
                copy_dict(ov, nv, overwrite)
                continue
        if overwrite or key not in to_dict:
            to_dict[key] = nv


def convert_tf_bert_to_torch(tf_checkpoint_path, bert_model_path):
    from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert
    print("Building PyTorch model from configuration...")
    config = BertConfig.from_json_file(os.path.join(tf_checkpoint_path, 'bert_config.json'))

    model = BertForPreTraining(config)
    print("Load weights from tensorflow model: {}".format(tf_checkpoint_path))
    load_tf_weights_in_bert(model, os.path.join(tf_checkpoint_path, 'bert_model.ckpt'))

    # Save pytorch-model
    print("Save PyTorch model to {}".format(bert_model_path))
    # noinspection PyUnresolvedReferences
    torch.save(model.state_dict(), os.path.join(bert_model_path, 'pytorch_model.bin'))
