# -*- coding: utf-8 -*-
# Authors: 李坤奇 <likunqi@sina.com>

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import mylib.torch_lib as tl
from dpc_parse.layers import MLP, Biaffine, SharedDropout


class DependencyModel(tl.BertModelBase):

    ModelFileName = "dependency_bert"

    def __init__(self, config: dict = None, **kwargs):
        self.embedding_matrix = kwargs.pop("embedding_matrix", None)
        super(DependencyModel, self).__init__(config, **kwargs)

    def build_(self):
        super(DependencyModel, self).build_()
        self.pad_index = 0
        embedding_size = self.get_config('model', 'word_embed_size')
        # 输入层
        self.word_embedding = nn.Embedding(num_embeddings=self.get_config('model', 'vocab_size'),
                                           embedding_dim=embedding_size,
                                           _weight=self.embedding_matrix)
        bert_projection = self.get_config('model', 'bert_projection')
        if bert_projection:
            self.bert_embed_size = self.bert.config.hidden_size
            self.bert_projection = None
        else:
            self.bert_projection = nn.Linear(self.bert.config.hidden_size, bert_projection)
            self.bert_embed_size = bert_projection
        self.input_dropout = nn.Dropout(p=self.get_config('model', 'input_dropout', 0.3))

        encode_hidden = self.get_config('model', 'encode_hidden', 192)
        self.encoder = nn.LSTM(
            input_size=embedding_size + self.bert_embed_size,
            hidden_size=encode_hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=self.get_config('model', 'num_layers', 1),
            dropout=0.2
        )
        # SharedDropout 会将同一样本所有word向量的相同维清零，影响Biaffine？
        self.encode_dropout = SharedDropout(p=self.get_config('model', 'encode_dropout', 0.3))
        encoding_size = 2 * encode_hidden

        # 输出层
        output_hidden = self.get_config('model', 'output_hidden', 96)
        dpc_kind_count = self.get_config('model', 'dpc_kind_count')
        output_dropout = self.get_config('model', 'output_dropout', 0.3)
        self.mlp_arc_h = MLP(n_in=encoding_size,
                             n_hidden=output_hidden,
                             dropout=output_dropout)
        self.mlp_arc_d = MLP(n_in=encoding_size,
                             n_hidden=output_hidden,
                             dropout=output_dropout)
        self.mlp_kind_h = MLP(n_in=encoding_size,
                              n_hidden=output_hidden,
                              dropout=output_dropout)
        self.mlp_kind_d = MLP(n_in=encoding_size,
                              n_hidden=output_hidden,
                              dropout=output_dropout)
        # Biaffine 解码层
        self.arc_attn = Biaffine(n_in=output_hidden,
                                 bias_x=True,
                                 bias_y=False)
        self.kind_attn = Biaffine(n_in=output_hidden,
                                  n_out=dpc_kind_count,
                                  bias_x=True,
                                  bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words_ids_list, char_ids_list, word_lens_list):
        mask = words_ids_list.ne(self.pad_index)

        word_embedded = self.word_embedding(words_ids_list)
        bert_output_seq = self.bert_encode_to_words(char_ids_list, word_lens_list)
        embedded_data = torch.cat((bert_output_seq, word_embedded), dim=-1)

        encoded_raw, _ = self.encoder(self.input_dropout(embedded_data))
        encoded_masked = encoded_raw * mask.to(dtype=torch.long).unsqueeze(-1)
        encoded_dropout = self.encode_dropout(encoded_masked)

        # apply MLPs to the encoded states
        arc_h = self.mlp_arc_h(encoded_dropout)
        arc_d = self.mlp_arc_d(encoded_dropout)
        kind_h = self.mlp_kind_h(encoded_dropout)
        kind_d = self.mlp_kind_d(encoded_dropout)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        arc_scores = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, num_dependencies]
        kind_scores = self.kind_attn(kind_d, kind_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        arc_scores.masked_fill_(~mask.unsqueeze(1), float('-inf'))  # ~逐位取反。

        return arc_scores, kind_scores, mask

    def bert_encode_to_words(self, char_ids_list, word_lens_list):
        batch_size, seq_word_count = word_lens_list.shape
        chars_mask = char_ids_list.gt(0)
        chars_mask_l = chars_mask.to(dtype=torch.long)
        word_mask = word_lens_list.gt(0)

        bert_embed_size = self.bert.config.hidden_size
        bert_output, _ = self.bert(char_ids_list, chars_mask_l, chars_mask_l)  # output_all_encoded_layers=False
        all_chars_output = bert_output[chars_mask]  # 所有有效字符向量，一维
        all_words_detail = all_chars_output.split(word_lens_list[word_mask].tolist())  # 所有词包含的字向量，一维
        all_words_output = torch.stack([chars_out.mean(0) for chars_out in all_words_detail])  # 所有词输出向量，一维
        empty_words_out = bert_output.new_zeros(batch_size, seq_word_count, bert_embed_size)
        bert_words_out = empty_words_out.masked_scatter_(word_mask.unsqueeze(-1), all_words_output)
        if self.bert_projection is not None:
            bert_words_out = self.bert_projection(bert_words_out)
        return bert_words_out

    def get_loss(self, arc_scores, kind_scores, arcs_input, kinds_input, mask):
        arc_scores, arcs = arc_scores[mask], arcs_input[mask]
        kind_scores, kinds = kind_scores[mask], kinds_input[mask]
        kind_scores = kind_scores[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(arc_scores, arcs)
        kind_loss = self.criterion(kind_scores, kinds)
        loss = arc_loss + kind_loss

        return loss

    def predict(self, data_loader, device):
        self.eval()
        all_predict_arcs, all_predict_kinds = [], []
        for batch_data in data_loader:
            batch_data = tuple(t.to(device) for t in batch_data)
            arc_scores, kind_scores, mask = self.forward(*batch_data)
            # mask[:, 0] = 0
            predict_arcs, predict_kinds = decode_dependency(arc_scores, kind_scores, None)  # no easier
            all_predict_arcs.extend(predict_arcs)
            all_predict_kinds.extend(predict_kinds)
        return torch.cat(all_predict_arcs, dim=0).cpu().tolist(), \
            torch.cat(all_predict_kinds, dim=0).cpu().tolist()


class DependencyMetric:

    def __init__(self):
        super(DependencyMetric, self).__init__()
        self.empty = True
        self.total = 1
        self.arcs_ok = 0
        self.kinds_ok = 0

    @property
    def uas(self):
        return self.arcs_ok / self.total

    @property
    def las(self):
        return self.kinds_ok / self.total

    @property
    def score(self):
        return self.las

    def __repr__(self):
        return f"UAS: {self.uas:4.2%} LAS: {self.las:4.2%}"

    def __call__(self, predict_arcs, predict_kinds, true_arcs, true_kinds, mask):
        arc_mask = predict_arcs.eq(true_arcs)[mask]
        kind_mask = predict_kinds.eq(true_kinds)[mask] & arc_mask
        self.total += len(arc_mask)
        self.arcs_ok += arc_mask.sum().item()
        self.kinds_ok += kind_mask.sum().item()
        if self.empty and self.total > 1:
            self.empty = False
            self.total -= 1

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other


def decode_dependency(arc_scores, kind_scores, eisner_mask):
    if eisner_mask is None:
        predict_arcs = arc_scores.argmax(-1)
    else:
        predict_arcs = eisner(arc_scores, eisner_mask)
    predict_kinds = kind_scores.argmax(-1)
    predict_kinds = predict_kinds.gather(-1, predict_arcs.unsqueeze(-1)).squeeze(-1)

    return predict_arcs, predict_kinds


def backtrack(p_i, p_c, heads, i, j, complete):
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    """Returns a diagonal stripe of the tensor.

    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example::
    >>> x = torch.range(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    """
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


# noinspection PyUnresolvedReferences
def eisner(scores, mask):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(mask.device))

    return pad_sequence(predicts, True)
