from __future__ import print_function

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics.cluster import v_measure_score
from scipy.optimize import linear_sum_assignment

from .utils import log_sum_exp, data_iter, to_input_tensor, \
                   write_conll
from .projection import *



class MarkovFlow(nn.Module):
    def __init__(self, args, num_dims, num_vocab):
        super(MarkovFlow, self).__init__()

        self.args = args
        self.device = args.device

        self.num_state = args.num_state
        self.num_dims = num_dims
        self.num_vocab = num_vocab

        # transition parameters in log space
        self.tparams = Parameter(
            torch.Tensor(self.num_state, self.num_state))

        self.prior_group = [self.tparams]

        # tag embedding
        self.tag_embed = Parameter(torch.Tensor(self.num_state, self.num_dims))
        self.word_embed = torch.Tensor(self.num_vocab, self.num_dims)
        self.transform = Parameter(torch.Tensor(self.num_state, self.num_state))
        self.correction = Parameter(torch.Tensor(self.num_state, self.num_vocab))

        if args.mode == "unsupervised" and args.freeze_prior:
            self.tparams.requires_grad = False

        self.proj_group = [self.tag_embed]

        if args.mode == "unsupervised":
            self.proj_group += [self.transform, self.correction]


        # prior
        self.pi = torch.zeros(self.num_state,
                              dtype=torch.float32,
                              requires_grad=False,
                              device=self.device).fill_(1.0/self.num_state)

        self.pi = torch.log(self.pi)

    def init_params(self, train_data, word_vec_dict, id_to_word):
        """
        init_seed:(sents, masks)
        sents: (seq_length, batch_size, features)
        masks: (seq_length, batch_size)

        """

        # initialize transition matrix params
        # self.tparams.data.uniform_().add_(1)
        self.tparams.data.uniform_()
        self.correction.zero_()
        nn.init.eye_(self.transform)

        word_embed = [word_vec_dict[i] for i in range(len(id_to_word))]
        self.word_embed = torch.Tensor(word_embed, dtype=torch.float32, requires_grad=False, device=self.device)

        # load pretrained model
        if self.args.load_nice != '':
            self.load_state_dict(torch.load(self.args.load_nice), strict=False)

            self.tag_embed_init = self.tag_embed.clone()
            self.tparams_init = self.tparams.clone()
            self.transform_init = torch.eyes(self.num_state)
            self.correction_init = torch.zeros((self.num_state, self.num_vocab))

            # self.means_init.requires_grad = False
            # self.tparams_init.requires_grad = False
            # for tensor in self.proj_init:
            #     tensor.requires_grad = False

            return

    def init_mean(self, train_data):
        emb_dict = {}
        cnt_dict = Counter()
        for iter_obj in train_data.data_iter(self.args.batch_size):
            sents_t = iter_obj.embed
            sents_t, _ = self.transform(sents_t, iter_obj.mask)
            sents_t = sents_t.transpose(0, 1)
            pos_t = iter_obj.pos.transpose(0, 1)
            mask_t = iter_obj.mask.transpose(0, 1)


            for emb_s, tagid_s, mask_s in zip(sents_t, pos_t, mask_t):
                for tagid, emb, mask in zip(tagid_s, emb_s, mask_s):
                    tagid = tagid.item()
                    mask = mask.item()
                    if tagid in emb_dict:
                        emb_dict[tagid] = emb_dict[tagid] + emb * mask
                    else:
                        emb_dict[tagid] = emb * mask

                    cnt_dict[tagid] += mask

        for tagid in emb_dict:
            self.means[tagid] = emb_dict[tagid] / cnt_dict[tagid]

    def init_var(self, train_data):
        cnt = 0
        mean_sum = 0.
        var_sum = 0.
        for iter_obj in train_data.data_iter(batch_size=self.args.batch_size):
            sents, masks = iter_obj.embed, iter_obj.mask
            sents, _ = self.transform(sents, masks)
            seq_length, _, features = sents.size()
            flat_sents = sents.view(-1, features)
            mean_sum = mean_sum + torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                flat_sents, dim=0)
            cnt += masks.sum().item()

        mean = mean_sum / cnt

        for iter_obj in train_data.data_iter(batch_size=self.args.batch_size):
            sents, masks = iter_obj.embed, iter_obj.mask
            sents, _ = self.transform(sents, masks)
            seq_length, _, features = sents.size()
            flat_sents = sents.view(-1, features)
            var_sum = var_sum + torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                                 ((flat_sents - mean.expand_as(flat_sents)) ** 2), dim = 0)
        var = var_sum / cnt
        self.var.copy_(var)

    def _calc_log_density_c(self):
        # return -self.num_dims/2.0 * (math.log(2) + \
        #         math.log(np.pi)) - 0.5 * self.num_dims * (torch.log(self.var))

        return -self.num_dims/2.0 * (math.log(2) + \
                math.log(np.pi)) - 0.5 * torch.sum(torch.log(self.var))

    def transform(self, x, masks=None):
        """
        Args:
            x: (sent_length, batch_size, num_dims)
        """
        jacobian_loss = torch.zeros(1, device=self.device, requires_grad=False)

        if self.args.model != 'gaussian':
            x, jacobian_loss_new = self.proj_layer(x, masks)
            jacobian_loss = jacobian_loss + jacobian_loss_new


        return x, jacobian_loss

    def MLE_loss(self):
        # diff1 = ((self.means - self.means_init) ** 2).sum()
        diff_prior = ((self.tparams - self.tparams_init) ** 2).sum()

        # diff = diff1 + diff2
        diff_tag = ((self.tag_embed - self.tag_embed_init) ** 2).sum()

        diff_trans = ((self.transform - self.transform_init) ** 2).sum()
        diff_correct = ((self.correction - self.correction_init) ** 2).sum()

        return 0.5 * self.args.beta * (diff_prior + diff_tag + diff_trans + diff_correct)

    def unsupervised_loss(self, words, masks):
        """
        Args:
            words: (sent_length, batch_size)
            masks: (sent_length, batch_size)

        Returns: Tensor1, Tensor2
            Tensor1: negative log likelihood, shape ([])
            Tensor2: jacobian loss, shape ([])


        """
        self.update_emission()
        max_length, batch_size = words.size()
        # sents, jacobian_loss = self.transform(sents, masks)

        self.logA = self._calc_logA()
        # self.log_density_c = self._calc_log_density_c()

        alpha = self.pi + self._eval_density(words[0])
        for t in range(1, max_length):
            density = self._eval_density(sents[t], words[t])
            mask_ep = masks[t].expand(self.num_state, batch_size) \
                      .transpose(0, 1)
            alpha = torch.mul(mask_ep,
                              self._forward_cell(alpha, density)) + \
                    torch.mul(1-mask_ep, alpha)

        # calculate objective from log space
        objective = torch.sum(log_sum_exp(alpha, dim=1))

        return -objective

    def supervised_loss(self, words, tags, masks):
        """
        Args:
            words: (sent_length, batch_size)
            masks: (sent_length, batch_size)
            tags:  (sent_length, batch_size)

        Returns: Tensor1, Tensor2
            Tensor1: negative log likelihood, shape ([])
            Tensor2: jacobian loss, shape ([])

        """

        sent_len, batch_size = words.size()
        self.update_emission()

        log_emission_prob = self.log_emission.view(1, 1, *self.log_emission.size())\
                            .expand(sent_len, batch_size, self.num_vocab, self.num_state)
        words = words.view(sent_len, batch_size, 1, 1).expand(
            sent_len, batch_size, 1, self.num_state)
        log_emission_prob = torch.gather(log_emission_prob, index=words, dim=2).squeeze(2)
        tags_exp = tags.unsqueeze(-1)

        # (sent_len, batch_size)
        log_emission_prob = torch.gather(log_emission_prob, index=tags_exp, dim=2).squeeze(2)

        log_emission_prob = torch.mul(masks, log_emission_prob).sum()

        # (num_state, num_state)
        log_trans = self._calc_logA()

        # (sent_len, batch_size, num_state, num_state)
        log_trans_prob = log_trans.view(1, 1, *log_trans.size()).expand(
            sent_len, batch_size, *log_trans.size())

        # (sent_len-1, batch_size, 1, num_state)
        tag_id = tags.view(*tags.size(), 1, 1).expand(sent_len,
            batch_size, 1, self.num_state)[:-1]

        # (sent_len-1, batch_size, 1, num_state)
        log_trans_prob = torch.gather(log_trans_prob[:-1], dim=2, index=tag_id)

        # (sent_len-1, batch_size, 1, 1)
        tag_id = tags.view(*tags.size(), 1, 1)[1:]

        # (sent_len-1, batch_size)
        log_trans_prob = torch.gather(log_trans_prob, dim=3,
            index=tag_id).squeeze()

        log_trans_prob = torch.mul(masks[1:], log_trans_prob)

        log_trans_prior = self.pi.expand(batch_size, self.num_state)
        tag_id = tags[0].unsqueeze(dim=1)

        # (batch_size)
        log_trans_prior = torch.gather(log_trans_prior, dim=1,
            index=tag_id).sum()

        log_trans_prob = log_trans_prior + log_trans_prob.sum()

        return -(log_trans_prob + log_emission_prob)


    def _calc_alpha(self, sents, masks):
        """
        sents: (sent_length, batch_size, self.num_dims)
        masks: (sent_length, batch_size)

        Returns:
            output: (batch_size, sent_length, num_state)

        """
        max_length, batch_size, _ = sents.size()

        alpha_all = []
        alpha = self.pi + self._eval_density(sents[0])
        alpha_all.append(alpha.unsqueeze(1))
        for t in range(1, max_length):
            density = self._eval_density(sents[t])
            mask_ep = masks[t].expand(self.num_state, batch_size) \
                      .transpose(0, 1)
            alpha = torch.mul(mask_ep, self._forward_cell(alpha, density)) + \
                    torch.mul(1-mask_ep, alpha)
            alpha_all.append(alpha.unsqueeze(1))

        return torch.cat(alpha_all, dim=1)

    def _forward_cell(self, alpha, density):
        batch_size = len(alpha)
        ep_size = torch.Size([batch_size, self.num_state, self.num_state])
        alpha = log_sum_exp(alpha.unsqueeze(dim=2).expand(ep_size) +
                            self.logA.expand(ep_size) +
                            density.unsqueeze(dim=1).expand(ep_size), dim=1)

        return alpha

    def _backward_cell(self, beta, density):
        """
        density: (batch_size, num_state)
        beta: (batch_size, num_state)

        """
        batch_size = len(beta)
        ep_size = torch.Size([batch_size, self.num_state, self.num_state])
        beta = log_sum_exp(self.logA.expand(ep_size) +
                           density.unsqueeze(dim=1).expand(ep_size) +
                           beta.unsqueeze(dim=1).expand(ep_size), dim=2)

        return beta

    def update_emission(self):
        if self.args.mode == "unsupervised":
            word_emb = torch.mm(self.word_embed, self.transform)
        else:
            word_emb = self.word_embed

        # (num_vocab, num_state)
        log_emission = torch.mm(word_emb, self.tag_embed.transpose(1, 0))

        if self.args.mode == "unsupervised":
            log_emission = log_emission + self.correction.transpose(1, 0)

        self.log_emission =  F.log_softmax(log_emission, dim=0)


    def _eval_density(self, words):
        """
        Args:
            words: (batch_size)

        Returns: Tensor1
            Tensor1: the density tensor with shape (batch_size, num_state)
        """

        batch_size = words.size(0)
        log_emission = self.log_emission.expand(batch_size, *self.log_emission.size())
        words = words.view(batch_size, 1, 1).expand(batch_size, 1, self.num_state)

        return torch.gather(log_emission, index=words, dim=1).squeeze(1)

    def _calc_logA(self):
        return (self.tparams - \
                log_sum_exp(self.tparams, dim=1, keepdim=True) \
                .expand(self.num_state, self.num_state))

    def _calc_log_mul_emit(self):
        return self.emission - \
                log_sum_exp(self.emission, dim=1, keepdim=True) \
                .expand(self.num_state, self.vocab_size)

    def _viterbi(self, sents, masks):
        """
        Args:
            sents: (sent_length, batch_size)
            masks: (sent_length, batch_size)
        """

        self.logA = self._calc_logA()

        length, batch_size = masks.size()

        # (batch_size, num_state)
        delta = self.pi + self._eval_density(sents[0])

        ep_size = torch.Size([batch_size, self.num_state, self.num_state])
        index_all = []

        # forward calculate delta
        for t in range(1, length):
            density = self._eval_density(sents[t])
            delta_new = self.logA.expand(ep_size) + \
                    density.unsqueeze(dim=1).expand(ep_size) + \
                    delta.unsqueeze(dim=2).expand(ep_size)
            mask_ep = masks[t].view(-1, 1, 1).expand(ep_size)
            delta = mask_ep * delta_new + \
                    (1 - mask_ep) * delta.unsqueeze(dim=1).expand(ep_size)

            # index: (batch_size, num_state)
            delta, index = torch.max(delta, dim=1)
            index_all.append(index)

        assign_all = []
        # assign: (batch_size)
        _, assign = torch.max(delta, dim=1)
        assign_all.append(assign.unsqueeze(dim=1))

        # backward retrieve path
        # len(index_all) = length-1
        for t in range(length-2, -1, -1):
            assign_new = torch.gather(index_all[t],
                                      dim=1,
                                      index=assign.view(-1, 1)).squeeze(dim=1)

            assign_new = assign_new.float()
            assign = assign.float()
            assign = masks[t+1] * assign_new + (1 - masks[t+1]) * assign
            assign = assign.long()

            assign_all.append(assign.unsqueeze(dim=1))

        assign_all = assign_all[-1::-1]

        return torch.cat(assign_all, dim=1)

    def test_supervised(self,
                        test_data):
        """Evaluate tagging performance with
        token-level supervised accuracy

        Args:
            test_data: ConlluData object

        Returns: a scalar accuracy value

        """
        total = 0.0
        correct = 0.0

        index_all = []
        eval_tags = []

        for iter_obj in test_data.data_iter(batch_size=self.args.batch_size,
                                            shuffle=False):
            sents_t = iter_obj.words
            masks = iter_obj.mask
            tags_t = iter_obj.pos

            # index: (batch_size, seq_length)
            index = self._viterbi(sents_t, masks)


            for index_s, tag_s, mask_s in zip(index, tags_t.transpose(0, 1), masks.transpose(0, 1)):
                for i in range(int(mask_s.sum().item())):
                    if index_s[i].item() == tag_s[i].item():
                        correct += 1
                    total += 1

        return correct / total


    def test_unsupervised(self,
                         test_data,
                         sentences=None,
                         tagging=False,
                         path=None,
                         null_index=None):
        """Evaluate tagging performance with
        many-to-1 metric, VM score and 1-to-1
        accuracy

        Args:
            test_data: ConlluData object
            tagging: output the predicted tags if True
            path: The output tag file path
            null_index: the null element location in Penn
                        Treebank, only used for writing unsupervised
                        tags for downstream parsing task

        Returns:
            Tuple1: (M1, VM score, 1-to-1 accuracy)

        """


        total = 0.0
        correct = 0.0
        cnt_stats = {}
        match_dict = {}

        index_all = []
        eval_tags = []

        gold_vm = []
        model_vm = []

        for i in range(self.num_state):
            cnt_stats[i] = Counter()

        for iter_obj in test_data.data_iter(batch_size=self.args.batch_size,
                                            shuffle=False):
            total += iter_obj.mask.sum().item()
            sents_t = iter_obj.words
            tags_t = iter_obj.pos
            masks = iter_obj.mask

            # index: (batch_size, seq_length)
            index = self._viterbi(sents_t, masks)

            index_all += list(index)

            tags = [tags_t[:int(masks[:,i].sum().item()), i] for i in range(index.size(0))]
            eval_tags += tags

            # count
            for (seq_gold_tags, seq_model_tags) in zip(tags, index):
                for (gold_tag, model_tag) in zip(seq_gold_tags, seq_model_tags):
                    model_tag = model_tag.item()
                    gold_tag = gold_tag.item()
                    gold_vm += [gold_tag]
                    model_vm += [model_tag]
                    cnt_stats[model_tag][gold_tag] += 1

        # evaluate one-to-one accuracy
        cost_matrix = np.zeros((self.num_state, self.num_state))
        for i in range(self.num_state):
            for j in range(self.num_state):
                cost_matrix[i][j] = -cnt_stats[j][i]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for (seq_gold_tags, seq_model_tags) in zip(eval_tags, index_all):
            for (gold_tag, model_tag) in zip(seq_gold_tags, seq_model_tags):
                model_tag = model_tag.item()
                gold_tag = gold_tag.item()

                if col_ind[gold_tag] == model_tag:
                    correct += 1

        one2one = correct / total

        correct = 0.

        # match
        for tag in cnt_stats:
            if len(cnt_stats[tag]) != 0:
                match_dict[tag] = cnt_stats[tag].most_common(1)[0][0]

        # eval many2one
        for (seq_gold_tags, seq_model_tags) in zip(eval_tags, index_all):
            for (gold_tag, model_tag) in zip(seq_gold_tags, seq_model_tags):
                model_tag = model_tag.item()
                gold_tag = gold_tag.item()
                if match_dict[model_tag] == gold_tag:
                    correct += 1

        if tagging:
            write_conll(path, sentences, index_all, null_index)

        return correct/total, v_measure_score(gold_vm, model_vm), one2one
