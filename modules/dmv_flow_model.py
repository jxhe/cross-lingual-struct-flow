from __future__ import print_function
import math
import pickle
import torch
import torch.nn as nn

import numpy as np

from collections import Counter, namedtuple
from .projection import NICETrans, LSTMNICE
from .dmv_viterbi_model import DMVDict

from torch.nn import Parameter

from .utils import log_sum_exp, \
                   unravel_index, \
                   data_iter, \
                   to_input_tensor, \
                   stable_math_log
NEG_INFINITY = -1e20

ParseTree = namedtuple("parsetree", ["tree", "decode_tag", "children"])


def test_piodict(piodict):
    """
    test PIOdict 0 value

    """
    for key, value in piodict.dict.iteritems():
        if value <= 0:
            print(key, value)
            return False
    return True

def log_softmax(input, dim):
    return (input - \
            log_sum_exp(input, dim=dim, keepdim=True) \
            .expand_as(input))


class DMVFlow(nn.Module):
    def __init__(self, args, num_state, num_dims,
                 punc_sym, word_vec_dict=None):
        super(DMVFlow, self).__init__()

        self.num_state = num_state
        self.num_dims = num_dims + args.pos_emb_dim
        self.pos_emb_dim = args.pos_emb_dim
        self.args = args
        self.device = args.device

        self.hidden_units = self.num_dims // 2
        self.lstm_hidden_units = self.num_dims

        self.punc_sym = punc_sym
        self.word2vec = word_vec_dict

        self.harmonic = False


        self.pos_embed = nn.Embedding(num_state, self.pos_emb_dim)
        self.proj_group = list(self.pos_embed.parameters())
        if args.freeze_pos_emb:
            self.pos_embed.weight.requires_grad = False


        self.means = Parameter(torch.Tensor(self.num_state, self.num_dims))

        if args.model == 'nice':
            self.proj_layer = NICETrans(self.args.couple_layers,
                                        self.args.cell_layers,
                                        self.hidden_units,
                                        self.num_dims,
                                        self.device)
        elif args.model == "lstmnice":
            self.proj_layer = LSTMNICE(self.args.lstm_layers,
                                       self.args.couple_layers,
                                       self.args.cell_layers,
                                       self.lstm_hidden_units,
                                       self.hidden_units,
                                       self.num_dims,
                                       self.device)


        # Gaussian Variance
        self.var = Parameter(torch.zeros((num_state, self.num_dims), dtype=torch.float32))

        if not self.args.train_var:
            self.var.requires_grad = False

        # dim0 is head and dim1 is dependent
        self.attach_left = Parameter(torch.Tensor(self.num_state, self.num_state))
        self.attach_right = Parameter(torch.Tensor(self.num_state, self.num_state))

        # (stop, adj, h)
        # dim0: 0 is nonstop, 1 is stop
        # dim2: 0 is nonadjacent, 1 is adjacent
        self.stop_right = Parameter(torch.Tensor(2, self.num_state, 2))
        self.stop_left = Parameter(torch.Tensor(2, self.num_state, 2))

        self.root_attach_left = Parameter(torch.Tensor(self.num_state))

        self.prior_group = [self.attach_left, self.attach_right, self.stop_left, self.stop_right, \
                            self.root_attach_left]

        if args.model == "gaussian":
            self.proj_group += [self.means, self.var]
        else:
            self.proj_group += list(self.proj_layer.parameters()) + [self.means, self.var]

        if self.args.freeze_prior:
            for x in self.prior_group:
                x.requires_grad = False

        if self.args.freeze_proj:
            for param in self.proj_layer.parameters():
                param.requires_grad = False

        if self.args.freeze_mean:
            self.means.requires_grad = False

    def init_params(self, init_seed, train_data):
        """
        init_seed:(sents, masks)
        sents: (seq_length, batch_size, features)
        masks: (seq_length, batch_size)

        """

        if self.args.load_nice != '':
            self.load_state_dict(torch.load(self.args.load_nice), strict=True)
            if self.args.init_mean:
                self.init_mean(train_data)

            if self.args.init_var:
                self.init_var(train_data)
            return

        if self.args.load_gaussian != '':
            self.load_state_dict(torch.load(self.args.load_gaussian), strict=True)
            return

        # init transition params
        self.attach_left.uniform_().add_(1.)
        self.attach_right.uniform_().add_(1.)
        self.root_attach_left.uniform_().add_(1)

        self.stop_right[0, :, 0].uniform_().add_(1)
        self.stop_right[1, :, 0].uniform_().add_(2)
        self.stop_left[0, :, 0].uniform_().add_(1)
        self.stop_left[1, :, 0].uniform_().add_(2)

        self.stop_right[0, :, 1].uniform_().add_(2)
        self.stop_right[1, :, 1].uniform_().add_(1)
        self.stop_left[0, :, 1].uniform_().add_(2)
        self.stop_left[1, :, 1].uniform_().add_(1)

        self.var.uniform_(0.5, 1.5)

        # initialize mean and variance with empirical values
        sents = init_seed.embed
        masks = init_seed.mask

        if self.pos_emb_dim > 0:
            pos = init_seed.pos
            pos_embed = self.pos_embed(pos)
            sents = torch.cat((sents, pos_embed), dim=-1)

        sents, _ = self.transform(sents, masks)


        features = sents.size(-1)
        flat_sents = sents.view(-1, features)
        seed_mean = torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                              flat_sents, dim=0) / masks.sum()
        seed_var = torch.sum(masks.view(-1, 1).expand_as(flat_sents) *
                             ((flat_sents - seed_mean.expand_as(flat_sents)) ** 2),
                             dim=0) / masks.sum()

        # self.var.copy_(2 * seed_var)
        self.init_mean(train_data)
        self.init_var(train_data)

    def init_mean(self, train_data):
        emb_dict = {}
        cnt_dict = Counter()
        for iter_obj in train_data.data_iter(self.args.batch_size):
            sents_t = iter_obj.embed
            if self.args.pos_emb_dim > 0:
                pos_embed_t = self.pos_embed(iter_obj.pos)
                sents_t = torch.cat((sents_t, pos_embed_t), dim=-1)

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
        emb_dict = {}
        cnt_dict = Counter()
        for iter_obj in train_data.data_iter(self.args.batch_size):
            sents_t = iter_obj.embed
            if self.args.pos_emb_dim > 0:
                pos_embed_t = self.pos_embed(iter_obj.pos)
                sents_t = torch.cat((sents_t, pos_embed_t), dim=-1)

            sents_t, _ = self.transform(sents_t, iter_obj.mask)
            sents_t = sents_t.transpose(0, 1)
            pos_t = iter_obj.pos.transpose(0, 1)
            mask_t = iter_obj.mask.transpose(0, 1)

            for emb_s, tagid_s, mask_s in zip(sents_t, pos_t, mask_t):
                for tagid, emb, mask in zip(tagid_s, emb_s, mask_s):
                    tagid = tagid.item()
                    mask = mask.item()
                    if tagid in emb_dict:
                        emb_dict[tagid] = emb_dict[tagid] + (emb - self.means[tagid]) ** 2 * mask
                    else:
                        emb_dict[tagid] = (emb - self.means[tagid]) ** 2 * mask

                    cnt_dict[tagid] += mask

        for tagid in emb_dict:
            self.var[tagid] = emb_dict[tagid] / cnt_dict[tagid]
            # self.var[tagid][self.num_dims:].fill_(5.)
            self.var[tagid][:].fill_(1.)

    def print_param(self):
        print("attatch left")
        print(self.attach_left)
        print("attach right")
        print(self.attach_right)
        print("stop left")
        print(self.stop_left)
        print("root attach left")
        print(self.root_attach_left)


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

    def tree_to_depset(self, root_max_index, sent_len):
        """
        Args:
            root_max_index: (batch_size, 2), [:0] represents the 
                            optimal state, [:1] represents the 
                            optimal index (location)
        """
        # add the root symbol (-1)
        batch_size = root_max_index.size(0)
        dep_list = []
        for batch in range(batch_size):
            res = set([(root_max_index[batch, 1].item(), -1, root_max_index[batch, 0].item())])
            start = 0
            end = sent_len[batch]
            res.update(self._tree_to_depset(start, end, 2, batch, root_max_index[batch, 0],
                                            root_max_index[batch, 1]))
            assert len(res) == sent_len[batch]
            dep_list += [sorted(res)]

        return dep_list

    def _tree_to_depset(self, start, end, mark, batch, symbol, index):
        left_child = self.left_child[start, end, mark][batch, symbol, index]
        right_child = self.right_child[start, end, mark][batch, symbol, index]

        if left_child[0] == 1 and right_child[0] == 1:
            if mark == 0:
                assert left_child[3] == 0
                assert right_child[3] == 2
                arg = right_child[-1]
                dep_symbol = right_child[4].item()
            elif mark == 1:
                assert left_child[3] == 2
                assert right_child[3] == 1
                arg = left_child[-1]
                dep_symbol = left_child[4].item()

            res = set([(arg.item(), index, dep_symbol)])
            res.update(self._tree_to_depset(left_child[1].item(), left_child[2].item(),
                                            left_child[3].item(), batch, left_child[4].item(),
                                            left_child[5].item()), \
                       self._tree_to_depset(right_child[1].item(), right_child[2].item(),
                                            right_child[3].item(), batch, right_child[4].item(),
                                            right_child[5].item()))

        elif left_child[0] == 1 and right_child[0] == 0:
            res = self._tree_to_depset(left_child[1].item(), left_child[2].item(),
                                       left_child[3].item(), batch, left_child[4].item(),
                                       left_child[5].item())
        elif left_child[0] == -1 and right_child[0] == -1:
            res = set()

        else:
            raise ValueError

        return res

    def test(self, test_data, batch_size=10):
        """
        Args:
            gold: A nested list of heads
            all_len: True if evaluate on all lengths
        """
        cnt = 0
        dir_cnt = 0.0
        undir_cnt = 0.0
        memory_sent_cnt = 0

        batch_id_ = 0
        if self.args.max_len > 20:
            batch_size = 2

        for iter_obj in test_data.data_iter(batch_size=batch_size,
                                            shuffle=False):

            batch_id_ += 1
            try:
                sents_t = iter_obj.embed
                if self.args.pos_emb_dim > 0:
                    pos_embed = self.pos_embed(iter_obj.pos)
                    sents_t = torch.cat((sents_t, pos_embed), dim=-1)

                sents_t, _ = self.transform(sents_t, iter_obj.mask)


                sents_t = sents_t.transpose(0, 1)
                # root_max_index: (batch_size, num_state, seq_length)
                batch_size, seq_length, _ = sents_t.size()
                symbol_index_t = self.attach_left.new([[[p, q] for q in range(seq_length)] \
                                                      for p in range(self.num_state)]) \
                                                      .expand(batch_size, self.num_state, seq_length, 2)
                root_max_index = self.dep_parse(sents_t, iter_obj, symbol_index_t)
                masks = iter_obj.mask
                batch_size = masks.size(1)
                sent_len = [torch.sum(masks[:, i]).item() for i in range(batch_size)]
                parse = self.tree_to_depset(root_max_index, sent_len)
            except RuntimeError:
                memory_sent_cnt += 1
                print('batch %d out of memory' % batch_id_)
                continue

            #TODO: check parse_s if follows original sentence order
            for pos_s, gold_s, parse_s, len_ in zip(iter_obj.pos.transpose(0, 1),
                    iter_obj.head.transpose(0, 1), parse, iter_obj.mask.sum(dim=0)):
                directed, length = self.measures(pos_s, gold_s, parse_s, len_.item())
                cnt += length
                dir_cnt += directed

        dir_acu = dir_cnt / cnt

        self.log_p_parse = {}
        self.left_child = {}
        self.right_child = {}

        return dir_acu

    def measures(self, pos_s, gold_s, parse_s, len_):
        # Helper for eval().
        d = 0.
        l = 0.
        for i in range(int(len_)):
            pos = pos_s[i]
            head = gold_s[i]
            tuple_ = parse_s[i]
            if pos.item() not in self.punc_sym:
                l += 1
                if head.item() == tuple_[1]:
                    d += 1

        return d, l

    def up_viterbi_em(self, train_data):
        attach_left = self.attach_left.new_ones((self.num_state, self.num_state))
        attach_right = self.attach_right.new_ones((self.num_state, self.num_state))

        stop_right = self.stop_right.new_ones((2, self.num_state, 2))
        stop_left = self.stop_left.new_ones((2, self.num_state, 2))

        root_attach_left = self.root_attach_left.new_ones(self.num_state)
        for iter_obj in train_data.data_iter(batch_size=batch_size,
                                             shuffle=False):
            sents_t = iter_obj.embed
            if self.args.pos_emb_dim > 0:
                pos_embed = self.pos_embed(iter_obj.pos)
                sents_t = torch.cat((sents_t, pos_embed), dim=-1)

            sents_t, _ = self.transform(sents_t, iter_obj.mask)            

            # root_max_index: (batch_size, num_state, seq_length)
            batch_size, seq_length, _ = sents_t.size()
            symbol_index_t = self.attach_left.new([[[p, q] for q in range(seq_length)] \
                                                  for p in range(self.num_state)]) \
                                                  .expand(batch_size, self.num_state, seq_length, 2)
            root_max_index = self.dep_parse(sents_t, iter_obj, symbol_index_t)
            masks = iter_obj.mask
            batch_size = masks.size(1)
            sent_len = [torch.sum(masks[:, i]).item() for i in range(batch_size)]
            parse = self.tree_to_depset(root_max_index, sent_len)

            for s in parse:
                length = len(s)
                left = [0] * length
                right = [0] * length

                # count number of left and right children
                for i in range(length):
                    head_id = s[i][1]
                    dep_id = s[i][0]
                    if dep_id < head_id:
                        left[head_id] += 1
                    elif dep_id > head_id:
                        right[head_id] += 1
                    else:
                        raise ValueError

                for i in range(length):
                    head_id = s[i][1]
                    head_pos = s[head][2]
                    dep_pos = s[i][2]
                    dep_id = s[i][0]

                    if head_id == -1:
                        root_attach_left[dep_pos] += 1
                        continue

                    assert(i == dep_id)

                    if dep_id < head_id:
                        attach_left[head_pos, dep_pos] += 1
                    elif dep_id > head_id:
                        attach_right[head_pos, dep_pos] += 1

                    if left[i] > 0:
                        stop_left[0, dep_pos, 1] += 1
                        stop_left[0, dep_pos, 0] += left[i] - 1
                        stop_left[1, dep_pos, 0] += 1
                    else:
                        stop_left[1, dep_pos, 1] += 1


                    if right[i] > 0:
                        stop_right[0, dep_pos, 1] += 1
                        stop_right[0, dep_pos, 0] += right[i] - 1
                        stop_right[1, dep_pos, 0] += 1
                    else:
                        stop_right[1, dep_pos, 1] += 1
        
        self.attach_left.copy_(torch.log(attach_left / attach_left.sum(dim=1, keepdim=True)))
        self.attach_right.copy_(torch.log(attach_right / attach_right.sum(dim=1, keepdim=True)))

        self.stop_right.copy_(torch.log(stop_right / stop_right.sum(dim=0, keepdim=False)))
        self.stop_left.copy_(torch.log(stop_left / stop_left.sum(dim=0, keepdim=False)))
        self.root_attach_left.copy_(torch.log(root_attach_left / root_attach_left.sum()))

    def _eval_log_density(self, s):
        """
        Args:
            s: A tensor with size (batch_size, seq_length, features)

        Returns:
            density: (batch_size, seq_length, num_state)

        """
        constant = -self.num_dims/2.0 * (math.log(2 * math.pi)) - \
                0.5 * torch.sum(torch.log(self.var), dim=-1)

        batch_size, seq_length, features = s.size()
        means = self.means.view(1, 1, self.num_state, features)
        words = s.unsqueeze(dim=2)
        var = self.var.view(1, 1, self.num_state, self.num_dims)
        return constant.view(1, 1, self.num_state) - \
               0.5 * torch.sum((means - words) ** 2 / var, dim=3)

    def _eval_log_density_supervised(self, sents, pos):
        """
        Args:
            sents: A tensor with size (batch_size, seq_len, features)
            pos: (batch_size, seq_len)

        Returns:
            density: (batch_size, seq_length)

        """
        constant = -self.num_dims/2.0 * (math.log(2 * math.pi)) - \
                0.5 * torch.sum(torch.log(self.var), dim=-1)

        batch_size, seq_length, features = sents.size()

        constant = constant.view(1, 1, self.num_state)
        constant = constant.expand(batch_size, seq_length, self.num_state)
        constant = torch.gather(constant, dim=2, index=pos.unsqueeze(2)).squeeze(2)

        means = self.means.view(1, 1, self.num_state, features)
        means = means.expand(batch_size, seq_length,
            self.num_state, self.num_dims)
        tag_id = pos.view(*pos.size(), 1, 1).expand(batch_size,
            seq_length, 1, self.num_dims)

        # (batch_size, seq_len, num_dims)
        means = torch.gather(means, dim=2, index=tag_id).squeeze(2)

        var = self.var.view(1, 1, self.num_state, self.num_dims)
        var = var.expand(batch_size, seq_length,
                self.num_state, self.num_dims)
        var = torch.gather(var, dim=2, index=tag_id).squeeze(2)

        return constant - \
               0.5 * torch.sum((means - sents) ** 2 / var, dim=-1)

    def set_dmv_params(self, train_data, pos_seq=None):
        self.attach_left.fill_(1.)
        self.attach_right.fill_(1.)
        self.root_attach_left.fill_(1.)
        self.stop_right.fill_(1.)
        self.stop_left.fill_(1.)

        if pos_seq is None:
            pos_seq = train_data.postags

        for pos_s, head_s, left_s, right_s in zip(pos_seq,
                                                  train_data.heads,
                                                  train_data.left_num_deps,
                                                  train_data.right_num_deps):
            assert(len(pos_s) == len(head_s))
            for i, pos in enumerate(pos_s):
                head = head_s[i]
                head_pos = pos_s[head]
                left = left_s[i]
                right = right_s[i]

                if head == -1:
                    self.root_attach_left[pos] += 1
                    continue

                assert(i != head)
                if i < head:
                    self.attach_left[head_pos, pos] += 1
                elif i > head:
                    self.attach_right[head_pos, pos] += 1

                if left > 0:
                    self.stop_left[0, pos, 1] += 1
                    self.stop_left[0, pos, 0] += left - 1
                    self.stop_left[1, pos, 0] += 1
                else:
                    self.stop_left[1, pos, 1] += 1

                if right > 0:
                    self.stop_right[0, pos, 1] += 1
                    self.stop_right[0, pos, 0] += right - 1
                    self.stop_right[1, pos, 0] += 1
                else:
                    self.stop_right[1, pos, 1] += 1

        self.attach_left.copy_(torch.log(self.attach_left / self.attach_left.sum(dim=1, keepdim=True)))
        self.attach_right.copy_(torch.log(self.attach_right / self.attach_right.sum(dim=1, keepdim=True)))

        self.stop_right.copy_(torch.log(self.stop_right / self.stop_right.sum(dim=0, keepdim=False)))
        self.stop_left.copy_(torch.log(self.stop_left / self.stop_left.sum(dim=0, keepdim=False)))
        self.root_attach_left.copy_(torch.log(self.root_attach_left / self.root_attach_left.sum()))



    def supervised_loss_wpos(self, iter_obj):
        """
        Args:
            iter_obj.embed: (seq_len, batch_size, num_dim)
            iter_obj.pos: (seq_len, batch_size)
            iter_obj.head: (seq_len, batch_size)
            iter_obj.l_deps: (seq_len, batch_size)
            iter_obj.r_deps: (seq_len, batch_size)
            iter_obj.mask: (seq_len, batch_size)

        """

        embed = iter_obj.embed.transpose(0, 1)
        # (batch_size, seq_len)
        pos_t = iter_obj.pos.transpose(0, 1)

        if self.args.pos_emb_dim > 0:
            pos_embed = self.pos_embed(pos_t)
            embed = torch.cat((embed, pos_t), dim=-1)

        embed, jacob = self.transform(embed, iter_obj.mask)

        density = self._eval_log_density_supervised(embed, pos_t)

        log_emission_prob = torch.mul(density, iter_obj.mask.transpose(0, 1)).sum()

        return -log_emission_prob, jacob


    def supervised_loss_wopos(self, tree, embed, pos):
        """This is the non-batched version of supervised loss when
        dep structure is known but pos tags are unknown.

        Args:
            tree: TreeToken object from conllu
            embed: list of embeddings
            pos: list of pos tag ids

        Returns: Tensor1, Tensor2
            Tensor1: a scalar tensor of nll
            Tensor2: Jacobian loss, scalar tensor

        """
        # normalizing parameters
        self.log_attach_left = self.args.prob_const * log_softmax(self.attach_left, dim=1)
        self.log_attach_right = self.args.prob_const * log_softmax(self.attach_right, dim=1)
        self.log_stop_right = self.args.prob_const * log_softmax(self.stop_right, dim=0)
        self.log_stop_left = self.args.prob_const * log_softmax(self.stop_left, dim=0)
        self.log_root_attach_left = self.args.prob_const * log_softmax(self.root_attach_left, dim=0)

        constant = -self.num_dims/2.0 * (math.log(2 * math.pi)) - \
                0.5 * torch.sum(torch.log(self.var), dim=-1)

        # (seq_len, num_dims)
        embed_t = torch.tensor(embed, dtype=torch.float32, requires_grad=False, device=self.device)
        if self.args.pos_emb_dim > 0:
            pos_t = torch.tensor(pos, dtype=torch.long, requires_grad=False, device=self.device)
            pos_embed = self.pos_embed(pos_t)
            embed_t = torch.cat((embed_t, pos_embed), dim=-1)

        embed_t, jacob = self.transform(embed_t.unsqueeze(1))
        embed_t = embed_t.squeeze(1)


        # (num_state)
        log_prob = self._calc_log_prob(tree, constant, embed_t)

        log_prob = self.log_root_attach_left + log_prob

        return -log_sum_exp(log_prob, dim=0), jacob

    def _calc_log_prob(self, tree, constant, embed_s):
        """recursion components to compute the log prob of the root
        of the current tree being a latent pos tag

        Args:
            tree: TreeToken
            consant: the Gaussian density constant

        Returns: Tensor1, Tensor2:
            Tensor1: log prob of root beging a latent pos tag,
                     shape [num_state]
            Tensor2: Jacobian loss, with shape []
        """

        token_id = tree.token["id"]
        embed = embed_s[token_id-1]

        # (num_state)
        embed = embed.unsqueeze(0)
        log_prob = constant - 0.5 * torch.sum(
            (self.means - embed)**2 / self.var, dim=1)

        # leaf nodes
        if tree.children == []:
            return log_prob


        left = []
        right = []

        for t in tree.children:
            if t.token["id"] < tree.token["id"]:
                left.append(t)
            elif t.token["id"] > tree.token["id"]:
                right.append(t)
            else:
                raise ValueError

        if left == []:
            log_prob = log_prob + self.log_stop_left[1, :, 1]
        else:
            for i, l in enumerate(left[::-1]):
                left_prob = self._calc_log_prob(l, constant, embed_s)

                # (num_state, num_state) --> (num_state)
                left_prob = self.log_attach_left + left_prob.unsqueeze(0)
                left_prob = log_sum_exp(left_prob, dim=1)

                # valence
                log_prob = log_prob + left_prob + self.log_stop_left[0, :, int(i==0)]

            log_prob = log_prob + self.log_stop_left[1, :, 0]


        if right == []:
            log_prob = log_prob + self.log_stop_right[1, :, 1]
        else:
            for i, r in enumerate(right):
                right_prob = self._calc_log_prob(r, constant, embed_s)

                # (num_state, num_state) --> (num_state)
                right_prob = self.log_attach_right + right_prob.unsqueeze(0)
                right_prob = log_sum_exp(right_prob, dim=1)

                # valence
                log_prob = log_prob + right_prob + self.log_stop_right[0, :, int(i==0)]

            log_prob = log_prob + self.log_stop_right[1, :, 0]

        return log_prob

    def parse_pos_seq(self, train_data):
        """decode the best latent tag sequences, given
        all the paramters and gold tree structures

        Return: List1

            List1: a list of decoded pos tag ids, format is like
                   train_data.pos
        """
        self.log_attach_left = log_softmax(self.attach_left, dim=1)
        self.log_attach_right = log_softmax(self.attach_right, dim=1)
        self.log_stop_right = log_softmax(self.stop_right, dim=0)
        self.log_stop_left = log_softmax(self.stop_left, dim=0)
        self.log_root_attach_left = log_softmax(self.root_attach_left, dim=0)

        constant = -self.num_dims/2.0 * (math.log(2 * math.pi)) - \
                    0.5 * torch.sum(torch.log(self.var), dim=-1)

        decoded_pos = []
        for embed, tree, gold_pos in zip(train_data.embed, train_data.trees, train_data.postags):
            pos = [0] * len(embed)
            parse_tree = ParseTree(tree, [], [])
            embed_t = torch.tensor(embed, dtype=torch.float32, requires_grad=False, device=self.device)

            if self.args.pos_emb_dim > 0:
                gold_pos = torch.tensor(gold_pos, dtype=torch.long, requires_grad=False, device=self.device)
                pos_embed = self.pos_embed(gold_pos)
                embed_t = torch.cat((embed_t, pos_embed), dim=-1)

            embed_t, _ = self.transform(embed_t.unsqueeze(1))
            embed_t = embed_t.squeeze(1)

            log_prob = self._find_best_path(tree, parse_tree, constant, embed_t)

            log_prob = self.log_root_attach_left + log_prob
            log_prob, root_index = torch.max(log_prob, dim=0)
            self._find_pos_seq(parse_tree, root_index, pos)

            decoded_pos.append(pos)

        return decoded_pos

    def _find_best_path(self, tree, parse_tree, constant, embed_s):
        """decode the most likely latent tag tree given
        current tree and sequence of latent embeddings,
        """

        token_id = tree.token["id"]
        embed = embed_s[token_id-1]

        embed = embed.unsqueeze(0)

        log_prob = constant - 0.5 * torch.sum(
            (self.means - embed)**2 / self.var, dim=1)

        # leaf nodes
        if tree.children == []:
            return log_prob


        left = []
        right = []

        for t in tree.children:
            if t.token["id"] < tree.token["id"]:
                left.append(t)
            elif t.token["id"] > tree.token["id"]:
                right.append(t)
            else:
                raise ValueError

        if left == []:
            log_prob = log_prob + self.log_stop_left[1, :, 1]
        else:
            for i, l in enumerate(left[::-1]):
                parse_tree_tmp = ParseTree(l, [], [])
                left_prob = self._find_best_path(l, parse_tree_tmp, constant, embed_s)

                # (num_state, num_state) --> (num_state)
                left_prob = self.log_attach_left + left_prob.unsqueeze(0)
                left_prob, left_prob_index = torch.max(left_prob, dim=1)

                # valence
                log_prob = log_prob + left_prob + self.log_stop_left[0, :, int(i==0)]

                parse_tree.children.append(parse_tree_tmp)
                parse_tree.decode_tag.append(left_prob_index)

            log_prob = log_prob + self.log_stop_left[1, :, 0]


        if right == []:
            log_prob = log_prob + self.log_stop_right[1, :, 1]
        else:
            for i, r in enumerate(right):
                parse_tree_tmp = ParseTree(r, [], [])
                right_prob = self._find_best_path(r, parse_tree_tmp, constant, embed_s)

                # (num_state, num_state) --> (num_state)
                right_prob = self.log_attach_right + right_prob.unsqueeze(0)
                right_prob, right_prob_index = torch.max(right_prob, dim=1)

                # valence
                log_prob = log_prob + right_prob + self.log_stop_right[0, :, int(i==0)]
                parse_tree.children.append(parse_tree_tmp)
                parse_tree.decode_tag.append(right_prob_index)

            log_prob = log_prob + self.log_stop_right[1, :, 0]

        return log_prob

    def _find_pos_seq(self, parse_tree, head_index, pos):
        """decode the pos sequence, results are stored in pos
        """

        token_id = parse_tree.tree.token["id"]
        pos[token_id-1] = head_index

        for child, decode_tag in zip(parse_tree.children, parse_tree.decode_tag):
            head_index_child = decode_tag[head_index]
            self._find_pos_seq(child, head_index_child, pos)

    # def supervised_loss(self, sents, iter_obj):

    #     """
    #     Args:
    #         sents: A tensor with size (batch_size, seq_len, features)
    #         pos: (seq_len, batch_size)
    #         head: (seq_len, batch_size)
    #         num_left_child: (seq_len, batch_size)
    #         num_right_child: (seq_len, batch_size)
    #         masks: (seq_len, batch_size)
    #     """
    #     pos = iter_obj.pos
    #     head = iter_obj.head
    #     num_left_child = iter_obj.l_deps
    #     num_right_child = iter_obj.r_deps
    #     masks = iter_obj.mask

    #     seq_len, batch_size = pos.size()

    #     attach_left_ = log_softmax(self.attach_left, dim=1).expand(batch_size, *self.attach_left.size())
    #     attach_right_ = log_softmax(self.attach_right, dim=1).expand(batch_size, *self.attach_right.size())
    #     root_attach_ = log_softmax(self.root_attach_left, dim=0).expand(batch_size, *self.root_attach_left.size())
    #     stop_right_s = log_softmax(self.stop_right, dim=0).expand(batch_size, *self.stop_right.size())
    #     stop_left_s = log_softmax(self.stop_left, dim=0).expand(batch_size, *self.stop_left.size())

    #     # (batch_size, num_state, 2)
    #     stop_right_ = stop_right_s[:, 1, :, :]
    #     stop_left_ = stop_left_s[:, 1, :, :]
    #     continue_right_ = stop_right_s[:, 0, :, :]
    #     continue_left_ = stop_left_s[:, 0, :, :]

    #     # (batch_size, seq_len)
    #     pos_t = pos.transpose(0, 1)
    #     density = self._eval_log_density_supervised(sents, pos_t)

    #     log_emission_prob = torch.mul(density, masks.transpose(0, 1)).sum()

    #     for i in range(seq_len):
    #         # 1 indicates left dependent
    #         dir_left = (i < head[i]).float()

    #         # (batch_size, 1)
    #         pos_sub = pos[i].unsqueeze(1)


    #         head_mask = (head[i] >= 0).long()
    #         head_index = torch.mul(head_mask, head[i])
    #         # (batch_size, 1, num_state)
    #         head_pos_sub = torch.gather(pos, index=head_index.unsqueeze(0), dim=0).squeeze(0) \
    #                             .view(batch_size, 1, 1).expand(batch_size, 1, self.num_state)

    #         # attach prob
    #         # (batch_size, num_state) --> (batch_size)
    #         log_attach_left_prob = torch.gather(attach_left_, index=head_pos_sub, dim=1).squeeze(1)
    #         log_attach_left_prob = torch.gather(log_attach_left_prob, index=pos_sub, dim=1).squeeze(1)

    #         log_attach_right_prob = torch.gather(attach_right_, index=head_pos_sub, dim=1).squeeze(1)
    #         log_attach_right_prob = torch.gather(log_attach_right_prob, index=pos_sub, dim=1).squeeze(1)

    #         log_attach = torch.mul(dir_left, log_attach_left_prob) + torch.mul(1.0 - dir_left, log_attach_right_prob)

    #         log_attach = torch.mul(log_attach, head_mask.float())

    #         # 1 indicates root
    #         dir_root = (head[i] == -1).float()
    #         log_root_prob = torch.gather(root_attach_, index=pos_sub, dim=1).squeeze(1)
    #         log_attach = log_attach + torch.mul(dir_root, log_root_prob)

    #         log_attach = torch.mul(log_attach, masks[i])

    #         log_prob = log_emission_prob + log_attach.sum()

    #         # stop prob
    #         # (batch_size, num_state, 1), 1 indicates no child
    #         stop_adj_left = (num_left_child[i] == 0).long().view(batch_size, 1, 1).expand(batch_size, self.num_state, 1)
    #         stop_adj_right = (num_right_child[i] == 0).long().view(batch_size, 1, 1).expand(batch_size, self.num_state, 1)

    #         # (batch_size, num_state) --> (batch_size)
    #         log_stop_right_prob = torch.gather(stop_right_, index=stop_adj_right, dim=2).squeeze(2)
    #         log_stop_right_prob = torch.gather(log_stop_right_prob, index=pos_sub, dim=1).squeeze(1)
    #         log_stop_left_prob = torch.gather(stop_left_, index=stop_adj_left, dim=2).squeeze(2)
    #         log_stop_left_prob = torch.gather(log_stop_left_prob, index=pos_sub, dim=1).squeeze(1)

    #         log_stop = torch.mul(log_stop_right_prob + log_stop_left_prob, masks[i])

    #         log_prob = log_prob + log_stop.sum()

    #         # continue prob, 1 represents the existence of continue prob
    #         pos_sub_ = pos_sub.unsqueeze(2).expand(batch_size, 1, 2)

    #         # (batch_size, 2)
    #         continue_right_sub = torch.gather(continue_right_, index=pos_sub_, dim=1).squeeze(1)
    #         continue_left_sub = torch.gather(continue_left_, index=pos_sub_, dim=1).squeeze(1)

    #         # (batch_size)
    #         continue_flag_left = (num_left_child[i] > 0)
    #         continue_flag_right = (num_right_child[i] > 0)

    #         continue_flag_left = continue_flag_left.float()
    #         continue_flag_right = continue_flag_right.float()

    #         log_continue_right_prob = torch.mul(continue_right_sub[:,1], continue_flag_right)
    #         log_continue_left_prob = torch.mul(continue_left_sub[:,1], continue_flag_left)

    #         log_continue_right_prob = log_continue_right_prob + \
    #             torch.mul(continue_flag_right, torch.mul((num_right_child[i]-1).float(), continue_right_sub[:,0]))
    #         log_continue_left_prob = log_continue_left_prob + \
    #             torch.mul(continue_flag_left, torch.mul((num_left_child[i]-1).float(), continue_left_sub[:,0]))

    #         log_continue = torch.mul(log_continue_left_prob + log_continue_right_prob, masks[i])

    #         log_prob = log_prob + log_continue.sum()


    #     return -log_prob


    def unsupervised_loss(self, iter_obj):
        """
        Args:
            sents: A tensor with size (batch_size, seq_length, features)
            masks: (seq_length, batch_size)

        Variable clarification:
            p_inside[i, j] is the prob of w_i, w_i+1, ..., w_j-1
            rooted at any possible nonterminals

        node marks clarification:
            0: no marks (right first)
            1: right stop mark
            2: both left and right stop marks

        """

        # normalizing parameters
        self.log_attach_left = log_softmax(self.attach_left, dim=1)
        self.log_attach_right = log_softmax(self.attach_right, dim=1)
        self.log_stop_right = log_softmax(self.stop_right, dim=0)
        self.log_stop_left = log_softmax(self.stop_left, dim=0)
        self.log_root_attach_left = log_softmax(self.root_attach_left, dim=0)

        sents = iter_obj.embed
        masks = iter_obj.mask
        pos_t = iter_obj.pos

        if self.args.pos_emb_dim > 0:
            pos_embed = self.pos_embed(pos_t)
            sents = torch.cat((sents, pos_embed), dim=-1)

        sents, jacob = self.transform(sents, masks)

        sents = sents.transpose(0, 1)

        # (batch_size, seq_length, num_state)
        density = self._eval_log_density(sents)

        # indexed by (start, end, mark)
        # each element is a tensor with size (batch_size, num_state, seq_length)
        self.log_p_inside = {}
        # n = len(s)

        batch_size, seq_length, _ = sents.size()

        for i in range(seq_length):
            j = i + 1
            cat_var = [torch.zeros((batch_size, self.num_state, 1),
                    dtype=torch.float32,
                    device=self.device).fill_(NEG_INFINITY) for _ in range(seq_length)]

            cat_var[i] = density[:, i, :].unsqueeze(dim=2)
            self.log_p_inside[i, j, 0] = torch.cat(cat_var, dim=2)
            self.unary_p_inside(i, j, batch_size, seq_length)

        log_stop_right = self.log_stop_right[0]
        log_stop_left = self.log_stop_left[0]

        #TODO(junxian): ideally, only the l loop is needed
        # but eliminate the rest loops would be a bit hard
        for l in range(2, seq_length+1):
            for i in range(seq_length-l+1):
                j = i + l
                log_p1 = []
                log_p2 = []
                index = torch.zeros((seq_length, j-i-1), dtype=torch.long,
                    device=self.device, requires_grad=False)
                # right attachment
                for k in range(i+1, j):

                    log_p1.append(self.log_p_inside[i, k, 0].unsqueeze(-1))
                    log_p2.append(self.log_p_inside[k, j, 2].unsqueeze(-1))
                    index[k-1, k-i-1] = 1

                log_p1 = torch.cat(log_p1, dim=-1)
                log_p2 = torch.cat(log_p2, dim=-1)
                index = index.unsqueeze(0).expand(self.num_state, *index.size())

                # (num_state, seq_len, k)
                log_stop_right_gather = torch.gather(
                    log_stop_right.unsqueeze(-1).expand(*log_stop_right.size(), j-i-1),
                    1, index)

                # log_p_tmp[b, i, m, j, n] = log_p1[b, i, m] + log_p2[b, j, n] + stop_right[0, i, m==k-1]
                # + attach_right[i, j]
                # log_p_tmp = log_p1_ep + log_p2_ep + log_attach_right + log_stop_right_gather

                # to save memory, first marginalize out j and n
                # (b, i, j, k) -> (b, i, k)
                log_p2_tmp = log_sum_exp(log_p2.unsqueeze(1), dim=3) + \
                             self.log_attach_right.view(1, *(self.log_attach_right.size()), 1)
                log_p2_tmp = log_sum_exp(log_p2_tmp, dim=2)

                # (b, i, m, k)
                log_p_tmp = log_p1 + log_p2_tmp.unsqueeze(2) + \
                            log_stop_right_gather.unsqueeze(0)

                self.log_p_inside[i, j, 0] = log_sum_exp(log_p_tmp, dim=-1)

                # left attachment
                log_p1 = []
                log_p2 = []
                index = torch.zeros((seq_length, j-i-1), dtype=torch.long,
                    device=self.device, requires_grad=False)
                for k in range(i+1, j):

                    log_p1.append(self.log_p_inside[i, k, 2].unsqueeze(-1))
                    log_p2.append(self.log_p_inside[k, j, 1].unsqueeze(-1))
                    index[k, k-i-1] = 1

                log_p1 = torch.cat(log_p1, dim=-1)
                log_p2 = torch.cat(log_p2, dim=-1)
                index = index.unsqueeze(0).expand(self.num_state, *index.size())

                log_stop_left_gather = torch.gather(
                    log_stop_left.unsqueeze(-1).expand(*log_stop_left.size(), j-i-1),
                    1, index)

                # log_p_tmp[b, i, m, j, n] = log_p1[b, i, m] + log_p2[b, j, n] + stop_left[0, j, n==k]
                # + self.attach_left[j, i]

                # to save memory, first marginalize out j and n
                # (b, i, j, k) -> (b, j, k)
                log_p1_tmp = log_sum_exp(log_p1.unsqueeze(2), dim=3) + \
                             self.log_attach_left.permute(1, 0).view(1, *(self.log_attach_left.size()), 1)
                log_p1_tmp = log_sum_exp(log_p1_tmp, dim=1)

                # (b, j, n, k)
                log_p_tmp = log_p1_tmp.unsqueeze(2) + log_p2 + \
                            log_stop_left_gather.unsqueeze(0)
                self.log_p_inside[i, j, 1] = log_sum_exp(log_p_tmp, dim=-1)

                self.unary_p_inside(i, j, batch_size, seq_length)


        # calculate log likelihood
        sent_len_t = masks.sum(dim=0).detach()
        log_p_sum = []
        for i in range(batch_size):
            sent_len = sent_len_t[i].item()
            log_p_sum += [self.log_p_inside[0, sent_len, 2][i].unsqueeze(dim=0)]
        log_p_sum_cat = torch.cat(log_p_sum, dim=0)

        log_root = log_p_sum_cat + self.log_root_attach_left.view(1, self.num_state, 1) \
                   .expand_as(log_p_sum_cat)

        return -torch.sum(log_sum_exp(log_root.view(batch_size, -1), dim=1)), jacob


    def dep_parse(self, sents, iter_obj, symbol_index_t):
        """
        Args:
            sents: tensor with size (batch_size, seq_length, features)
        Returns:
            returned t is a nltk.tree.Tree without root node
        """

        masks = iter_obj.mask
        gold_pos = iter_obj.pos

        # normalizing parameters
        self.log_attach_left = self.args.prob_const * log_softmax(self.attach_left, dim=1)
        self.log_attach_right = self.args.prob_const * log_softmax(self.attach_right, dim=1)
        self.log_stop_right = self.args.prob_const * log_softmax(self.stop_right, dim=0)
        self.log_stop_left = self.args.prob_const * log_softmax(self.stop_left, dim=0)
        self.log_root_attach_left = self.args.prob_const * log_softmax(self.root_attach_left, dim=0)

        # (batch_size, seq_length, num_state)
        density = self._eval_log_density(sents)

        # evaluate with gold pos tag
        # batch_size, seq_len, _ = sents.size()
        # density = torch.zeros((batch_size, seq_len, self.num_state), device=self.device,
        #         requires_grad=False).fill_(NEG_INFINITY)
        # for b in range(batch_size):
        #     for s in range(seq_len):
        #         density[b, s, gold_pos[s, b]] = 0.



        # in the parse case, log_p_parse[i, j, mark] is not the log prob
        # of some symbol as head, instead it is the prob of the most likely
        # subtree with some symbol as head
        self.log_p_parse = {}

        # child is indexed by (i, j, mark), and each element is a
        # LongTensor with size (batch_size, symbol, seq_length, 6)
        # the last dimension represents the child's
        # (indicator, i, j, mark, symbol, index), used to index the child,
        # indicator is 1 represents childs exist, 0 not exist, -1 means
        # reaching terminal symbols. For unary connection, left child indicator
        # is 1 and right child indicator is 0 (for non-terminal symbols)
        self.left_child = {}
        self.right_child = {}

        batch_size, seq_length, _ = sents.size()


        for i in range(seq_length):
            j = i + 1
            cat_var = [torch.zeros((batch_size, self.num_state, 1),
                    dtype=torch.float32,
                    device=self.device).fill_(NEG_INFINITY) for _ in range(seq_length)]
            cat_var[i] = density[:, i, :].unsqueeze(dim=2)
            self.log_p_parse[i, j, 0] = torch.cat(cat_var, dim=2)
            self.left_child[i, j, 0] = torch.zeros((batch_size, self.num_state, seq_length, 6),
                                                    dtype=torch.long,
                                                    device=self.device).fill_(-1)
            self.right_child[i, j, 0] = torch.zeros((batch_size, self.num_state, seq_length, 6),
                                                    dtype=torch.long,
                                                    device=self.device).fill_(-1)
            self.unary_parses(i, j, batch_size, seq_length, symbol_index_t)

        log_stop_right = self.log_stop_right[0]
        log_stop_left = self.log_stop_left[0]

        # ideally, only the l loop is needed
        # but eliminate the rest loops would be a bit hard
        for l in range(2, seq_length+1):
            for i in range(seq_length-l+1):
                j = i + l

                # right attachment
                log_p1 = []
                log_p2 = []
                index = torch.zeros((seq_length, j-i-1), dtype=torch.long,
                    device=self.device, requires_grad=False)
                for k in range(i+1, j):

                    # right attachment
                    log_p1.append(self.log_p_parse[i, k, 0].unsqueeze(-1))
                    log_p2.append(self.log_p_parse[k, j, 2].unsqueeze(-1))
                    index[k-1, k-i-1] = 1

                log_p1 = torch.cat(log_p1, dim=-1)
                log_p2 = torch.cat(log_p2, dim=-1)
                index = index.unsqueeze(0).expand(self.num_state, *index.size())

                # (num_state, seq_len, k)
                log_stop_right_gather = torch.gather(
                    log_stop_right.unsqueeze(-1).expand(*log_stop_right.size(), j-i-1),
                    1, index)

                # log_p2_tmp: (b, j, k)
                # max_index_loc: (b, j, n)
                log_p2_tmp, max_index_loc = torch.max(log_p2, 2)

                # log_p2_tmp: (b, i, k)
                # max_index_symbol: (b, i, k)
                log_p2_tmp, max_index_symbol = torch.max(log_p2_tmp.unsqueeze(1) +
                    self.log_attach_right.view(1, *(self.log_attach_right.size()), 1), 2)

                # (b, i, m, k)
                log_p_tmp = log_p1 + log_p2_tmp.unsqueeze(2) + log_stop_right_gather.unsqueeze(0)

                # log_p_max: (batch_size, num_state, seq_length)
                # max_index_k: (batch_size, num_state, seq_length)
                log_p_max, max_index_k = torch.max(log_p_tmp, dim=-1)
                self.log_p_parse[i, j, 0] = log_p_max

                # (b, j, k) --> (b, i, k)
                max_index_loc = torch.gather(max_index_loc, index=max_index_symbol, dim=1)

                # (b, i, k) --> (b, i, m)
                max_index_symbol = torch.gather(max_index_symbol, index=max_index_k, dim=2)
                max_index_loc = torch.gather(max_index_loc, index=max_index_k, dim=2)

                # (batch_size, num_state, seq_len, 3)
                max_index_r = torch.cat((max_index_k.unsqueeze(-1),
                                         max_index_symbol.unsqueeze(-1),
                                         max_index_loc.unsqueeze(-1)), dim=-1)


                # left attachment
                log_p1 = []
                log_p2 = []
                index = torch.zeros((seq_length, j-i-1), dtype=torch.long,
                    device=self.device, requires_grad=False)
                for k in range(i+1, j):

                    log_p1.append(self.log_p_parse[i, k, 2].unsqueeze(-1))
                    log_p2.append(self.log_p_parse[k, j, 1].unsqueeze(-1))
                    index[k, k-i-1] = 1

                log_p1 = torch.cat(log_p1, dim=-1)
                log_p2 = torch.cat(log_p2, dim=-1)
                index = index.unsqueeze(0).expand(self.num_state, *index.size())

                # (num_state, seq_len, k)
                log_stop_left_gather = torch.gather(
                    log_stop_left.unsqueeze(-1).expand(*log_stop_left.size(), j-i-1),
                    1, index)

                # log_p1_tmp: (b, i, k)
                # max_index_loc: (b, i, k)
                log_p1_tmp, max_index_loc = torch.max(log_p1, 2)

                # log_p1_tmp: (b, j, k)
                # max_index_symbol: (b, j, k)
                log_p1_tmp, max_index_symbol = torch.max(log_p1_tmp.unsqueeze(2) +
                    self.log_attach_left.permute(1, 0).view(1, *(self.log_attach_left.size()), 1), 1)

                # (b, j, n, k)
                log_p_tmp = log_p1_tmp.unsqueeze(2) + log_p2 + log_stop_left_gather.unsqueeze(0)


                # log_p_max: (batch_size, num_state, seq_length)
                # max_index_k: (batch_size, num_state, seq_length)
                log_p_max, max_index_k = torch.max(log_p_tmp, dim=-1)
                self.log_p_parse[i, j, 1] = log_p_max

                # (b, i, k) --> (b, j, k)
                max_index_loc = torch.gather(max_index_loc, index=max_index_symbol, dim=1)

                # (b, j, k) --> (b, j, m)
                max_index_symbol = torch.gather(max_index_symbol, index=max_index_k, dim=2)
                max_index_loc = torch.gather(max_index_loc, index=max_index_k, dim=2)

                # (batch_size, num_state, seq_len, 3)
                max_index_l = torch.cat((max_index_k.unsqueeze(-1),
                                         max_index_symbol.unsqueeze(-1),
                                         max_index_loc.unsqueeze(-1)), dim=-1)

                right_child_index_r = index.new(batch_size, self.num_state, seq_length, 6)
                left_child_index_r = index.new(batch_size, self.num_state, seq_length, 6)
                right_child_index_l = index.new(batch_size, self.num_state, seq_length, 6)
                left_child_index_l = index.new(batch_size, self.num_state, seq_length, 6)
                # assign symbol and index
                right_child_index_r[:, :, :, 4:] = max_index_r[:, :, :, 1:]

                # left_child_symbol_index: (num_state, seq_length, 2)
                left_child_symbol_index_r = symbol_index_t

                left_child_index_r[:, :, :, 4:] = left_child_symbol_index_r

                right_child_symbol_index_l = symbol_index_t

                right_child_index_l[:, :, :, 4:] = right_child_symbol_index_l
                left_child_index_l[:, :, :, 4:] = max_index_l[:, :, :, 1:]

                # assign indicator
                right_child_index_r[:, :, :, 0] = 1
                left_child_index_r[:, :, :, 0] = 1

                right_child_index_l[:, :, :, 0] = 1
                left_child_index_l[:, :, :, 0] = 1

                # assign starting point
                right_child_index_r[:, :, :, 1] = max_index_r[:, :, :, 0] + i + 1
                left_child_index_r[:, :, :, 1] = i

                right_child_index_l[:, :, :, 1] = max_index_l[:, :, :, 0] + i + 1
                left_child_index_l[:, :, :, 1] = i

                # assign end point
                right_child_index_r[:, :, :, 2] = j
                left_child_index_r[:, :, :, 2] = max_index_r[:, :, :, 0] + i + 1

                right_child_index_l[:, :, :, 2] = j
                left_child_index_l[:, :, :, 2] = max_index_l[:, :, :, 0] + i + 1

                right_child_index_r[:, :, :, 3] = 2
                left_child_index_r[:, :, :, 3] = 0

                right_child_index_l[:, :, :, 3] = 1
                left_child_index_l[:, :, :, 3] = 2

                assert (i, j, 0) not in self.left_child
                self.left_child[i, j, 0] = left_child_index_r
                self.right_child[i, j, 0] = right_child_index_r

                self.left_child[i, j, 1] = left_child_index_l
                self.right_child[i, j, 1] = right_child_index_l

                self.unary_parses(i, j, batch_size, seq_length, symbol_index_t)

        log_p_sum = []
        sent_len_t = masks.sum(dim=0)
        for i in range(batch_size):
            sent_len = sent_len_t[i].item()
            log_p_sum += [self.log_p_parse[0, sent_len, 2][i].unsqueeze(dim=0)]
        log_p_sum_cat = torch.cat(log_p_sum, dim=0)
        log_root = log_p_sum_cat + self.log_root_attach_left.view(1, self.num_state, 1) \
                   .expand_as(log_p_sum_cat)
        log_root_max, root_max_index = torch.max(log_root.view(batch_size, -1), dim=1)

        # (batch_size, 2)
        root_max_index = unravel_index(root_max_index, (self.num_state, seq_length))

        return root_max_index

    def unary_p_inside(self, i, j, batch_size, seq_length):

        non_stop_mark = self.log_p_inside[i, j, 0]
        log_stop_left = self.log_stop_left[1].expand(batch_size, self.num_state, 2)
        log_stop_right = self.log_stop_right[1].expand(batch_size, self.num_state, 2)

        index_ladj = torch.zeros((batch_size, self.num_state, seq_length),
                dtype=torch.long,
                device=self.device,
                requires_grad=False)
        index_radj = torch.zeros((batch_size, self.num_state, seq_length),
                dtype=torch.long,
                device=self.device,
                requires_grad=False)

        index_ladj[:, :, i].fill_(1)
        index_radj[:, :, j-1].fill_(1)

        log_stop_right = torch.gather(log_stop_right, 2, index_radj)
        inter_right_stop_mark = non_stop_mark + log_stop_right

        if (i, j, 1) in self.log_p_inside:
            right_stop_mark = self.log_p_inside[i, j, 1]
            right_stop_mark = torch.cat((right_stop_mark.unsqueeze(dim=3), \
                                    inter_right_stop_mark.unsqueeze(dim=3)), \
                                    dim=3)
            right_stop_mark = log_sum_exp(right_stop_mark, dim=3)

        else:
            right_stop_mark = inter_right_stop_mark



        log_stop_left = torch.gather(log_stop_left, 2, index_ladj)
        self.log_p_inside[i, j, 2] = right_stop_mark + log_stop_left
        self.log_p_inside[i, j, 1] = right_stop_mark

    def unary_parses(self, i, j, batch_size, seq_length, symbol_index_t):
        non_stop_mark = self.log_p_parse[i, j, 0]
        log_stop_left = self.log_stop_left[1].expand(batch_size, self.num_state, 2)
        log_stop_right = self.log_stop_right[1].expand(batch_size, self.num_state, 2)

        index_ladj = torch.zeros((batch_size, self.num_state, seq_length),
                dtype=torch.long,
                device=self.device,
                requires_grad=False)
        index_radj = torch.zeros((batch_size, self.num_state, seq_length),
                dtype=torch.long,
                device=self.device,
                requires_grad=False)

        left_child_index_mark2 = index_ladj.new(batch_size, self.num_state, seq_length, 6)
        right_child_index_mark2 = index_ladj.new(batch_size, self.num_state, seq_length, 6)
        left_child_index_mark1 = index_ladj.new(batch_size, self.num_state, seq_length, 6)
        right_child_index_mark1 = index_ladj.new(batch_size, self.num_state, seq_length, 6)

        index_ladj[:, :, i].fill_(1)
        index_radj[:, :, j-1].fill_(1)

        log_stop_right = torch.gather(log_stop_right, 2, index_radj)
        inter_right_stop_mark = non_stop_mark + log_stop_right

        # assign indicator
        left_child_index_mark1[:, :, :, 0] = 1
        right_child_index_mark1[:, :, :, 0] = 0

        # assign mark
        left_child_index_mark1[:, :, :, 3] = 0
        right_child_index_mark1[:, :, :, 3] = 0

        # start point
        left_child_index_mark1[:, :, :, 1] = i
        right_child_index_mark1[:, :, :, 1] = i

        # end point
        left_child_index_mark1[:, :, :, 2] = j
        right_child_index_mark1[:, :, :, 2] = j

        # assign symbol and index
        left_child_symbol_index_mark1 = symbol_index_t
        left_child_index_mark1[:, :, :, 4:] = left_child_symbol_index_mark1
        right_child_index_mark1[:, :, :, 4:] = left_child_symbol_index_mark1

        if (i, j, 1) in self.log_p_parse:
            right_stop_mark = self.log_p_parse[i, j, 1]

            # max_index (batch_size, num_state, index) (value is 0 or 1)
            right_stop_mark, max_index = torch.max(torch.cat((right_stop_mark.unsqueeze(dim=3), \
                                    inter_right_stop_mark.unsqueeze(dim=3)), \
                                    dim=3), dim=3)

            # mask: (batch_size, num_state, index)
            mask = (max_index == 1)
            mask_ep = mask.unsqueeze(dim=-1).expand(batch_size, self.num_state, seq_length, 6)
            left_child_index_mark1 = self.left_child[i, j, 1].masked_fill_(mask_ep, 0) + \
                                     left_child_index_mark1.masked_fill_(1 - mask_ep, 0)
            right_child_index_mark1 = self.right_child[i, j, 1].masked_fill_(mask_ep, 0) + \
                                     right_child_index_mark1.masked_fill_(1 - mask_ep, 0)


        else:
            right_stop_mark = inter_right_stop_mark

        log_stop_left = torch.gather(log_stop_left, 2, index_ladj)
        self.log_p_parse[i, j, 2] = right_stop_mark + log_stop_left
        self.log_p_parse[i, j, 1] = right_stop_mark


        # assign indicator
        left_child_index_mark2[:, :, :, 0] = 1
        right_child_index_mark2[:, :, :, 0] = 0

        # assign starting point
        left_child_index_mark2[:, :, :, 1] = i
        right_child_index_mark2[:, :, :, 1] = i


        # assign end point
        left_child_index_mark2[:, :, :, 2] = j
        right_child_index_mark2[:, :, :, 2] = j


        # assign mark
        left_child_index_mark2[:, :, :, 3] = 1
        right_child_index_mark2[:, :, :, 3] = 1

        # assign symbol and index
        left_child_symbol_index_mark2 = symbol_index_t
        left_child_index_mark2[:, :, :, 4:] = left_child_symbol_index_mark2
        right_child_index_mark2[:, :, :, 4:] = left_child_symbol_index_mark2


        self.left_child[i, j, 2] = left_child_index_mark2
        self.right_child[i, j, 2] = right_child_index_mark2
        self.left_child[i, j, 1] = left_child_index_mark1
        self.right_child[i, j, 1] = right_child_index_mark1
