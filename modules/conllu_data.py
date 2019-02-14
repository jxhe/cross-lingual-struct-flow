import numpy as np
import torch
from collections import defaultdict, namedtuple
from conllu import parse_incr, parse_tree_incr

IterObj = namedtuple("iter_object", ["embed", "pos", "head", "r_deps", "l_deps", "mask"])

class ConlluData(object):
    """docstring for ConlluData"""
    def __init__(self, fname, embed, device,
                 max_len=1e3, pos_to_id_dict=None,
                 read_tree=False):
        super(ConlluData, self).__init__()
        self.device = device

        if pos_to_id_dict is None:
            pos_to_id = defaultdict(lambda: len(pos_to_id))
        else:
            pos_to_id = pos_to_id_dict

        text = []
        tags = []
        trees = []
        heads = []
        right_num_deps = []
        left_num_deps = []
        fin = open(fname, "r", encoding="utf-8")
        fin_tree = open(fname, "r", encoding="utf-8")
        data_file_tree = parse_tree_incr(fin_tree)
        data_file = parse_incr(fin)
        for sent, tree in zip(data_file, data_file_tree):
            sent_list = []
            tag_list = []
            head_list = []
            right_num_deps_ = []
            left_num_deps_ = []
            sent_n = []

            # delete multi-word token
            for token in sent:
                if isinstance(token["id"], int):
                    sent_n += [token]

            for token in sent_n:
                sent_list.append(token["form"])
                pos_id = pos_to_id[token["upostag"]] if token["upostag"] != '_' else pos_to_id["X"]
                tag_list.append(pos_id)
                # -1 represents root
                head_list.append(token["head"]-1)

            if len(tag_list) > max_len:
                continue

            right_num_deps_ = [0] * len(head_list)
            left_num_deps_ = [0] * len(head_list)

            for i, head_id in enumerate(head_list):
                if head_id != -1:
                    if i < head_id:
                        left_num_deps_[head_id] += 1
                    elif i > head_id:
                        right_num_deps_[head_id] += 1
                    else:
                        raise ValueError("head is itself !")

            text.append(sent_list)
            tags.append(tag_list)
            heads.append(head_list)
            right_num_deps.append(right_num_deps_)
            left_num_deps.append(left_num_deps_)
            trees.append(tree)

        self.trees = trees
        self.text = text
        self.postags = tags
        self.heads = heads
        self.right_num_deps = right_num_deps
        self.left_num_deps = left_num_deps
        self.pos_to_id = pos_to_id
        self.id_to_pos = {v:k for (k, v) in pos_to_id.items()}
        self.length = len(self.text)

        self.text_to_embed(embed)

        fin.close()
        fin_tree.close()

    def __len__(self):
        return self.length

    def text_to_embed(self, embedding):
        self.embed = []
        for sent in self.text:
            sample = [embedding[word] if word in embedding else np.zeros(embedding.n_dim) for word in sent]
            self.embed.append(sample)

    def input_transpose(self, embed, pos, head, r_deps, l_deps):
        max_len = max(len(s) for s in pos)
        batch_size = len(embed)
        embed_t = []
        pos_t = []
        head_t = []
        r_deps_t = []
        l_deps_t = []
        masks = []
        pad = embed[0][0]
        for i in range(max_len):
            embed_t.append([embed_s[i] if len(embed_s) > i else pad for embed_s in embed])
            pos_t.append([pos_s[i] if len(pos_s) > i else 0 for pos_s in pos])
            head_t.append([head_s[i] if len(head_s) > i else -2 for head_s in head])
            r_deps_t.append([r_deps_s[i] if len(r_deps_s) > i else 0 for r_deps_s in r_deps])
            l_deps_t.append([l_deps_s[i] if len(l_deps_s) > i else 0 for l_deps_s in l_deps])
            masks.append([1 if len(embed_s) > i else 0 for embed_s in embed])

        return embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks

    def to_input_tensor(self, embed, pos, head, r_deps, l_deps):
        """
        return a tensor of shape (src_sent_len, batch_size)
        """

        embed, pos, head, r_deps, l_deps, masks = self.input_transpose(embed, pos, head, r_deps, l_deps)


        embed_t = torch.tensor(embed, dtype=torch.float32, requires_grad=False, device=self.device)
        pos_t = torch.tensor(pos, dtype=torch.long, requires_grad=False, device=self.device)
        head_t = torch.tensor(head, dtype=torch.long, requires_grad=False, device=self.device)
        r_deps_t = torch.tensor(r_deps, dtype=torch.long, requires_grad=False, device=self.device)
        l_deps_t = torch.tensor(l_deps, dtype=torch.long, requires_grad=False, device=self.device)
        masks_t = torch.tensor(masks, dtype=torch.float32, requires_grad=False, device=self.device)

        return embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t

    def data_iter(self, batch_size, shuffle=True):
        index_arr = np.arange(self.length)
        # in_place operation

        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(self.length / float(batch_size))
        for i in range(batch_num):
            batch_ids = index_arr[i * batch_size: (i + 1) * batch_size]
            batch_embed = []
            batch_pos = []
            batch_head = []
            batch_right_num_deps = []
            batch_left_num_deps = []
            for index in batch_ids:
                batch_embed += [self.embed[index]]
                batch_pos += [self.postags[index]]
                batch_head += [self.heads[index]]
                batch_right_num_deps += [self.right_num_deps[index]]
                batch_left_num_deps += [self.left_num_deps[index]]

            sort_index = list(range(len(batch_ids)))
            len_list = [len(x) for x in batch_pos]
            sort_index.sort(key=lambda s: -len_list[s])
            batch_embed = [batch_embed[x] for x in sort_index]
            batch_pos = [batch_pos[x] for x in sort_index]
            batch_head = [batch_head[x] for x in sort_index]
            batch_right_num_deps = [batch_right_num_deps[x] for x in sort_index]
            batch_left_num_deps = [batch_left_num_deps[x] for x in sort_index]

            embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t = self.to_input_tensor(
                batch_embed, batch_pos, batch_head, batch_right_num_deps, batch_left_num_deps)

            yield IterObj(embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t)

    def data_iter_efficient(self, mem_limit=250):
        """This function batches similar-length sentences together,
        with a memory limit that satisfies batch_size x length <= mem_limit.
        Such batching is only used in test to accelarate evaluation
        """

        sents_len = np.array([len(sent) for sent in self.postags])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        curr = 0
        while curr < len(sort_len):
            batch_data = []
            mem = 0
            next_ = curr
            mem = 0
            cnt = 0
            while next_ < len(sort_len):
                cnt += 1
                mem = cnt * sort_len[next_]
                if mem > mem_limit:
                    break
                next_ += 1

            index_ = [sort_idx[x] for x in range(curr, next_)]
            index_ = index_[::-1]

            curr = next_

            batch_embed = [self.embed[x] for x in index_]
            batch_pos = [self.postags[x] for x in index_]
            batch_head = [self.heads[x] for x in index_]
            batch_right_num_deps = [self.right_num_deps[x] for x in index_]
            batch_left_num_deps = [self.left_num_deps[x] for x in index_]

            embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t = self.to_input_tensor(
                batch_embed, batch_pos, batch_head, batch_right_num_deps, batch_left_num_deps)

            yield IterObj(embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t)

