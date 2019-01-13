from collections import defaultdict, namedtuple
from conllu import parse_incr

IterObj = namedtuple("iter object", ["embed", "pos", "head", "r_deps", "l_deps", "mask"])

class ConlluData(object):
    """docstring for ConlluData"""
    def __init__(self, fname, embed, device, 
                 max_len=1e3, exclude=["PUNCT", "SYM"],
                 pos_to_id_dict=None):
        super(ConlluData, self).__init__()
        self.device = device

        if pos_to_id_dict is None:
            pos_to_id = defaultdict(lambda: len(pos_to_id))
            
        text = []
        tags = []
        heads = []
        right_num_deps = []
        left_num_deps = []
        fin = open(fname, "r", encoding="utf-8")
        data_file = parse_incr(fin)
        for sent in data_file:
            sent_list = []
            tag_list = []
            head_list = []
            exclude_list = []
            right_num_deps_ = []
            left_num_deps_ = []
            for token in sent:
                if token["upostag"] not in exclude:
                    sent_list.append(token["form"])
                    pos_id = pos_to_id[token["upostag"]] if token["upostag"] != '_' else pos_to_id["X"]
                    tag_list.append(pos_id)
                    # -1 represents root
                    head_list.append(token["head"]-1)
                else:
                    exclude_list += [token["id"]-1]

            if len(tag_list) > max_len:
                continue

            # recompute head id for current sequence
            head_list_copy = head_list.copy()
            for exclude_id in exclude_list:
                for i, head_id in enumerate(head_list_copy):
                    assert(head_id != exclude_id)
                    if exclude_id < head_id:
                        head_list[i] -= 1

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
            
        self.text = text
        self.postags = tags
        self.heads = heads
        self.right_num_deps = right_num_deps
        self.left_num_deps = left_num_deps
        self.pos_to_id = pos_to_id
        self.id_to_pos = {v:k for (k, v) in pos_to_id.items()}
        self.length = len(self.text)

        self.text_to_embed(embed)


    def __len__(self):
        return self.length

    def text_to_embed(self, embedding):
        self.embed = []
        for sent in self.text:
            sample = [embedding[word] for word in sent]
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

        embed, pos, head, r_deps, l_deps, masks = input_transpose(embed, pos, head, r_deps, l_deps)


        embed_t = torch.tensor(embed, dtype=torch.float32, requires_grad=False, device=self.device)
        pos_t = torch.tensor(pos, dtype=torch.long, requires_grad=False, device=self.device)
        head_t = torch.tensor(head, dtype=torch.long, requires_grad=False, device=self.device)
        r_deps_t = torch.tensor(r_deps, dtype=torch.long, requires_grad=False, device=self.device)
        l_deps_t = torch.tensor(l_deps, dtype=torch.long, requires_grad=False, device=self.device)
        masks_t = torch.tensor(masks, dtype=torch.float32, requires_grad=False, device=self.device)

        return sents_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t

    def data_iter(self, batch_size, shuffle=True):
        index_arr = np.arange(self.length)
        # in_place operation

        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for i in range(batch_num):
            batch_ids = index_arr[i * batch_size: (i + 1) * batch_size]
            batch_embed = []
            batch_pos = []
            batch_head = []
            batch_right_num_deps = []
            batch_left_num_deps = []
            for index in batch_ids:
                batch_embed += [self.embeddings[index]]
                batch_pos += [self.postags[index]]
                batch_head += [self.heads[index]]
                batch_right_num_deps += [self.right_num_deps[index]]
                batch_left_num_deps += [self.left_num_deps[index]]

            embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t = self.to_input_tensor(
                batch_embed, batch_pos, batch_head, batch_right_num_deps, batch_left_num_deps)

            yield IterObj(embed_t, pos_t, head_t, r_deps_t, l_deps_t, masks_t)
