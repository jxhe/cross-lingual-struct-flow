"""
This script computes pretrained BERT representation
for the dataset

"""

import os
import importlib
import argparse
import h5py
import torch
import numpy as np

from tqdm import tqdm
from modules import ConlluData
from pytorch_pretrained_bert import BertTokenizer, BertModel

def save_emb_to_hdf5(data, fname, tokenizer, model, device):
    print("writing {}\n".format(fname))
    with h5py.File(fname, "w") as fout:
        for id_, sent in tqdm(enumerate(data.text)):
            bert_tokens = "[CLS]"
            orig_to_tok_map = []
            for token in sent:
                orig_to_tok_map.append(len(bert_tokens))
                bert_tokens.extend(tokenizer.tokenize(token))
            bert_tokens.append("[SEP]")
            indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
            tokens_tensor = torch.tensor([indexed_tokens], device=device)
            encoded_layers, _ = model(tokens_tensor)

            # use the last layer of embedding
            emb = encoded_layers[-1]

            # (seq_len, nfeatures)
            emb = emb[orig_to_tok_map].cpu().numpy()

            fout.create_dataset(str(id_), emb.shape, dtype="float32", data=emb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lang", type=str, help="")
    parser.add_argument("--bert", type=str, default="bert-base-multilingual-cased")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    config_file = "config.config_{}".format(args.lang)
    params = importlib.import_module(config_file).params_markov
    args = argparse.Namespace(**vars(args), **params)

    save_dir = "{}/{}".format(args.bert, args.lang)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if args.cuda else "cpu")

    train_data = ConlluData(args.train_file, None, device=device)
    val_data = ConlluData(args.val_file, None, device=device)
    test_data = ConlluData(args.test_file, None, device=device)

    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=False)
    model = BertModel.from_pretrained(args.bert)
    model.eval()
    model.to(device)

    with torch.no_grad():
        save_emb_to_hdf5(train_data, os.path.join(save_dir, "{}_train.hdf5".format(args.lang)), 
            tokenizer, model, device)
        save_emb_to_hdf5(val_data, os.path.join(save_dir, "{}_dev.hdf5".format(args.lang)), 
            tokenizer, model, device)
        save_emb_to_hdf5(test_data, os.path.join(save_dir, "{}_test.hdf5".format(args.lang)), 
            tokenizer, model, device)

