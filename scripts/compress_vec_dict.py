"""
This script compresses the fastText dictionary,
to a new one that only contains the words at dataset.

This new dict is created mainly for debugging
and tuning purpose

"""

import argparse
import importlib
from io import open

from modules import *
from multilingual_trans.fasttext import FastVector

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lang', type=str, help='')

args = parser.parse_args()

config_file = "config.config_{}".format(args.lang)
params = importlib.import_module(config_file).params
args = argparse.Namespace(**vars(args), **params)

word_vec_dict = FastVector(vector_file="fastText_data/wiki.{}.vec".format(args.lang))
ndim = word_vec_dict.n_dim

train_text, train_tags = read_conll(args.train_file)
val_text, val_tags = read_conll(args.val_file)
test_text, test_tags = read_conll(args.test_file)

vocab = set()
_ = [[vocab.update([word]) for word in sent] for sent in train_text]
_ = [[vocab.update([word]) for word in sent] for sent in val_text]
_ = [[vocab.update([word]) for word in sent] for sent in test_text]

print("vocab length {}".format(len(vocab)))

# compute mean
cnt = 0
oov = []
for word in vocab:
    if word in word_vec_dict:
        if cnt == 0:
            sum_ = word_vec_dict[word]
        else:
            sum_ += word_vec_dict[word]
        cnt += 1
    else:
        oov.append(word)
        print("oov: {}".format(word))

print("out of vocab # {}".format(len(oov)))
mean = sum_ / cnt

suffix = args.train_file.split('/')[-1].split('-')[0].split('_')[1]
out_file = "fastText_data/wiki.{}.{}.vec".format(args.lang, suffix)
with open(out_file, "w", encoding="utf-8") as fout:
    fout.write("{} {}\n".format(len(vocab), ndim))
    for word in vocab:
        fout.write("{} ".format(word))
        if word in word_vec_dict:
            for val in word_vec_dict[word]:
                fout.write("{} ".format(val))
        else:
            for val in mean:
                fout.write("{} ".format(val))
        fout.write('\n')
