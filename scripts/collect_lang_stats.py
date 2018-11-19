"""
This script collects language statistics from UD treebank dataset

"""

import os

from io import open
from conllu import parse_incr

lang_stats = []

for root, subdirs, files in os.walk("ud-treebanks-v2.2"):
    train_flag = False
    valid_dir = False

    for fname in files:
        if fname.strip().split('.')[-1] == "conllu":
            valid_dir = True
            lang = fname.strip().split('.')[0].split('_')[0]
            break

    if valid_dir:
        for fname in files:
            if fname.strip().split('.')[-1] == "conllu":  
                train = fname.strip().split('.')[0].split('-')[-1]
                if train != "train":
                    continue
                train_flag = True
                fin = open(os.path.join(root, fname), "r", encoding="utf-8")
                sents = list(parse_incr(fin))
                lang_stats.append((lang, root, len(sents)))
                break

        # no training file
        if not train_flag:
            lang_stats.append((lang, root, 0))

with open("lang_stats.txt", "w") as fout:
    for name_root_value in sorted(lang_stats, key=lambda name_root_value: name_root_value[2]):
        name, root, value = name_root_value
        fout.write("{} {} {}\n".format(name, root, value))
