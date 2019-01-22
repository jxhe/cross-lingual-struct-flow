"""
This script pre-processes the conllu files and convert them
to the required format of the tagger
"""
import os

from io import open
from conllu import parse_incr

for root, subdirs, files in os.walk("ud-treebanks-v2.2"):
    for fname in files:
        if fname.strip().split('.')[-1] == "conllu":
            fin = open(os.path.join(root, fname), "r", encoding="utf-8")
            fout_name = fname.split('.')[0] + ".tag"
            fout = open(os.path.join(root, fout_name), "w", encoding="utf-8")
            data_file = parse_incr(fin)
            for sent in data_file:
                for token in sent:
                    if token["upostag"] != "_":
                        fout.write("{} {}\n".format(token["form"], token["upostag"]))
                fout.write("\n")
            fout.close()


