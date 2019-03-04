# This script output to-eng-distance and number of training data
# statistics for each language in UD treebank, according to uriel,
# and then rank them in terms of GEOGRAPHIC distance

import os
import numpy as np
from io import open
from collections import namedtuple
from conllu import parse_incr

LangObj = namedtuple("lang", ["name", "old_id", "new_id", "num_train", "distance", "obj"])

# rank_obj = ["GEOGRAPHIC", "GENETIC", "SYNTACTIC"]
rank_obj = ["GEOGRAPHIC", "GENETIC", "SYNTACTIC", "INVENTORY", "PHONOLOGICAL"]
distance = np.load("uriel_v0_2/distances/distances.npz")

en_id = np.asscalar(np.where(distance["langs"]=="eng")[0])

# map identifier from language name to iso-639-3
fid = open("statistics/iso-639-3.tab", "r", encoding="utf-8")
_ = fid.readline()

lang_to_id = {}

for line in fid:
    line_ = line.strip().split("\t")
    lang_to_id[line_[-1]] = line_[0]

print("complete mapping identifier")

fout = open("statistics/distance_all_allmetrics.txt", "w")

fout.write("LANG\t")
for name in distance["sources"]:
    fout.write(name + "\t")
fout.write("AVG\t")
fout.write("TRAIN\n")

obj_id = [np.asscalar(np.where(distance["sources"]==x)[0]) for x in rank_obj]
res = {}

cnt = 0

for root, subdirs, files in os.walk("ud-treebanks-v2.2"):
    valid_dir = False
    for fname in files:
        if fname.endswith("conllu"):
            valid_dir = True
            lang = root.split("/")[1].split("-")[0].split("_")[1]
            old_id = fname.split("_")[0]
            break

    if valid_dir:
        if lang in res:
            continue

        if lang in lang_to_id:
            identifier = lang_to_id[lang]
        else:
            continue
        tgt_id = np.asscalar(np.where(distance["langs"]==identifier)[0])
        dist = distance["data"][en_id, tgt_id]

        train_num = 0
        # for fname in files:
        #     if fname.endswith("conllu"):
        #         train = fname.strip().split('.')[0].split('-')[-1]
        #         if train != "train":
        #             continue

        #         with open(os.path.join(root, fname), "r", encoding="utf-8") as fdata:
        #             train_num = len(list(parse_incr(fdata)))
        #         break
        dist_new = sum([dist[x] for x in obj_id]) / len(obj_id)
        res[lang] = LangObj(lang, old_id, identifier, train_num, dist, dist_new)
        cnt += 1
        # if cnt == 10:
        #     break

for lang_obj in sorted(res.values(), key=lambda s: -s.obj):
    fout.write("{}/{}\t".format(lang_obj.name, lang_obj.old_id))
    for num in lang_obj.distance:
        fout.write("{:.3f}\t".format(num))
    fout.write("{:.3f}\t".format(lang_obj.obj))
    fout.write("{}\n".format(lang_obj.num_train))

fid.close()
fout.close()

