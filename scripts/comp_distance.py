# This script output to-eng-distance statistics for each language
# in lang_list.txt, according to uriel

import numpy as np
from io import open

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

fquery = open("statistics/lang_list.txt", "r")
fout = open("statistics/distance_query.txt", "w")

fout.write("LANG ")
for name in distance["sources"]:
    fout.write(name + "\t")
fout.write("\n")

for line in fquery:
    lang, old_id, _ = line.split()
    identifier = lang_to_id[lang]
    tgt_id = np.asscalar(np.where(distance["langs"]==identifier)[0])
    dist = distance["data"][en_id, tgt_id]
    fout.write("{}/{}\t".format(lang, old_id))
    for num in dist:
        fout.write("{:.3f}\t".format(num))
    fout.write("\n")

fid.close()
fquery.close()
fout.close()