"""
This script creates a tag map to map tag to ids

"""

from modules import *

train_text, train_tags = read_conll("ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu")
train_tag_ids, tag_dict = sents_to_tagid(train_tags)

with open("tag_map.txt", "w") as fout:
	for tag in tag_dict:
		fout.write("{} {}\n".format(tag, tag_dict[tag]))

