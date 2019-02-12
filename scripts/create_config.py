# This script reads "lang_list.txt" and create all
# config files in config folder

import os
import json

fin = open("lang_list.txt", "r")
params_markov = {
    "couple_layers": 8,
    "cell_layers": 1,
    "lstm_layers": 2,
    "valid_nepoch": 1,
    "lstm_layers": 2,
    "epochs": 5,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "",
    "val_file":"",
    "test_file":"",
    "vec_file": "",
    "align_file": ""
}

params_dmv={
    "couple_layers": 8,
    "cell_layers": 1,
    "valid_nepoch": 1,
    "lstm_layers": 2,
    "epochs": 5,
    "batch_size": 16,
    "emb_dir": "fastText_data",
    "train_file": "",
    "val_file":"",
    "test_file":"",
    "vec_file": "",
    "align_file": ""
}

for line in fin:
    lang_name, abbr, treebank = line.split()
    if abbr == "en":
        continue
    dir_ = "ud-treebanks-v2.2/UD_{}-{}".format(lang_name, treebank)
    treebank_s = treebank.lower()
    train_file = "{}_{}-ud-train.conllu".format(abbr, treebank_s)
    dev_file = "{}_{}-ud-dev.conllu".format(abbr, treebank_s)
    test_file = "{}_{}-ud-test.conllu".format(abbr, treebank_s)

    train_file = os.path.join(dir_, train_file)
    dev_file = os.path.join(dir_, dev_file)
    test_file = os.path.join(dir_, test_file)

    if not os.path.exists(dev_file):
        dev_file = test_file

    vec_file = "fastText_data/wiki.{}.{}.vec.new".format(abbr, treebank_s)
    align_file = "multilingual_trans/alignment_matrices/{}.txt".format(abbr)
    params_markov["train_file"] = params_dmv["train_file"] = train_file
    params_markov["val_file"] = params_dmv["val_file"] = dev_file
    params_markov["test_file"] = params_dmv["test_file"] = test_file
    params_markov["vec_file"] = params_dmv["vec_file"] = vec_file
    params_markov["align_file"] = params_dmv["align_file"] = align_file

    with open("config/config_{}.py".format(abbr), "w") as fout:
        fout.write("params_markov="+json.dumps(params_markov, indent=4))
        fout.write("\n\n")
        fout.write("params_dmv="+json.dumps(params_dmv, indent=4))

