params_markov={
    "couple_layers": 8,
    "cell_layers": 1,
    "lstm_layers": 2,
    "valid_nepoch": 1,
    "epochs": 5,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-train.conllu",
    "val_file": "ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-dev.conllu",
    "test_file": "ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-test.conllu",
    "vec_file": "fastText_data/wiki.he.htb.vec.new",
    "align_file": "multilingual_trans/alignment_matrices/he.txt"
}

params_dmv={
    "couple_layers": 8,
    "cell_layers": 1,
    "valid_nepoch": 1,
    "lstm_layers": 2,
    "epochs": 5,
    "batch_size": 16,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-train.conllu",
    "val_file": "ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-dev.conllu",
    "test_file": "ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-test.conllu",
    "vec_file": "fastText_data/wiki.he.htb.vec.new",
    "align_file": "multilingual_trans/alignment_matrices/he.txt"
}