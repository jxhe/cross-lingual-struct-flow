
params_markov={
    "couple_layers": 8,
    "cell_layers": 1,
    "valid_nepoch": 1,
    "epochs": 50,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu",
    "val_file":"ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu",
    "test_file":"ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu",
    "vec_file": "fastText_data/wiki.en.ewt.vec",
    "align_file": "multilingual_trans/alignment_matrices/en.txt"
}

params_dmv={
    "couple_layers": 8,
    "cell_layers": 1,
    "valid_nepoch": 1,
    "epochs": 50,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu",
    "val_file":"ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu",
    "test_file":"ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-test.conllu",
    "vec_file": "fastText_data/wiki.en.ewt.vec",
    "align_file": "multilingual_trans/alignment_matrices/en.txt"
}
