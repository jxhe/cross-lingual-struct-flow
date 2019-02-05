
params_markov={
    "couple_layers": 8,
    "cell_layers": 1,
    "valid_nepoch": 1,
    "epochs": 50,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu",
    "val_file": "ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu",
    "test_file": "ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu",
    "vec_file": "fastText_data/wiki.hi.hdtb.vec",
    "align_file": "multilingual_trans/alignment_matrices/hi.txt"
}

params_dmv={
    "couple_layers": 4,
    "cell_layers": 1,
    "lstm_layers": 2,
    "valid_nepoch": 1,
    "epochs": 50,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu",
    "val_file": "ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu",
    "test_file": "ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-test.conllu",
    "vec_file": "fastText_data/wiki.hi.hdtb.vec",
    "align_file": "multilingual_trans/alignment_matrices/hi.txt"
}
