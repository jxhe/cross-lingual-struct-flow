
params_markov={
    "couple_layers": 8,
    "cell_layers": 1,
    "valid_nepoch": 1,
    "lstm_layers": 2,
    "epochs": 20,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "ptb_tag_train.conllu",
    "val_file": "ptb_tag_train.conllu",
    "test_file": "ptb_tag_train.conllu",
    "vec_file": "fastText_data/wiki.en.ewt.vec",
    "align_file": "multilingual_trans/alignment_matrices/en.txt"
}

params_dmv={
    "couple_layers": 8,
    "cell_layers": 1,
    "lstm_layers": 2,
    "valid_nepoch": 1,
    "epochs": 10,
    "batch_size": 32,
    "emb_dir": "fastText_data",
    "train_file": "ptb_tag_train.conllu",
    "val_file": "ptb_tag_train.conllu",
    "test_file": "ptb_tag_train.conllu",
    "vec_file": "fastText_data/wiki.en.ewt.vec",
    "align_file": "multilingual_trans/alignment_matrices/en.txt"
}
