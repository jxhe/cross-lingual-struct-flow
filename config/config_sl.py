params_markov={
    "couple_layers": 8,
    "cell_layers": 1,
    "lstm_layers": 2,
    "valid_nepoch": 1,
    "epochs": 10,
    "batch_size": 16,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_Slovenian-SSJSST/sl_ssjsst-ud-train.conllu",
    "val_file": "ud-treebanks-v2.2/UD_Slovenian-SSJSST/sl_ssjsst-ud-dev.conllu",
    "test_file": "ud-treebanks-v2.2/UD_Slovenian-SSJSST/sl_ssjsst-ud-test.conllu",
    "vec_file": "fastText_data/wiki.sl.ssjsst.vec.new",
    "align_file": "multilingual_trans/alignment_matrices/sl.txt"
}

params_dmv={
    "couple_layers": 8,
    "cell_layers": 1,
    "valid_nepoch": 1,
    "lstm_layers": 2,
    "epochs": 5,
    "batch_size": 16,
    "emb_dir": "fastText_data",
    "train_file": "ud-treebanks-v2.2/UD_Slovenian-SSJSST/sl_ssjsst-ud-train.conllu",
    "val_file": "ud-treebanks-v2.2/UD_Slovenian-SSJSST/sl_ssjsst-ud-dev.conllu",
    "test_file": "ud-treebanks-v2.2/UD_Slovenian-SSJSST/sl_ssjsst-ud-test.conllu",
    "vec_file": "fastText_data/wiki.sl.ssjsst.vec.new",
    "align_file": "multilingual_trans/alignment_matrices/sl.txt"
}