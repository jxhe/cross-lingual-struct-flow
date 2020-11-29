#! /bin/bash
#
# run_tagger.sh
# Copyright (C) 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.
#


CUDA_VISIBLE_DEVICES=$1 python -u markov_flow_train.py \
        --lang en \
        --model nice \
        --mode supervised \
        --bert_dir bert-base-multilingual-cased-emb \
        --set_seed \

