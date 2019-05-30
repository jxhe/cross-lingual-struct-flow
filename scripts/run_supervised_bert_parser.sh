#! /bin/bash
#
# run_parser.sh
# Copyright (C) 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.
#

CUDA_VISIBLE_DEVICES=$1 python -u dmv_flow_train.py \
        --lang en \
        --mode supervised_wopos \
        --model nice \
        --prob_const 1. \
        --bert_dir bert-base-multilingual-cased \
        --max_len 20 \
        --pos_emb_dim 0 \
        # --em_train \
        # --set_seed \
        # --taskid $1 \
        # --pos_emb_dim 300 \

