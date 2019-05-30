#! /bin/bash
#
# run_parser.sh
# Copyright (C) 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.
#

python -u dmv_flow_train.py \
        --lang en \
        --mode supervised_wopos \
        --model nice \
        --prob_const 1. \
        --max_len 200 \
        --pos_emb_dim 300 \
        --em_train \
        --taskid $1 \
        # --set_seed \

