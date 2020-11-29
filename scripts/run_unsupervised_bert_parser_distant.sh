#! /bin/bash
#
# run_parser.sh
# Copyright (C) 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.
#

CUDA_VISIBLE_DEVICES=$1 python -u dmv_flow_train.py \
        --lang $2 \
        --mode unsupervised \
        --set_seed \
        --model nice \
        --prob_const 1. \
        --max_len 150 \
        --train_max_len 40 \
        --pos_emb_dim 300 \
        --proj_lr 0.001 \
        --prior_lr 0.01 \
        --freeze_pos_emb \
        --bert_dir bert-base-multilingual-cased-emb \
        --load_nice outputs/parsing/en_supervised_wopos_nice_bert-base-multilingual-cased_bprior0.0_bproj0.0_bmean0.0_em/model.pt \
        --beta_prior 0.1 \
        --beta_proj 0.1 \
        --beta_mean 0.1 \

