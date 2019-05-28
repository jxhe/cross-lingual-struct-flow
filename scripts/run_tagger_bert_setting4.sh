#! /bin/bash
#
# run_tagger.sh
# Copyright (C) 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.
#


CUDA_VISIBLE_DEVICES=$1 python -u markov_flow_train.py \
        --lang $2 \
        --model nice \
        --mode unsupervised \
        --set_seed \
        --load_nice dump_models/markov/en_supervised_nice_bert-base-multilingual-cased_8_1_0_0.pt \
        --bert_dir bert-base-multilingual-cased \
        --beta_proj 80. \
        --beta_prior 0. \
        --prior_lr 0.001 \
        --proj_lr 0.0001 \
        --beta_mean 500. \
        --taskid $3 > $4
        # --freeze_mean \
