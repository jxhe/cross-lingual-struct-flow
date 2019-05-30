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
        --load_nice dump_models/markov/en_supervised_nice_0_102.pt \
        --beta_proj 100. \
        --beta_prior 0. \
        --prior_lr 0.001 \
        --proj_lr 0.0001 \
        --beta_mean 100. \
        --freeze_mean \
        --taskid $3 > $4
