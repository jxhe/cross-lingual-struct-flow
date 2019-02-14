#! /bin/bash
#
# run_tagger.sh
# Copyright (C) 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.
#


python -u markov_flow_train.py \
        --lang $1 \
        --model nice \
        --mode unsupervised \
        --set_seed \
        --load_nice dump_models/markov/en_supervised_nice_0_102.pt \
        --freeze_mean \
        --beta_proj 100. \
        --beta_prior 0.1 \
        --prior_lr 0.01 \
        --proj_lr 0.001 \
        --taskid $2 \