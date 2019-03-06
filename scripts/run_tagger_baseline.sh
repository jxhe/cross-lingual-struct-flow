#! /bin/bash
#
# run_tagger.sh
# Copyright (C) 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.
#


CUDA_VISIBLE_DEVICES=$1 python -u markov_flow_train.py \
        --lang $2 \
        --mode unsupervised \
        --set_seed \
        --load_nice dump_models/baseline/en_supervised_gaussian_0_0.pt \
        --beta 0.01 \
        --prior_lr 0.001 \
        --proj_lr 0.001 \
        --taskid $3 > $4
        # --beta_mean 20. \
