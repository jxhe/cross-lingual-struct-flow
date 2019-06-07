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
        --load_nice outputs/tagging/en_supervised_nice_couple8_cell1_bprior0.0_bproj0.0_bmean0.0/model.pt \
        --save_dir test_it \
        --beta_proj 80. \
        --beta_prior 0. \
        --beta_mean 500. \
        --prior_lr 0.001 \
        --proj_lr 0.0001 \

