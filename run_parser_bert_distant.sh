#! /bin/bash
#
# run_parser_similar_1.sh
# Copyright (C) 2019-03-02 Junxian <He>
#
# Distributed under terms of the MIT license.
#


python scripts/setup.py --gpu $1 --lang $2 --command ./scripts/run_parser_bert_distant.sh --task parse
