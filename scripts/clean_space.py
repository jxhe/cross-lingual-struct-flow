#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019-02-11 Junxian <He>
#
# Distributed under terms of the MIT license.

# This script cleans all the words that contain spaces in
# .vec file

import os
from io import open

for root, subdirs, files in os.walk("fastText_data"):
    for fname in files:
        if fname.endswith("vec") and len(fname.split(".")) == 4:
            fout_name = fname + ".new"
            fin = open(os.path.join(root, fname), "r", encoding="utf-8")
            fout = open(os.path.join(root, fout_name), "w", encoding="utf-8")
            num_words, ndim = fin.readline().split()
            ndim = int(ndim)
            num_words = int(num_words)

            data = []
            for line in fin:
                if len(line.split()) != (ndim + 1):
                    continue
                else:
                    data.append(line)

            fout.write("{} {}\n".format(len(data), ndim))
            for line in data:
                fout.write(line)

            fout.close()
