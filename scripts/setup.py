# This script is used to set up multiple experiments

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='setup')
parser.add_argument('--lang', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--task', choices=['tag', 'parse'])

args = parser.parse_args()

if args.task == 'tag':
    command = "./run_tagger.sh"
    out_dir = "exp_out/tagging"
else:
    command = "./run_parser.sh"
    out_dir = "exp_out/parsing"

if not os.exists(out_dir):
    os.makedirs(out_dir)


max_ = -1
for root, dirs, filenames in os.walk(out_dir):
    for fname in filenames:
        try:
            id_ = int(filenames)
            max_ = id_ if id_ > max_ else max_
    break

id_ = max_ + 1
out_dir = os.path.join(out_dir, str(id_))

for lang in args.lang.split(","):
    log_path = os.path.join(out_dir, "{}.log".format(lang))
    subprocess.run("CUDA_VISIBLE_DEVICES={} {} {} {}".format(args.gpu, command, lang, id_))
