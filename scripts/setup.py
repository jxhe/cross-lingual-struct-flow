# This script is used to set up multiple experiments

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='setup')
parser.add_argument('--lang', type=str)
parser.add_argument('--gpu', type=str)
parser.add_argument('--task', choices=['tag', 'parse'])

args = parser.parse_args()

if args.task == 'tag':
    command = "./scripts/run_tagger.sh"
    out_dir = "exp_out/tagging"
else:
    command = "./scripts/run_parser.sh"
    out_dir = "exp_out/parsing"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


max_ = -1
for root, dirs, filenames in os.walk(out_dir):
    for dir_ in dirs:
        try:
            id_ = int(dir_)
            max_ = id_ if id_ > max_ else max_
        except:
            pass
    break

id_ = max_ + 1
out_dir = os.path.join(out_dir, str(id_))

os.makedirs(out_dir)

for lang in args.lang.split(","):
    log_path = os.path.join(out_dir, "{}.log".format(lang))
    subprocess.run([command, args.gpu, lang, str(id_), log_path])

