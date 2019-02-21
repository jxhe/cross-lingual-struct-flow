# This script collect results in args.input directory,
# and write them into a csv file

import csv
import os
import argparse

parser = argparse.ArgumentParser(description='write results to csv')
parser.add_argument('--input', type=str, help="this is a directory")

args = parser.parse_args()

hash_map = {}
for root, subdirs, files in os.walk(args.input):
    for fname in files:
        if fname.endswith(".log"):
            lang = fname.split(".")[0]
            with open(os.path.join(root, fname), "r") as fin:
                for line in fin:
                    if line.startswith("TEST: *****"):
                        acc = float(line.split()[-1].split("*")[0]) * 100
                        acc = "{:.2f}".format(acc)
                        acc = float(acc)
                        hash_map[lang] = acc

out_csv = open("result_{}_{}.csv".format(args.input.split("/")[-2], args.input.split("/")[-1]), "w", newline="")
csv_writer = csv.writer(out_csv, delimiter=",")
with open("statistics/lang_list.txt", "r") as fin:
    for line in fin:
        lang = line.split()[1]
        if lang == "en":
            continue
        row = [lang]
        if lang in hash_map:
            row += [hash_map[lang]]
        else:
            row += ["-"]
        csv_writer.writerow(row)

out_csv.close()
