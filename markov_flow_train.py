from __future__ import print_function

import pickle
import argparse
import sys
import time
import importlib
import os

import torch
import numpy as np

from modules import *
from multilingual_trans.fasttext import FastVector

lr_decay = 0.5

def init_config():

    parser = argparse.ArgumentParser(description='POS tagging')

    # train and test data
    parser.add_argument('--lang', type=str, help='language')

    # model config
    parser.add_argument('--model', choices=['gaussian', 'nice'], default='gaussian')
    parser.add_argument('--mode',
                         choices=['supervised', 'unsupervised', 'both', 'eval'],
                         default='supervised')

    # optimization params
    parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam')

    # pretrained model options
    parser.add_argument('--load_nice', default='', type=str,
        help='load pretrained projection model, ignored by default')
    parser.add_argument('--load_gaussian', default='', type=str,
        help='load pretrained Gaussian model, ignored by default')
    parser.add_argument('--seed', default=783435, type=int, help='random seed')
    parser.add_argument('--set_seed', action='store_true', default=False, help='if set seed')

    # these are for slurm purpose to save model
    # they can also be used to run multiple random restarts with various settings,
    # to save models that can be identified with ids
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "dump_models/markov"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "{}_{}_{}_{}_{}".format(args.lang, args.mode, args.model, args.jobid, args.taskid)
    save_path = os.path.join(save_dir, id_ + '.pt')
    args.save_path = save_path
    print("model save path: ", save_path)

    # load config file into args
    config_file = "config.config_{}".format(args.lang)
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)

    # if args.tag_from != '':
    #     if args.model == 'nice':
    #         args.load_nice = args.tag_from
    #     else:
    #         args.load_gaussian = args.tag_from
    #     args.tag_path = "pos_%s_%slayers_tagging%d_%d.txt" % \
    #     (args.model, args.couple_layers, args.jobid, args.taskid)

    if args.set_seed:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed * 13 / 7)

    print(args)

    return args

def main(args):

    word_vec_dict = FastVector(vector_file=args.vec_file)
    word_vec_dict.apply_transform(args.align_file)
    print('complete loading word vectors')

    train_text, train_tags = read_conll(args.train_file)
    val_text, val_tags = read_conll(args.val_file)
    test_text, test_tags = read_conll(args.test_file)

    train_vec = sents_to_vec(word_vec_dict, train_text)
    val_vec = sents_to_vec(word_vec_dict, val_text)
    test_vec = sents_to_vec(word_vec_dict, test_text)

    tag_dict = read_tag_map("tag_map.txt")

    train_tag_ids, _ = sents_to_tagid(train_tags, tag_dict)
    val_tag_ids, _ = sents_to_tagid(val_tags, tag_dict)
    test_tag_ids, _ = sents_to_tagid(test_tags, tag_dict)

    num_dims = len(train_vec[0][0])
    print('complete reading data')

    print("embedding dims {}".format(num_dims))
    print("#tags {}".format(len(tag_dict)))
    print("#train sentences: {}".format(len(train_vec)))
    print("#dev sentences: {}".format(len(val_vec)))
    print("#test sentences: {}".format(len(test_vec)))

    args.num_state = len(tag_dict)

    log_niter = (len(train_vec)//args.batch_size)//10

    pad = np.zeros(num_dims)
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    seed_vec, seed_tags = generate_seed(train_vec, train_tag_ids, args.batch_size)
    init_seed = to_input_tensor(seed_vec, seed_tags, pad, device=device)

    model = MarkovFlow(args, num_dims).to(device)

    model.init_params(init_seed)

    # if args.tag_from != '':
    #     model.eval()
    #     with torch.no_grad():
    #         accuracy, vm = model.test(test_data, test_tags, sentences=test_text,
    #             tagging=True, path=args.tag_path, null_index=null_index)
    #     print('\n***** M1 %f, VM %f, max_var %.4f, min_var %.4f*****\n'
    #           % (accuracy, vm, model.var.data.max(), model.var.data.min()), file=sys.stderr)
    #     return

    opt_dict = {"not_improved": 0, "lr": 0., "best_score": 0}

    if args.mode == "eval":
        model.eval()
        with torch.no_grad():
            acc = model.test_supervised(test_vec, test_tag_ids)
            m1, vm, oneone = model.test_unsupervised(test_vec, test_tag_ids)
        print("accuracy {}".format(acc))
        print("M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))
        return

    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        opt_dict["lr"] = 0.001
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        opt_dict["lr"] = 1.
    else:
        raise ValueError("{} is not supported".format(args.opt))

    begin_time = time.time()
    print('begin training')

    train_iter = report_obj = report_jc = report_ll = report_num_words = 0

    # print the accuracy under init params
    model.eval()
    with torch.no_grad():
        acc = model.test_supervised(test_vec, test_tag_ids)
        m1, vm, oneone = model.test_unsupervised(test_vec, test_tag_ids)
        print("\nTEST: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))
    print("\n*****starting acc {}, max_var {:.4f}, min_var {:.4f}*****\n".format(
          acc, model.var.max().item(), model.var.min().item()))
    print("\nstarting: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))

    model.train()
    for epoch in range(args.epochs):
        # model.print_params()
        report_obj = report_jc = report_ll = report_num_words = 0
        for sents, tags in data_iter(list(zip(train_vec, train_tag_ids)), batch_size=args.batch_size,
                               label=True, shuffle=True):
            train_iter += 1
            batch_size = len(sents)
            num_words = sum(len(sent) for sent in sents)
            sents_t, tags_t, masks = to_input_tensor(sents, tags, pad, device=args.device)
            optimizer.zero_grad()

            if args.mode == "unsupervised":
                nll, jacobian_loss = model.unsupervised_loss(sents_t, masks)
            elif args.mode == "supervised":
                nll, jacobian_loss = model.supervised_loss(sents_t, tags_t, masks)
            else:
                raise ValueError("{} mode is not supported".format(args.mode))

            avg_ll_loss = (nll + jacobian_loss)/batch_size

            avg_ll_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            log_likelihood_val = -nll.item()
            jacobian_val = -jacobian_loss.item()
            obj_val = log_likelihood_val + jacobian_val

            report_ll += log_likelihood_val
            report_jc += jacobian_val
            report_obj += obj_val
            report_num_words += num_words

            if train_iter % log_niter == 0:
                print('epoch %d, iter %d, log_likelihood %.2f, jacobian %.2f, obj %.2f, max_var %.4f ' \
                      'min_var %.4f time elapsed %.2f sec' % (epoch, train_iter, report_ll / report_num_words, \
                      report_jc / report_num_words, report_obj / report_num_words, model.var.max(), \
                      model.var.min(), time.time() - begin_time), file=sys.stderr)

        print('\nepoch %d, log_likelihood %.2f, jacobian %.2f, obj %.2f\n' % \
            (epoch, report_ll / report_num_words, report_jc / report_num_words,
             report_obj / report_num_words), file=sys.stderr)

        model.eval()
        if args.mode == "supervised":
            with torch.no_grad():
                acc = model.test_supervised(val_vec, val_tag_ids)
                print('\nDEV: *****epoch {}, iter {}, acc {}*****\n'.format(
                    epoch, train_iter, acc))

            if acc > opt_dict["best_score"]:
                opt_dict["best_score"] = acc
                opt_dict["not_improved"] = 0
                torch.save(model.state_dict(), args.save_path)
            else:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= 5:
                    opt_dict["best_score"] = acc
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    model.load_state_dict(torch.load(args.save_path))
                    print("new lr: {}".format(opt_dict["lr"]))
                    if args.opt == "adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=opt_dict["lr"])
                    elif args.opt == "sgd":
                        optimizer = torch.optim.SGD(model.parameters(), lr=opt_dict["lr"])
        else:
            torch.save(model.state_dict(), args.save_path)

        with torch.no_grad():
            acc = model.test_supervised(test_vec, test_tag_ids)
            m1, vm, oneone = model.test_unsupervised(test_vec, test_tag_ids)
        print('\nTEST: *****epoch {}, iter {}, acc {}*****\n'.format(
            epoch, train_iter, acc))
        print("\nTEST: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))

        model.train()

    model.eval()
    model.load_state_dict(torch.load(args.save_path))
    with torch.no_grad():
        acc = model.test_supervised(test_vec, test_tag_ids)
        m1, vm, oneone = model.test_unsupervised(test_vec, test_tag_ids)
        print('\nTEST: *****epoch {}, iter {}, acc {}*****\n'.format(
            epoch, train_iter, acc))
        print("\nTEST: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))

if __name__ == '__main__':
    parse_args = init_config()
    main(parse_args)
