#!/usr/bin/env python
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
    parser.add_argument('--model', choices=['gaussian', 'nice', 'lstmnice'], default='gaussian')
    parser.add_argument('--mode',
                         choices=['supervised', 'unsupervised', 'both', 'eval'],
                         default='supervised')

    # optimization params
    parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--prior_lr', type=float, default=0.001)
    parser.add_argument('--proj_lr', type=float, default=0.001)
    parser.add_argument('--freeze_proj', action='store_true', default=False)
    parser.add_argument('--freeze_prior', action='store_true', default=False)
    parser.add_argument('--freeze_mean', action='store_true', default=False)
    parser.add_argument('--train_var', action='store_true', default=False,
            help="if make variance variable trainable")
    parser.add_argument('--init_var_one', action='store_true', default=False)
    parser.add_argument('--aggressive', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=0., help="regularize params")

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
    params = importlib.import_module(config_file).params_markov
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
        np.random.seed(args.seed)

    print(args)

    return args

def main(args):

    word_vec_dict = FastVector(vector_file=args.vec_file)
    word_vec_dict.apply_transform(args.align_file)
    print('complete loading word vectors')

    pos_to_id = read_tag_map("tag_map.txt")
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    train_data = ConlluData(args.train_file, word_vec_dict,
        device=device, pos_to_id_dict=pos_to_id)
    val_data = ConlluData(args.val_file, word_vec_dict,
        device=device, pos_to_id_dict=pos_to_id)
    test_data = ConlluData(args.test_file, word_vec_dict,
        device=device, pos_to_id_dict=pos_to_id)


    num_dims = len(train_data.embed[0][0])
    print('complete reading data')

    print("embedding dims {}".format(num_dims))
    print("#tags {}".format(len(pos_to_id)))
    print("#train sentences: {}".format(train_data.length))
    print("#dev sentences: {}".format(val_data.length))
    print("#test sentences: {}".format(test_data.length))

    args.num_state = len(pos_to_id)

    log_niter = (train_data.length//args.batch_size)//10

    model = MarkovFlow(args, num_dims).to(device)

    with torch.no_grad():
        model.init_params(train_data)
    print("complete init")

    # if args.tag_from != '':
    #     model.eval()
    #     with torch.no_grad():
    #         accuracy, vm = model.test(test_data, test_tags, sentences=test_text,
    #             tagging=True, path=args.tag_path, null_index=null_index)
    #     print('\n***** M1 %f, VM %f, max_var %.4f, min_var %.4f*****\n'
    #           % (accuracy, vm, model.var.data.max(), model.var.data.min()), file=sys.stderr)
    #     return

    opt_dict = {"not_improved": 0, "prior_lr": args.prior_lr,
                "proj_lr": args.proj_lr, "best_score": 0}

    if args.mode == "eval":
        model.eval()
        with torch.no_grad():
            acc = model.test_supervised(test_data)
            m1, vm, oneone = model.test_unsupervised(test_data)
        print("accuracy {}".format(acc))
        print("M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))
        return

    if args.opt == "adam":
        prior_optimizer = torch.optim.Adam(model.prior_group, lr=args.prior_lr)
        proj_optimizer = torch.optim.Adam(model.proj_group, lr=args.proj_lr)
    elif args.opt == "sgd":
        prior_optimizer = torch.optim.SGD(model.prior_group, lr=args.prior_lr)
        proj_optimizer = torch.optim.SGD(model.proj_group, lr=args.proj_lr)
    else:
        raise ValueError("{} is not supported".format(args.opt))

    begin_time = time.time()
    print('begin training')

    train_iter = report_obj = report_jc = report_ll = report_num_words = 0

    # print the accuracy under init params
    model.eval()
    with torch.no_grad():
        acc = model.test_supervised(test_data)
        m1, vm, oneone = model.test_unsupervised(test_data)
        print("\nTEST: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))
    print("\n*****starting acc {}, max_var {:.4f}, min_var {:.4f}*****\n".format(
          acc, model.var.max().item(), model.var.min().item()))
    print("\nstarting: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))

    model.train()
    for epoch in range(args.epochs):
        # model.print_params()
        report_obj = report_jc = report_ll = report_num_words = 0
        for iter_obj in train_data.data_iter(batch_size=args.batch_size,
                                                shuffle=True):

            if args.aggressive:
                inner_iter = 0
                for iter_obj_tmp in train_data.data_iter(batch_size=args.batch_size,
                                                                shuffle=True):
                    proj_optimizer.zero_grad()
                    prior_optimizer.zero_grad()
                    batch_size = iter_obj_tmp.pos.size(1)
                    sents_t, tags_t, masks = iter_obj_tmp.embed, iter_obj_tmp.pos, iter_obj_tmp.mask
                    nll, jacobian_loss = model.unsupervised_loss(sents_t, masks)
                    avg_ll_loss = (nll + jacobian_loss)/batch_size

                    avg_ll_loss = avg_ll_loss + model.MLE_loss()
                    avg_ll_loss.backward()

                    prior_optimizer.step()
                    inner_iter += 1
                    if inner_iter > 10:
                        break

            # if args.mode == "supervised":
            #     optimizer.zero_grad()
            # elif args.mode == "unsupervised":
            #     prior_optimizer.zero_grad()
            #     proj_optimizer.zero_grad()
            # else:
            #     raise ValueError

            train_iter += 1
            batch_size = iter_obj.pos.size(1)
            num_words = iter_obj.mask.sum().item()
            sents_t, tags_t, masks = iter_obj.embed, iter_obj.pos, iter_obj.mask
            # optimizer.zero_grad()
            prior_optimizer.zero_grad()
            proj_optimizer.zero_grad()

            if args.mode == "unsupervised":
                nll, jacobian_loss = model.unsupervised_loss(sents_t, masks)
            elif args.mode == "supervised":
                nll, jacobian_loss = model.supervised_loss(sents_t, tags_t, masks)
            else:
                raise ValueError("{} mode is not supported".format(args.mode))

            avg_ll_loss = (nll + jacobian_loss)/batch_size

            if args.beta > 0:
                avg_ll_loss = avg_ll_loss + model.MLE_loss()
                # avg_ll_loss = model.MLE_loss()

            avg_ll_loss.backward()

            if args.mode != "unsupervised":
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(model.proj_layer.parameters(), 5.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.proj_group, 5.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # optimizer.step()
            prior_optimizer.step()
            proj_optimizer.step()

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
                      model.var.min(), time.time() - begin_time))

                # if args.mode == "unsupervised":
                #     with torch.no_grad():
                #         acc = model.test_supervised(test_vec, test_tag_ids)
                #         m1, vm, oneone = model.test_unsupervised(test_vec, test_tag_ids)
                #     print('\nTEST: *****epoch {}, iter {}, acc {}*****\n'.format(
                #         epoch, train_iter, acc))
                #     print("\nTEST: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))


        print('\nepoch %d, log_likelihood %.2f, jacobian %.2f, obj %.2f\n' % \
            (epoch, report_ll / report_num_words, report_jc / report_num_words,
             report_obj / report_num_words))

        model.eval()
        if args.mode == "supervised":
            with torch.no_grad():
                acc = model.test_supervised(val_data)
                print('\nDEV: *****epoch {}, iter {}, acc {}*****\n'.format(
                    epoch, train_iter, acc))

            if acc > opt_dict["best_score"]:
                opt_dict["best_score"] = acc
                opt_dict["not_improved"] = 0
                torch.save(model.state_dict(), args.save_path)
            else:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= 2:
                    opt_dict["not_improved"] = 0
                    opt_dict["prior_lr"] = opt_dict["prior_lr"] * lr_decay
                    opt_dict["proj_lr"] = opt_dict["proj_lr"] * lr_decay
                    model.load_state_dict(torch.load(args.save_path))
                    print("new prior lr: {}".format(opt_dict["prior_lr"]))
                    print("new proj lr: {}".format(opt_dict["proj_lr"]))
                    if args.opt == "adam":
                        prior_optimizer = torch.optim.Adam(model.prior_group, lr=opt_dict["prior_lr"])
                        proj_optimizer = torch.optim.Adam(model.proj_group, lr=opt_dict["proj_lr"])
                    elif args.opt == "sgd":
                        prior_optimizer = torch.optim.SGD(model.prior_group, lr=opt_dict["prior_lr"])
                        proj_optimizer = torch.optim.SGD(model.proj_group, lr=opt_dict["proj_lr"])
                    else:
                        raise ValueError("{} is not supported".format(args.opt))
        else:
            torch.save(model.state_dict(), args.save_path)

        with torch.no_grad():
            acc = model.test_supervised(test_data)
            m1, vm, oneone = model.test_unsupervised(test_data)
        print('\nTEST: *****epoch {}, iter {}, acc {}*****\n'.format(
            epoch, train_iter, acc))
        print("\nTEST: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))

        model.train()

    model.eval()
    model.load_state_dict(torch.load(args.save_path))
    with torch.no_grad():
        acc = model.test_supervised(test_data)
        m1, vm, oneone = model.test_unsupervised(test_data)
        print('\nTEST: *****epoch {}, iter {}, acc {}*****\n'.format(
            epoch, train_iter, acc))
        print("\nTEST: M1 {}, VM {}, one-to-one {}".format(m1, vm, oneone))

if __name__ == '__main__':
    parse_args = init_config()
    main(parse_args)
