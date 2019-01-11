from __future__ import print_function

import os
import argparse
import time
import sys
import pickle

import torch
import numpy as np

import modules.dmv_flow_model as dmv
from modules import data_iter, \
                    read_conll, \
                    sents_to_vec, \
                    sents_to_tagid, \
                    to_input_tensor, \
                    generate_seed


def init_config():

    parser = argparse.ArgumentParser(description='dependency parsing')

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

    save_dir = "dump_models/dmv"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "{}_{}_{}_{}_{}".format(args.lang, args.mode, args.model, args.jobid, args.taskid)
    save_path = os.path.join(save_dir, id_ + '.pt')
    args.save_path = save_path

    print("model save path: ", save_path)

    # load config file into args
    config_file = "config.config_{}".format(args.lang)
    params = importlib.import_module(config_file).params_dmv
    args = argparse.Namespace(**vars(args), **params)

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

    train_text, train_tags, train_heads = read_conll(args.train_file)
    val_text, val_tags, val_heads = read_conll(args.val_file)
    test_text, test_tags, test_heads = read_conll(args.test_file)

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

    # train_tagid, tag2id = sents_to_tagid(train_sents)
    # print('%d types of tags' % len(tag2id))
    # id2tag = {v: k for k, v in tag2id.items()}

    # pad = np.zeros(num_dims)
    # device = torch.device("cuda" if args.cuda else "cpu")
    # args.device = device

    model = dmv.DMVFlow(args, tag_dict, num_dims).to(device)

    init_seed = to_input_tensor(generate_seed(train_vec, args.batch_size),
                                pad, device=device)

    with torch.no_grad():
        model.init_params(init_seed, train_tagid, train_vec)
    print('complete init')

    if args.train_from != '':
        model.load_state_dict(torch.load(args.train_from))
        with torch.no_grad():
            directed, undirected = model.test(test_deps, test_vec, verbose=False)
        print('acc on length <= 10: #trees %d, undir %2.1f, dir %2.1f' \
              % (len(test_gold), 100 * undirected, 100 * directed))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_niter = (len(train_vec)//args.batch_size)//5
    report_ll = report_num_words = report_num_sents = epoch = train_iter = 0
    stop_avg_ll = stop_num_words = 0
    stop_avg_ll_last = 1
    dir_last = 0
    begin_time = time.time()

    print('begin training')

    with torch.no_grad():
        directed, undirected = model.test(test_deps, test_vec)
    print('starting acc on length <= 10: #trees %d, undir %2.1f, dir %2.1f' \
          % (len(test_deps), 100 * undirected, 100 * directed))

    for epoch in range(args.epochs):
        report_ll = report_num_sents = report_num_words = 0
        for sents in data_iter(train_vec, batch_size=args.batch_size):
            batch_size = len(sents)
            num_words = sum(len(sent) for sent in sents)
            stop_num_words += num_words
            optimizer.zero_grad()

            sents_var, masks = to_input_tensor(sents, pad, device)
            sents_var, _ = model.transform(sents_var)
            sents_var = sents_var.transpose(0, 1)
            log_likelihood = model.p_inside(sents_var, masks)

            avg_ll_loss = -log_likelihood / batch_size

            avg_ll_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            report_ll += log_likelihood.item()
            report_num_words += num_words
            report_num_sents += batch_size

            stop_avg_ll += log_likelihood.item()

            if train_iter % log_niter == 0:
                print('epoch %d, iter %d, ll_per_sent %.4f, ll_per_word %.4f, ' \
                      'max_var %.4f, min_var %.4f time elapsed %.2f sec' % \
                      (epoch, train_iter, report_ll / report_num_sents, \
                      report_ll / report_num_words, model.var.data.max(), \
                      model.var.data.min(), time.time() - begin_time), file=sys.stderr)

            train_iter += 1
        if epoch % args.valid_nepoch == 0:
            with torch.no_grad():
                directed, undirected = model.test(test_deps, test_vec)
            print('\n\nacc on length <= 10: #trees %d, undir %2.1f, dir %2.1f, \n\n' \
                  % (len(test_deps), 100 * undirected, 100 * directed))

        stop_avg_ll = stop_avg_ll / stop_num_words
        rate = (stop_avg_ll - stop_avg_ll_last) / abs(stop_avg_ll_last)

        print('\n\nlikelihood: %.4f, likelihood last: %.4f, rate: %f\n' % \
                (stop_avg_ll, stop_avg_ll_last, rate))

        if rate < 0.001 and epoch >= 5:
            break

        stop_avg_ll_last = stop_avg_ll
        stop_avg_ll = stop_num_words = 0

    torch.save(model.state_dict(), args.save_path)

    # eval on all lengths
    if args.eval_all:
        test_sents, _ = read_conll(args.test_file)
        test_deps = [sent["head"] for sent in test_sents]
        test_vec = sents_to_vec(word_vec, test_sents)
        print("start evaluating on all lengths")
        with torch.no_grad():
            directed, undirected = model.test(test_deps, test_vec, eval_all=True)
        print('accuracy on all lengths: number of trees:%d, undir: %2.1f, dir: %2.1f' \
              % (len(test_gold), 100 * undirected, 100 * directed))


if __name__ == '__main__':
    parse_args = init_config()
    main(parse_args)
