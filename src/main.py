# -*- coding: UTF-8 -*-

import os
import sys
import time
import argparse
import numpy as np
# from sklearn.externals import joblib
import joblib

from src.models.SLRC_BPR import SLRCBPR
from src.models.SLRC_Tensor import SLRCTensor
from src.models.SLRC_NCF import SLRCNCF
from src.Corpus import Corpus
from src.common.constants import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run SLRC.")
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--model', nargs='?', default='../model/SLRC/',
                        help='Model save path.')
    parser.add_argument('--dataset', nargs='?', default='order',
                        help='Choose a dataset.')
    parser.add_argument('--restore', type=bool, default=False,
                        help='Whether to restore exist model in model save path.')
    parser.add_argument('--gpu', type=str, default='',
                        help='Set CUDA_VISIBLE_DEVICES')

    parser.add_argument('--random_seed', type=int, default=2018,
                        help='Random seed of numpy and tensorflow.')
    parser.add_argument('--user_min', type=int, default=5,
                        help='Minimum transactions that users should have.')
    parser.add_argument('--item_min', type=int, default=5,
                        help='Minimum times that items should be bought.')
    parser.add_argument('--topk', nargs='?', default='[10,20]',
                        help='The number of items recommended to each user.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')

    parser.add_argument('--cf', nargs='?', default='BPR',
                        help='Choose a CF method to calculate base intensity: BPR, Tensor, NCF.')
    parser.add_argument('--epoch', type=int, default=5,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--l2', type=float, default=1e-4,
                        help='Weight of l2_regularize in loss.')
    # For BPR
    parser.add_argument('--K', type=int, default=100,
                        help='Number of embedding dimension.')
    # For Tensor
    parser.add_argument('--time_bin', type=int, default=100,
                        help='Number of bins time divided.')
    # For NCF
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability for each deep layer')
    parser.add_argument('--layers', nargs='?', default='[100,50]',
                        help="MLP layers. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[1e-4,1e-2]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    return parser.parse_args()


def get_model(args, corpus, random_seed):
    model_path = '{}{}-{}-{}.ckpt'.format(args.model, args.cf, args.dataset, random_seed)
    if not os.path.exists(model_path[:model_path.rfind('/')]):
        os.makedirs(model_path[:model_path.rfind('/')])

    model = None
    if args.cf.upper() == 'BPR':
        model = SLRCBPR(cf=args.cf, user_num=corpus.n_users, item_num=corpus.n_items, click_num=corpus.n_clicks,
                        time_span=corpus.max_time - corpus.min_time, avg_repeat_interval=corpus.total_avg_interval,
                        learning_rate=args.lr, batch_size=args.batch_size, epoch=args.epoch, l2_regularize=args.l2,
                        emb_dim=args.K, topk=eval(args.topk), random_seed=random_seed, model_path=model_path,
                        sample_candidate=NEG_SAMPLE[args.dataset])
    elif args.cf.upper() == 'TENSOR':
        model = SLRCTensor(cf=args.cf, user_num=corpus.n_users, item_num=corpus.n_items, click_num=corpus.n_clicks,
                           time_span=corpus.max_time - corpus.min_time, avg_repeat_interval=corpus.total_avg_interval,
                           learning_rate=args.lr, batch_size=args.batch_size, epoch=args.epoch, l2_regularize=args.l2,
                           emb_dim=args.K, topk=eval(args.topk), random_seed=random_seed, model_path=model_path,
                           sample_candidate=NEG_SAMPLE[args.dataset], min_time=corpus.min_time, time_bin=args.time_bin)
    elif args.cf.upper() == 'NCF':
        model = SLRCNCF(cf=args.cf, user_num=corpus.n_users, item_num=corpus.n_items, click_num=corpus.n_clicks,
                        time_span=corpus.max_time - corpus.min_time, avg_repeat_interval=corpus.total_avg_interval,
                        learning_rate=args.lr, batch_size=args.batch_size, epoch=args.epoch, l2_regularize=args.l2,
                        emb_dim=args.K, topk=eval(args.topk), random_seed=random_seed, model_path=model_path,
                        sample_candidate=NEG_SAMPLE[args.dataset], dropout=args.dropout,
                        layers=eval(args.layers), reg_layers=eval(args.reg_layers))
    else:
        print('Unrecongnized CF Method Error. (Available: BPR, Tensor, NCF)')
        exit(1)
    return model


def print_res(model, corpus, t1, restore=False):
    if not restore:
        # Find the best validation result across iterations
        best_valid_score = max(model.valid_loss)
        best_epoch = model.valid_loss.index(best_valid_score)
        print("\nBest Iter(validation)= %d\t valid = %.4f, test = %.4f"
              % (best_epoch + 1, model.valid_loss[best_epoch], model.test_loss[best_epoch]))
        best_test_score = max(model.test_loss)
        best_epoch = model.test_loss.index(best_test_score)
        print("Best Iter(test)= %d\t valid = %.4f, test = %.4f"
              % (best_epoch + 1, model.valid_loss[best_epoch], model.test_loss[best_epoch]))

    # Evaluate
    all_val_result = dict()
    all_test_result = dict()
    sample_users = 5000
    # if self.user_num > sample_users:
    valid_indice = np.random.choice(len(corpus.data['dev']), replace=False, size=sample_users)
    validation_data = np.array(corpus.data['dev'])[valid_indice]
    test_indice = np.random.choice(len(corpus.data['test']), replace=False, size=sample_users)
    test_data = np.array(corpus.data['test'])[test_indice]
    for topk in model.topk:
        # valid_result = model.evaluate(corpus.data['dev'], corpus.book, topk)
        # test_result = model.evaluate(corpus.data['test'], corpus.book, topk)
        valid_result = model.evaluate(validation_data, corpus.book, topk)
        test_result = model.evaluate(test_data, corpus.book, topk)
        print('\nTop@{}'.format(topk))
        print('Validation: recall = {:<.4f}, ndcg = {:<.4f}, mrr = {:<.4f}\n'
              'Test:       recall = {:<.4f}, ndcg = {:<.4f}, mrr = {:<.4f}\n[{:<.1f} s]'.format(
               valid_result['recall'], valid_result['ndcg'], valid_result['mrr'],
               test_result['recall'], test_result['ndcg'], test_result['mrr'],
               time.time() - t1))
        all_test_result[topk] = test_result
        all_val_result[topk] = valid_result
    return all_test_result, all_val_result


def main(args):
    # Data loading
    if not LOAD_CORPUS:
        corpus = Corpus(args.path, args.dataset)
        corpus.load_data()
        # joblib.dump(corpus, '{}corpus-{}.npz'.format(args.path, args.dataset))
    else:
        print('Load corpus from {}corpus-{}.npz'.format(args.path, args.dataset))
        corpus = joblib.load('{}corpus-{}.npz'.format(args.path, args.dataset))

    print("\narguments:")
    print(args)

    fold = 3
    topk = eval(args.topk)
    total_val_result = dict()
    total_result = dict()
    for k in topk:
        total_val_result[k] = {
            'precision': [], 'recall': [], 'f1': [], 'ndcg': [], 'mrr': [],
        }
        total_result[k] = {
            'precision': [], 'recall': [], 'f1': [], 'ndcg': [], 'mrr': [],
        }
    for f in range(fold):
        print('\nFold {}'.format(f))
        # Training
        print('Training start...')
        sys.stdout.flush()
        t1 = time.time()
        random_seed = args.random_seed - f
        model = get_model(args, corpus, random_seed)
        if not args.restore:
            model.train(corpus)
        else:
            model.saver.restore(model.sess, model.model_path)
        test_result, val_result = print_res(model, corpus, t1, restore=args.restore)
        for k in topk:
            total_result[k]['precision'].append(test_result[k]['precision'])
            total_result[k]['recall'].append(test_result[k]['recall'])
            total_result[k]['f1'].append(test_result[k]['f1'])
            total_result[k]['ndcg'].append(test_result[k]['ndcg'])
            total_result[k]['mrr'].append(test_result[k]['mrr'])

            total_val_result[k]['precision'].append(val_result[k]['precision'])
            total_val_result[k]['recall'].append(val_result[k]['recall'])
            total_val_result[k]['f1'].append(val_result[k]['f1'])
            total_val_result[k]['ndcg'].append(val_result[k]['ndcg'])
            total_val_result[k]['mrr'].append(val_result[k]['mrr'])

    print('\nAverage in valid dataset:')
    for k in topk:
        print('Top@{:<2}:'.format(k), end=' ')
        print('recall = {:<.4f}±{:<.4f}, ndcg = {:<.4f}±{:<.4f}, mrr = {:<.4f}±{:<.4f}'.format(
            # np.mean(total_result[k]['precision']), np.std(total_result[k]['precision']),
            np.mean(total_val_result[k]['recall']), np.std(total_val_result[k]['recall']),
            # np.mean(total_result[k]['f1']), np.std(total_result[k]['f1']),
            np.mean(total_val_result[k]['ndcg']), np.std(total_val_result[k]['ndcg']),
            np.mean(total_val_result[k]['mrr']), np.std(total_val_result[k]['mrr']),
        ))

    print('\nAverage in test dataset:')
    for k in topk:
        print('Top@{:<2}:'.format(k), end=' ')
        print('recall = {:<.4f}±{:<.4f}, ndcg = {:<.4f}±{:<.4f}, mrr = {:<.4f}±{:<.4f}'.format(
            # np.mean(total_result[k]['precision']), np.std(total_result[k]['precision']),
            np.mean(total_result[k]['recall']), np.std(total_result[k]['recall']),
            # np.mean(total_result[k]['f1']), np.std(total_result[k]['f1']),
            np.mean(total_result[k]['ndcg']), np.std(total_result[k]['ndcg']),
            np.mean(total_result[k]['mrr']), np.std(total_result[k]['mrr']),
        ))


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Avoid showing tensorflow warning for instructions supporting
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main(args)
