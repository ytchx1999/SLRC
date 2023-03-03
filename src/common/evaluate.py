# -*- coding: UTF-8 -*-

import math
import numpy as np


def evaluate_metric(rank_lsts, purchase_lsts, topk):
    precisions, recalls, ndcgs, mrrs = [], [], [], []
    num_interaction = 0
    for rank_lst, purchase_lst in zip(rank_lsts, purchase_lsts):
        num_interaction += len(purchase_lst)
        precision, recall = get_precision_recall(rank_lst[:topk], purchase_lst)
        ndcg, mrr = get_ndcg(rank_lst[:topk], purchase_lst)
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
    avg_precision, avg_recall, avg_ndcg, avg_mrr = np.mean(precisions), np.sum(recalls) / num_interaction, np.mean(ndcgs), np.sum(mrrs) / num_interaction
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if avg_precision + avg_recall > 0 else 0.0
    result = {'precision': avg_precision,
              'recall': avg_recall,
              'f1': f1,
              'ndcg': avg_ndcg,
              'mrr': avg_mrr}
    return result


def get_precision_recall(rank_lst, purchase_lst):
    intersection = len(set(rank_lst).intersection(set(purchase_lst)))
    precision = (intersection / len(rank_lst)) if len(rank_lst) != 0 else 0
    recall = intersection # (intersection / len(purchase_lst))
    return precision, recall


def get_ndcg(rank_lst, purchase_lst):
    dcg = 0
    mrr = 0
    for i in range(len(rank_lst)):
        item = rank_lst[i]
        if item in purchase_lst:
            dcg += math.log(2) / math.log(i + 2)
            mrr += (float(1) / float(i + 1))
    idcg = 0
    for i in range(len(purchase_lst)):
        idcg += math.log(2) / math.log(i + 2)
    return float(dcg / idcg) / len(purchase_lst), (mrr)
