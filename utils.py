import os
import torch
import random
import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import math
from torch_scatter import scatter
from torch_geometric.utils import degree


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path) 

def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True)  # set min to zero [0, +infinity]
    scores = scores + 0.0001 # set min to 0.0001 [0.0001, +infinity]
    bid, eid = labels.nonzero()
    label_scores = np.where(filters[bid, :]==1, -100, scores[bid, :])  # set ground truth to be -100
    label_scores[np.arange(label_scores.shape[0]), eid] = scores[bid, eid]  # set object label entity scores to real scores
    # method: average. Entities get average rank if they share same score.
    label_ranks = rankdata(-label_scores, method='average', axis=-1)[np.arange(label_scores.shape[0]), eid]  # object ranks
    label_ranks = label_ranks.tolist()
    return label_ranks


def cal_performance(ranks, masks):
    mrr = (1. / ranks).sum() / len(ranks)
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_3 = sum(ranks<=3) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    h_10_50 = []
    for i, rank in enumerate(ranks):
        num_sample = 50
        threshold = 10
        score = 0
        fp_rate = (rank - 1) / masks[i]
        for i in range(threshold):
            num_comb = math.factorial(num_sample) / math.factorial(i) / math.factorial(num_sample - i)
            score += num_comb * (fp_rate**i) * ((1-fp_rate) **(num_sample-i))
        h_10_50.append(score)
    h_10_50 = np.mean(h_10_50)
            
    return mrr, m_r, h_1, h_3, h_10, h_10_50


def output_time(start, end, content, output=None):
    during = end-start
    hours = int(during // 3600)  # 1h=3600s
    during %= 3600
    minutes = int(during // 60)  # 
    during %= 60
    seconds = int(during)
    content = f'{content} [{hours:02d}:{minutes:02d}:{seconds:02d}]'
    if output is not None:
        output(content)
    return content


def agg_pna(input, index, dim_size, node_dim, eps):
    mean = scatter(input, index, dim=node_dim, dim_size=dim_size, reduce="mean")
    sq_mean = scatter(input ** 2, index, dim=node_dim, dim_size=dim_size, reduce="mean")
    max = scatter(input, index, dim=node_dim, dim_size=dim_size, reduce="max")
    min = scatter(input, index, dim=node_dim, dim_size=dim_size, reduce="min")
    std = (sq_mean - mean ** 2).clamp(min=eps).sqrt()
    features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
    features = features.flatten(-2)
    degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
    scale = degree_out.log()
    scale = scale / scale.mean()
    scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
    output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
    return output