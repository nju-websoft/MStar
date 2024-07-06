import os
import time
import torch
import queue

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from torch import Tensor
from typing import Optional
from functools import reduce
from collections import defaultdict as ddict


import torch.nn.functional as F
from torch.autograd import Variable

SPLIT = '*' * 30


# copy from torch_geometric\utils\num_nodes
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

# copy from NBFNet
def degree(index, num_nodes: Optional[int] = None,
           dtype: Optional[int] = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)

# copy from NBFNet
def edge_match(edge_index, query_index):

    base = edge_index.max(dim=1)[0] + 1
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)  # h_num, h_num * r_num (r_num: considering reverse relations)
    scale = scale[-1] // scale  # r_num, 1
    edge_index_t = edge_index.t()
    query_index_t = query_index.t()
    # where(edge_index_t, query_index_t[0], True)
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)  # hash: h/r -> h*r_num+r
    edge_hash, order = edge_hash.sort()  # value, indices
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start

    offset = num_match.cumsum(0) - num_match  # prefix-sum (without self)
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)
    return order[range], num_match

# copy from NBFNet
def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])  # data_all - h,r  (2, edge_num)
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])  # batch - h,r (2, batch_size)
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index,  query_index) 
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]  # truth_t
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(
        num_t_truth) 
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool,
                        device=batch.device)  # (batch_size, num_nodes)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0  # true
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)
    # mask (batch_size, num_nodes)
    return t_mask, h_mask



def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask

