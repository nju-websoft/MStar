import os
import copy
import logging
import argparse
import numpy as np
import scipy.sparse as ssp
import multiprocessing as mp
from tqdm import tqdm

def load_dict(path):
    dict = {}
    inverse_dict = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for lin in lines:
            v, id = lin.split('\t')
            dict[v] = int(id)
            inverse_dict[int(id)] = v
    return dict, inverse_dict

def load_dict_withoutid(path):
    dict = {}
    inverse_dict = {}
    id = 0
    with open(path, "r") as f:
        lines = f.readlines()
        for lin in lines:
            v = lin.strip("\n")
            dict[v] = id
            inverse_dict[id] = v
            id += 1
    return dict, inverse_dict
def load_links(path, e2id, r2id):
    if not os.path.exists(path):
        print(f'{path} does not exist.')
        return None
    links = []
    with open(path, "r") as f:
        lines = f.readlines()
        for lin in lines:
            h, r, t = lin.split('\n')[0].split('\t')
            h, t = e2id[h], e2id[t]
            if r not in r2id:
                assert False, 'r not in r2id!'
            r = r2id[r]
            links.append([h, r, t])
    return links

def load_dataset(base_path, r2id):
    e2id_path = os.path.join(base_path, "entities.txt")
    e2id, id2e = load_dict(e2id_path)
    train = load_links(os.path.join(base_path, "train.txt"), e2id, r2id)
    valid = load_links(os.path.join(base_path, "valid.txt"), e2id, r2id)
    test = load_links(os.path.join(base_path, "test.txt"), e2id, r2id)
    return e2id, id2e, train, valid, test



def output_distance_info(distance):
    distance[distance == np.inf] = -1
    max_distance = int(distance.max())
    all_distance = list(range(1, max_distance + 1)) + [-1]
    sum_rate = 0
    logging.info("Dist\tNum \tRate    \tSumRate")
    for d in all_distance:
        link_idx = (distance == d).nonzero()[0]
        curr_rate = 100.0 * len(link_idx) / len(distance)
        sum_rate += curr_rate
        # logging.info(f"Dist: {d}\tNum: {len(link_idx)}\tRate: {curr_rate:.4f}%\tSumRate: {sum_rate:.4f}%")
        logging.info(f"{d:<4d}\t{len(link_idx):<4d}\t{curr_rate:.4f}% \t{sum_rate:.4f}%")


def initialization(_support_link, _e2id):
    global support_link, e2id
    support_link = _support_link
    e2id = _e2id
    

def calcu_dist(support_link, links, e2id):
    distance = np.zeros(len(links))
    for idx, link in enumerate(tqdm(links)):
        h, r, t = link
        in_support_link = (link in support_link)
        if in_support_link:
            support_link.remove([h,r,t])

        # distance between h and t
        support = np.array(support_link)  # links: [[h,r,t], ...]
        e_num = len(e2id)
        g = ssp.csr_matrix((np.ones(support.shape[0]), (support[:, 0], support[:, 2])),
                           shape=(e_num, e_num))
        dist, pre = ssp.csgraph.dijkstra(csgraph=g, directed=False, indices=h, return_predecessors=True,
                                         unweighted=True)  # 13
        dist2t = dist[t]
        distance[idx] = dist2t
        if in_support_link:
            support_link.append([h,r,t])

    return distance

def inductive(args):
    dataset = args.dataset
    base_data_train_path = os.path.join('data', f'{dataset}')
    base_data_test_path = os.path.join('data', f'{dataset}_ind')
    r2id_path = os.path.join('data', dataset)
    output_test_path = os.path.join('data', f'{dataset}_ind', "test4.txt")

    r2id, id2r = load_dict(os.path.join(r2id_path, 'relations.txt'))
    r_num = len(r2id)

    tr_e2id, tr_id2e, tr_train, tr_valid, tr_test = load_dataset(base_data_train_path, r2id)
    te_e2id, te_id2e, te_train, te_valid, te_test = load_dataset(base_data_test_path, r2id)

    # train/train.txt
    distance_train = calcu_dist(tr_train, copy.deepcopy(tr_train), tr_e2id)
    output_distance_info(distance_train)

    # train/valid.txt
    # distance_train = calcu_dist(tr_train, tr_valid, tr_e2id)
    # output_distance_info(distance_train)

    # test/test.txt
    distance_test = calcu_dist(te_train, te_test, te_e2id)
    output_distance_info(distance_test)

    # output test/test4.txt
    with open(output_test_path, "w") as f:
        for link, d in zip(te_test, distance_test):
            h, r, t = link
            f.write(f"{te_id2e[h]}\t{id2r[r]}\t{te_id2e[t]}\t{int(d)}\n")  # h, r, t, d
    logging.info(f"output new test.txt to {output_test_path} over!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for Distance Processing")
    parser.add_argument('--dataset', '-D', type=str, default='fb237_v1')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename=os.path.join("analysis", "dist_logs", f"dist_{args.dataset}.log"),
                        filemode='w')
    inductive(args)