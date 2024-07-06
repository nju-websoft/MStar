import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import helper

def load_dict(path):
    dict2id = {}
    id2dict = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for lin in lines:
            v, id = lin.split('\t')
            dict2id[v] = int(id)
            id2dict[int(id)] = v
    return dict2id, id2dict

def double_links(links, n_rel):
    triples = []
    for h, r, t in links:
        triples.append([h, r, t])
        triples.append([t, r+n_rel, h])
    return triples

def load_links(path, bef_e2id, bef_r2id):
    e2id = {} if bef_e2id is None else bef_e2id
    r2id = {} if bef_r2id is None else bef_r2id

    triples = []

    def add2dict(v, v2id):
        if v not in v2id:
            v2id[v] = len(v2id)
        return v2id[v]

    links = []
    with open(path, "r") as f:
        for line in f:
            h, r, t = line.strip().split()
            if bef_e2id is None:
                h = add2dict(h, e2id)
                t = add2dict(t, e2id)
            else:
                if h not in e2id or t not in e2id:
                    continue
                h = e2id[h]
                t = e2id[t]
            if bef_r2id is None:
                r = add2dict(r, r2id)
            else:
                if r not in r2id:
                    assert False
                r = r2id[r]
            links.append([h, r, t])
    n_rel = len(r2id)
    triples = double_links(links, len(r2id))
    id2e = {v:k for k, v in e2id.items()}
    id2r = {v:k for k, v in r2id.items()}
    return triples, e2id, id2e, r2id, id2r

def load_grail_links(path, e2id, r2id, n_rel):
    links = []
    with open(path, "r") as f:
        for line in f:
            h, r, t = line.strip().split()
            h = e2id[h]
            r = r2id[r]
            t = e2id[t]
            links.append([h, r, t])
            links.append([t, r+n_rel, h])
    return links

def output_dict(path, v2id):
    with open(path, "w") as f:
        for v, id in v2id.items():
            f.write(f"{v}\t{id}\n")

class DataLoader:
    def __init__(self, args, dist=None, index=None, n_batch=32):
        self.device = f"cuda:{args.gpu}"
        self.n_batch = n_batch

        dataset = args.dataset
        self.load_grail(dataset, dist, index)

    def load_grail(self, dataset, dist, index):
        # load train_entity, test_entity, all_relation
        train_path = os.path.join("data", dataset)
        test_path = os.path.join("data", f"{dataset}_ind")
        self.entity2id, self.id2entity = load_dict(os.path.join(train_path, 'entities.txt'))
        self.relation2id, id2relation = load_dict(os.path.join(train_path, 'relations.txt'))
        self.entity2id_ind, self.id2entity_ind = load_dict(os.path.join(test_path, 'entities.txt'))
        for i in range(len(self.relation2id)):
            id2relation[i+len(self.relation2id)] = id2relation[i] + '_inv'
        id2relation[len(self.relation2id) * 2] = 'idd'
        self.id2relation = id2relation
        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.n_ent_ind = len(self.entity2id_ind)

        # load links
        self.tra_train = load_grail_links(os.path.join(train_path, "train.txt"), self.entity2id, self.relation2id, self.n_rel)
        self.tra_valid = load_grail_links(os.path.join(train_path, "valid.txt"), self.entity2id, self.relation2id, self.n_rel)
        self.tra_test = load_grail_links(os.path.join(train_path, "test.txt"), self.entity2id, self.relation2id, self.n_rel)
        self.ind_train = load_grail_links(os.path.join(test_path, "train.txt"), self.entity2id_ind, self.relation2id, self.n_rel)
        self.ind_valid = load_grail_links(os.path.join(test_path, "valid.txt"), self.entity2id_ind, self.relation2id, self.n_rel)
        self.ind_test = load_grail_links(os.path.join(test_path, "test.txt"), self.entity2id_ind, self.relation2id, self.n_rel)
        self.ind_test_no_sort = load_grail_links(os.path.join(test_path, "test.txt"), self.entity2id_ind, self.relation2id, self.n_rel)
        if dist is not None:
            self.ind_test, self.ind_test_d = self.read_triples_distance(test_path, 'test4.txt', dist=dist, mode='inductive')
            self.ind_test_no_sort, _ = self.read_triples_distance(test_path, 'test4.txt', dist=dist, mode='inductive')
        if index is not None:
            self.ind_test = [self.ind_test[index]]
            self.ind_test_no_sort = [self.ind_test_no_sort[index]]

        self.tra_train_links = self.tra_train
        self.ind_train_links = self.ind_train
        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')

        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])

        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)
        self.ind_KG, self.ind_sub = self.load_graph(self.ind_train, 'inductive')

        self.tra_train = np.array(self.tra_train)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)
        self.ind_val_qry, self.ind_val_ans = self.load_query(self.ind_valid)
        self.ind_tst_qry, self.ind_tst_ans = self.load_query(self.ind_test)
        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.test_q,  self.test_a  = self.ind_tst_qry, self.ind_tst_ans

        self.n_train = len(self.tra_train)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        if dist is None and index is None:
            print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, directory, filename, mode='transductive'):
        triples = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                triples.append([h,r,t])
                triples.append([t, r+self.n_rel, h])
        return triples

    def read_triples_distance(self, directory, filename, dist, mode='transductive'):
        triples = []
        all_dist = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t, d, = line.strip().split()
                all_dist.append(int(d))
                if dist is not None and int(d) not in dist:
                    continue
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                triples.append([h,r,t])
                triples.append([t, r+self.n_rel, h])
        return triples, all_dist

    def load_graph(self, triples, mode='transductive'):
        n_ent = self.n_ent if mode=='transductive' else self.n_ent_ind
        
        KG = np.array(triples)
        idd = np.concatenate([np.expand_dims(np.arange(n_ent),1), 2*self.n_rel*np.ones((n_ent, 1)), np.expand_dims(np.arange(n_ent),1)], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]

        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:,0])), shape=(n_fact, n_ent))
        return KG, M_sub
    
    def my_load_graph(self, triples, mode='transductive'):
        n_ent = self.n_ent if mode == 'transductive' else self.n_ent_ind
        KG = np.array(triples)  # 10820

        # idd already in KG
        # idd = np.concatenate([np.expand_dims(np.arange(n_ent),1), 2*self.n_rel*np.ones((n_ent, 1)), np.expand_dims(np.arange(n_ent),1)], 1)
        # KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]  # h
        # M_sub[i,j]: i->triplet_id, j->head_id
        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])), shape=(n_fact, n_ent))
        return KG, M_sub
    
    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return self.tra_train[batch_idx]
        if data=='valid':
            query = np.array(self.valid_q)
            answer = np.array(self.valid_a)
            n_ent = self.n_ent
        if data=='test':
            query = np.array(self.test_q)
            answer = np.array(self.test_a)
            n_ent = self.n_ent_ind

        subs = []
        rels = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def get_filter(self, data='valid'):
        filters = defaultdict(lambda: set())
        if data == 'valid':
            links = [self.tra_train, self.tra_valid, self.tra_test]
        else:
            links = [self.ind_train, self.ind_valid, self.ind_test]
        for triplets in links:
            if len(triplets) == 0:
                continue
            for triple in triplets:
                h, r, t = triple
                filters[(h,r)].add(t)
        return filters


    def get_edges(self, q_sub, q_rel, q_obj, mode='transductive'):
        device = q_sub.device
        if mode == 'transductive':  # train/valid
            edges = torch.tensor(self.tra_train, dtype=torch.long, device=device)
            ent_num = self.n_ent
            if q_obj is not None:  # train: remove self
                edges = edges.t()  # (3,edge_num)
                remove_edges = torch.cat([q_sub.unsqueeze(-1), q_rel.unsqueeze(-1), q_obj.unsqueeze(-1)], dim=-1).t()  # (3,query_num)
                index = helper.edge_match(edges, remove_edges)[0]
                mask = ~helper.index_to_mask(index, edges.size(-1))
                edges = edges[:, mask].t()
        else:  # test
            edges = torch.tensor(self.ind_train_links, dtype=torch.long, device=device)
            ent_num = self.n_ent_ind

        # self_loop
        idd_ht = torch.arange(ent_num, dtype=torch.long, device=device).unsqueeze(-1)
        idd_r = torch.ones((ent_num,1), dtype=torch.long, device=device) * (2 * self.n_rel)
        idd = torch.cat([idd_ht, idd_r, idd_ht], dim=-1)
        edges = torch.cat([edges, idd], dim=0)

        # matrix
        # 2.1
        # M_sub = torch.sparse_csr_tensor(torch.arange(len(edges)), edges[:, 0], torch.ones_like(edges[:, 0]), dtype=torch.long, device=device)
        np_edges = edges.detach().cpu().numpy()
        data = np.ones(len(np_edges))
        row_ind = np.arange(len(np_edges))
        col_ind = np_edges[:, 0]  # head
        M_sub = csr_matrix((data, (row_ind, col_ind)), shape=(len(np_edges), ent_num))
        return edges, M_sub

    def get_next_layer_nodes_edges(self, nodes, n_ent, M_sub, np_filter_edges):
        device = nodes.device
        np_nodes = nodes.detach().cpu().numpy()
        node_1hot = csr_matrix((np.ones(len(np_nodes)), (np_nodes[:, 1], np_nodes[:, 0])),
                               shape=(n_ent, np_nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)  #
        edges = np.nonzero(edge_1hot)  # x,y x-edge_id, y-batch_id
        np_selected_edges = np.concatenate([np.expand_dims(edges[1], 1), np_filter_edges[edges[0]]], axis=1)
        selected_edges = torch.tensor(np_selected_edges, dtype=torch.long, device=device)  # out links
        next_layer_nodes = torch.unique(selected_edges[:, [0, -1]], dim=0, sorted=False)  # tail, batch_id
        return next_layer_nodes, selected_edges