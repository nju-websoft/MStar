import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import cal_ranks, cal_performance
from tqdm import tqdm

from model_mstar import FrameWork as GNNModel
import logging

class BaseModel(object):
    def __init__(self, args, loader):
        self.metric = args.metric
        self.model = GNNModel(args, loader)
        self.model.to(args.gpu)
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.params = args

    def train_batch(self, args):
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)
        self.model.train()

        device = self.model.rela_embed.weight.data.device
        total_sample_num = 0
        bad_sample_num = 0
        shuffle_idx = np.random.permutation(self.n_train)


        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(self.n_train, (i+1)*batch_size)
            batch_idx = shuffle_idx[np.arange(start, end)]
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores, visited, case_study = self.model(triple[:,0], triple[:,1], triple[:,2], work_mode='train')
            pos_scores = scores[[torch.arange(len(scores), device=device),torch.tensor(triple[:,2], dtype=torch.long, device=device)]]
            pos_visited = visited[[torch.arange(len(scores)).to(device), torch.LongTensor(triple[:, 2]).to(device)]]
            if args.train_good:  # LinkVerify
                good_sample = pos_visited
                total_sample_num += len(batch_idx)
                bad_sample_num += len(batch_idx) - good_sample.sum().cpu().numpy().tolist()
                if good_sample.sum() == 0:
                    continue
                scores = scores[good_sample]
                pos_scores = pos_scores[good_sample]

            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))

            loss.backward()
            self.optimizer.step()

            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()


        self.scheduler.step()
        print(f"Learning Rate: {self.optimizer.state_dict()['param_groups'][0]['lr']:.12f}")
        valid_mrr, test_mrr, out_str = self.evaluate(args)
        if args.train_good:
            logging.info(f'total_sample-{total_sample_num}, bad_sample-{bad_sample_num}, bad/total-{100.0 * bad_sample_num / total_sample_num:.4f}%')
            print(f'total_sample-{total_sample_num}, bad_sample-{bad_sample_num}, bad/total-{100.0 * bad_sample_num / total_sample_num:.4f}%')

        out_str = f'loss: {epoch_loss:.4f} | {out_str}'
        return valid_mrr, test_mrr, out_str

    def evaluate(self, args):
        batch_size = self.n_batch
        def evaluate_dataset(n_data, data, n_ent, filter, work_mode, mode):
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []
            masks = []
            self.model.eval()
            for i in range(n_batch):
                start = i * batch_size
                end = min(n_data, (i + 1) * batch_size)
                batch_idx = np.arange(start, end)
                subs, rels, objs = self.loader.get_batch(batch_idx, data=data)

                scores, visited, case_study = self.model(subs, rels, objs, work_mode=work_mode, mode=mode)
                scores = scores.data.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = filter[(subs[i], rels[i])]
                    filt_1hot = np.zeros((n_ent,))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                    masks += [n_ent - len(filt)] * int(objs[i].sum())
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
 
                ranking += ranks
            ranking = np.array(ranking)
            info = cal_performance(ranking, masks)
            # v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050 = info
            return ranking, info

        # valid & test
        v_ranking, v_info = evaluate_dataset(self.n_valid, 'valid', self.n_ent, self.loader.val_filters, work_mode='valid', mode='transductive')
        v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050 = v_info
        t_ranking, t_info = evaluate_dataset(self.n_test, 'test', self.n_ent_ind, self.loader.tst_filters, work_mode='test', mode='inductive')
        t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050 = t_info

        out_str = ('valid: %.4f %.1f %.4f %.4f %.4f %.4f | '
                   'test: %.4f %.1f %.4f %.4f %.4f %.4f') % (
                  v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050, t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050)
        
        v_metric = None
        t_metric = None
        if self.metric == 'hits@10':
            v_metric = v_h10
            t_metric = t_h10
        elif self.metric == 'mrr':
            v_metric = v_mrr
            t_metric = t_mrr
        return v_metric, t_metric, out_str

    def test(self, loader):
        batch_size = self.n_batch
        self.model.loader = loader
        self.loader = loader
        self.n_test = loader.n_test

        # print(f"n_test: {self.n_test}")
        def evaluate_dataset(n_data, data, n_ent, filter, work_mode, mode):
            n_batch = n_data // batch_size + (n_data % batch_size > 0)
            ranking = []
            masks = []
            self.model.eval()
            curr_num = 0
            for i in range(n_batch):
                start = i * batch_size
                end = min(n_data, (i + 1) * batch_size)
                batch_idx = np.arange(start, end)
                curr_num += len(batch_idx)
                # print(f'curr total triplets: {curr_num}')
                subs, rels, objs = self.loader.get_batch(batch_idx, data=data)

                scores, visited, case_study = self.model(subs, rels, objs, work_mode=work_mode, mode=mode)
                scores = scores.data.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = filter[(subs[i], rels[i])]
                    filt_1hot = np.zeros((n_ent,))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                    masks += [n_ent - len(filt)] * int(objs[i].sum())
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks
            ranking = np.array(ranking)
            info = cal_performance(ranking, masks)
            return ranking, info

        t_ranking, t_info = evaluate_dataset(self.n_test, 'test', self.n_ent_ind, self.loader.tst_filters,
                                             work_mode='test', mode='inductive')
        t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050 = t_info

        out_str = f"{t_mrr:.4f} {t_mr:.4f} {t_h1:.4f} {t_h3:.4f} {t_h10:.4f} {t_h1050:.4f}"
        return t_mrr, t_h10, out_str

