import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"
import time
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn
import warnings
warnings.filterwarnings("ignore", category=Warning)

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import utils
import params_helper
from load_data import DataLoader
from base_model import BaseModel


SPLIT = '*' * 30
parser = argparse.ArgumentParser(description="Parser for MStar")
parser.add_argument('--dataset', '-D', type=str, default='fb237_v1')  # fb237_v1 WN18RR_v1 nell_v1
parser.add_argument('--task', '-T', type=str, default='test', choices=['train', 'test'])
parser.add_argument('--exp', '-E', type=str, default="tmp")
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='test')

parser.add_argument('--method', '-M', type=str, default='mstar', choices=['None', 'mstar', 'random_query', 'degree_query'])
parser.add_argument('--train_bad', action='store_true', default=False) 
parser.add_argument('--high_way', '-HW',  action='store_true', default=False)
parser.add_argument('--remove_aware', action='store_true', default=False)  # whether use query-dependent relation embedding or not

parser.add_argument('--metric', type=str, default='mrr', choices=['mrr', 'hits@10'])
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--specific',  action='store_false', default=True)

parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--decay_rate', '-DR', type=float, default=None)
parser.add_argument('--lamb', type=float, default=None)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--early', type=int, default=None)

parser.add_argument('--hidden_dim', '-HD', type=int, default=None)
parser.add_argument('--attn_dim', type=int, default=None)
parser.add_argument('--type_topk', '-RT', type=int, default=None)
parser.add_argument('--type_num', type=int, default=None)
parser.add_argument('--n_layer', '-LN', type=int, default=None)
parser.add_argument('--n_batch', type=int, default=None)
args = parser.parse_args()
set_seed(args.seed)

base_path = ""
args.train_good = not args.train_bad
args.rela_independent = not args.remove_aware

# experiments
prefix = f"{args.dataset}_{args.method}_{args.high_way}_{args.exp}"
big_dataset = args.dataset[:-3]  # fb237/nell/WN18RR
exp_path = os.path.join(base_path, "experiments", big_dataset)
all_exp_dir = os.path.join(base_path, "experiments")
utils.check_dir(all_exp_dir)
bash_dir = os.path.join(base_path, "experiments", args.save_dir)
utils.check_dir(bash_dir)
exp_path = os.path.join(bash_dir, big_dataset)
utils.check_dir(exp_path)

# save_path
args.model_save_path = os.path.join(base_path, exp_path, f"{prefix}.pt")
args.logging_save_path = os.path.join(base_path, exp_path, f"{prefix}.log")
args.predict_save_path = os.path.join(base_path, exp_path, f"{prefix}_predict.txt")
logging.basicConfig(level=logging.INFO, filename=args.logging_save_path, filemode="a", format='%(message)s')
print(f'ModelSavePath: {args.model_save_path}\nLoggingSavePath: {args.logging_save_path}')

class Options(object):
    pass

opts = Options
var4 = ["lr", "decay_rate", "lamb", "dropout", "early"]
var5 = ["hidden_dim", "attn_dim", "type_topk", "type_num", "n_layer", "n_batch"]

def run_model(params):
    # pre-defined
    for p in var4+var5:
        setattr(opts, p, params[p])
    # only in args
    new_par = ['gpu', 'method', 'task', 'metric', 'specific', 'high_way']
    for p in new_par:
        setattr(opts, p, getattr(args, p))

    # output important params
    params1 = f"Dataset: {args.dataset}\t| Metric: {args.metric}\t| Method: {args.method}\t| UseHighWay: {args.high_way}"
    params2 = f"TrainGood: {args.train_good}\t| Independent: {args.rela_independent}\t| Specific: {args.specific}"
    params3 = f"Exp: {args.exp}\t| Task: {args.task}\t| GPU: {args.gpu}\t| ExpPath: {args.model_save_path}\t| Early: {params['early']}"
    params4 = "\t| ".join([f"{v}: {params[v]}" for v in var4])
    params5 = "\t| ".join([f"{v}: {params[v]}" for v in var5])
    content = "\n>>> ".join([params1, params2, params3, params4, params5])
    content = f">>> {content}\n{'=' * 50} Training {'=' * 50}"
    print(content)
    logging.info(content)


    opts.rela_independent = args.rela_independent
    loader = DataLoader(args, n_batch=opts.n_batch)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    model = BaseModel(opts, loader)
    if args.task == 'train':
        best_mrr = 0
        early_stop = 0
        early = params['early']
        start = time.time()
        for epoch in range(300):
            mrr, t_mrr, out_str = model.train_batch(args)
            if mrr > best_mrr:
                best_mrr = mrr
                best_str = out_str
                early_stop = 0
                print(f'[{epoch:<2d}] Better MRR!')
                logging.info(f'[{epoch:<2d}] Better MRR!')
                torch.save(model.model.state_dict(), args.model_save_path)
            else:
                early_stop += 1
            time_info = utils.output_time(start, time.time(), "", None)
            curr_content = f'[{epoch:<2d}] -{time_info} | early: {early_stop} | mrr: {mrr:.5f} | {out_str}'
            print(curr_content)
            logging.info(curr_content)
            if early_stop == early:
                print(f'[{epoch:<2d}] Early Stop!')
                logging.info(f'[{epoch:<2d}] Early Stop!')
                break
        output_info = f'v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050, t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050\n{best_str}'
        print(output_info)
        logging.info(output_info)

    elif args.task == 'test':
        model.model.load_state_dict(torch.load(args.model_save_path))
        mrr, h10, out_str = model.test(loader)
        print(f'Test: {out_str}')
        logging.info(f'Test: {out_str}')

        loader = DataLoader(args, n_batch=opts.n_batch, dist=[])
        all_dist = np.array(loader.ind_test_d)
        all_dist = all_dist[all_dist!=-1]
        max_d = int(all_dist.max())
        print(f'Max Distance: {max_d}')
        unique_dist = list(range(1, max_d+1)) + [-1]
        with open("test_results.txt", "a") as f:
            f.write(f"==================Test Result of {args.dataset} & Method {args.method}==================\n")
            f.write(f"Total Result MRR: {mrr:.4f}\t| H@10: {h10:.4f}| Exp: {args.model_save_path}\n")
            f.write(f"Dist\tNum \tMRR    \tH@10   \n")
            for d in unique_dist:
                dist = [d]
                loader = DataLoader(args, n_batch=opts.n_batch, dist=dist)
                if len(loader.ind_test) == 0:
                    f.write(f'{d:<4d}\t{len(loader.ind_test):<4d}\t{0:.4f}\t{0:.4f}\n')
                else:
                    mrr, h10, _ = model.test(loader)
                    f.write(f'{d:<4d}\t{len(loader.ind_test):<4d}\t{mrr:.4f}\t{h10:.4f}\n')
        print(f"Test Dist Over")
        
    return

content = f"{'=' * 50}Parameters{'=' * 50}"
print(content)
logging.info(content)
params = params_helper.set_params(args)
# modify parameters from argparse
for v in var4+var5:
    if getattr(args, v) is not None:
        params[v] = getattr(args, v)
        modify_content = f"[Modified] {v}: {params[v]}"
        print(modify_content)
        logging.info(modify_content)
        

t1 = time.time()
run_model(params)
utils.output_time(t1, time.time(), 'run_model', output=print)
utils.output_time(t1, time.time(), 'run_model', output=logging.info)


