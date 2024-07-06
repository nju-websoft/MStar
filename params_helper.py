
def wn(args, params):
    if 'WN18RR_v1' in args.dataset:
        params['lr'] = 0.001
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.0001
        params["dropout"] = None
        params['early'] = 5

        params['hidden_dim'] = 32
        params['attn_dim'] = 5
        params['type_topk'] = 32
        params['type_num'] = 3
        params['n_layer'] = 5
        params['n_batch'] = 10

        

    if 'WN18RR_v2' in args.dataset:
        params['lr'] = 0.0021
        params['decay_rate'] = 0.99
        params['lamb'] = 0.002
        params["dropout"] = 0.05
        params['early'] = 5

        params['hidden_dim'] = 64
        params['attn_dim'] = 3
        params['type_topk'] = 16
        params['type_num'] = 2
        params['n_layer'] = 6
        params['n_batch'] = 20


    if 'WN18RR_v3' in args.dataset:
        params['lr'] = 0.005
        params['decay_rate'] = 0.9
        params['lamb'] = 0.01
        params["dropout"] = None
        params['early'] = 5

        params['hidden_dim'] = 32
        params['attn_dim'] = 5
        params['type_topk'] = 32
        params['type_num'] = 5
        params['n_layer'] = 5
        params['n_batch'] = 20


    if 'WN18RR_v4' in args.dataset:
        params['lr'] = 0.005
        params['decay_rate'] = 0.99
        params['lamb'] = 0.1
        params["dropout"] = None
        params['early'] = 5

        params['hidden_dim'] = 64
        params['attn_dim'] = 3
        params['type_topk'] = 32
        params['type_num'] = 9
        params['n_layer'] = 5
        params['n_batch'] = 64

    return params

def fb(args, params):
    if 'fb237_v1' in args.dataset:
        params['lr'] = 0.001
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.0001
        params["dropout"] = None
        params['early'] = 5

        params['hidden_dim'] = 64
        params['attn_dim'] = 5
        params['type_topk'] = 64
        params['type_num'] = 3
        params['n_layer'] = 3
        params['n_batch'] = 20


    if 'fb237_v2' in args.dataset:
        params['lr'] = 0.005
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.000025
        params["dropout"] = None
        params['early'] = 5

        params['hidden_dim'] = 32
        params['attn_dim'] = 5
        params['type_topk'] = 64
        params['type_num'] = 5
        params['n_layer'] = 3
        params['n_batch'] = 64


    if 'fb237_v3' in args.dataset:
        params['lr'] = 0.005
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.0005
        params["dropout"] = None
        params['early'] = 5

        params['hidden_dim'] = 32
        params['attn_dim'] = 5
        params['type_topk'] = 64
        params['type_num'] = 5
        params['n_layer'] = 3
        params['n_batch'] = 64


    if 'fb237_v4' in args.dataset:
        params['lr'] = 0.005
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.000186
        params["dropout"] = 0.07
        params['early'] = 5

        params['hidden_dim'] = 32
        params['attn_dim'] = 5
        params['type_topk'] = 64
        params['type_num'] = 5
        params['n_layer'] = 3
        params['n_batch'] = 64

    return params

def nl(args, params):
    if 'nell_v1' in args.dataset:
        params['lr'] = 0.0001
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.01
        params["dropout"] = None
        params['early'] = 2

        params['hidden_dim'] = 32
        params['attn_dim'] = 5
        params['type_topk'] = 1
        params['type_num'] = 2
        params['n_layer'] = 3
        params['n_batch'] = 20


    if 'nell_v2' in args.dataset:
        params['lr'] = 0.005
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.0001
        params["dropout"] = 0.05
        params['early'] = 2

        params['hidden_dim'] = 32
        params['attn_dim'] = 5
        params['type_topk'] = 4
        params['type_num'] = 9
        params['n_layer'] = 3
        params['n_batch'] = 20


    if 'nell_v3' in args.dataset:
        params['lr'] = 0.002
        params['decay_rate'] = 0.99
        params['lamb'] = 0.02
        params["dropout"] = None
        params['early'] = 5

        params['hidden_dim'] = 48
        params['attn_dim'] = 3
        params['type_topk'] = 64
        params['type_num'] = 5
        params['n_layer'] = 3
        params['n_batch'] = 20


    if 'nell_v4' in args.dataset:
        params['lr'] = 0.005
        params['decay_rate'] = 0.9980
        params['lamb'] = 0.1
        params["dropout"] = 0.03
        params['early'] = 5

        params['hidden_dim'] = 48
        params['attn_dim'] = 5
        params['type_topk'] = 8
        params['type_num'] = 2
        params['n_layer'] = 3
        params['n_batch'] = 20
        
    return params

def set_params(args):
    params = {}
    if args.dataset.startswith("WN18RR"):
        params = wn(args, params)
    elif args.dataset.startswith("fb237"):
        params = fb(args, params)
    elif args.dataset.startswith("nell"):
        params = nl(args, params)
    else:
        assert False

    print(params)
    return params