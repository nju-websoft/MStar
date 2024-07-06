import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
from torch_scatter import scatter
SPLIT = '*' * 30
from pre_emb_gnn import PreEmbLayer


class MultiConditionGNN(torch.nn.Module):
    eps = 1e-6
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, rela_independent, act=lambda x:x, total_rel=None, use_attn=True, mlp_num=2):
        super(MultiConditionGNN, self).__init__()
        self.message_func = 'distmult'  # transe distmult
        self.aggregate_func = 'sum'  # pna sum mean

        self.rela_independent = rela_independent
        self.hidden_dim = in_dim

        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        act = 'relu'
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        self.act = acts[act]

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.W_attn = nn.Linear(attn_dim, 1, bias=False)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        self.W_1 = nn.Linear(in_dim * 4, 1, bias=True)


        need_rel = total_rel
        self.need_rel = need_rel
        self.rela_embed = nn.Embedding(need_rel, in_dim)
        nn.init.xavier_normal_(self.rela_embed.weight.data)
        self.relation_linear = nn.Linear(in_dim, need_rel * out_dim)

        self.agg_linear = None
        if self.aggregate_func == 'pna':
            self.agg_linear = nn.Linear(self.hidden_dim * 12, self.hidden_dim)
        else:
            self.agg_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.use_attn = use_attn

        mlp = [nn.Identity()]
        for x in range(mlp_num):
            mlp.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
        mlp.append(nn.ReLU() if mlp_num != 0 else nn.Identity())
        self.mlp = nn.Sequential(*mlp)


    def forward(self, query, q_sub, q_rel, hidden, edges, nodes, rela=None):
        batch_size = hidden.shape[0]
        ent_num = hidden.shape[1]
        dim_size = hidden.shape[2]
        # edges: [b_id, h, r, t]
        id_bat = edges[:, 0]  # batch_id
        id_sub = edges[:, 1]
        id_rel = edges[:, 2]
        id_obj = edges[:, 3]
        id_que = q_rel[id_bat]
        all_ent_hidden = hidden.flatten(0, 1)

        message_head = torch.index_select(all_ent_hidden, index=id_sub, dim=0)
        if rela is None:
            if self.rela_independent:
                relation = self.relation_linear(query).view(batch_size, -1, self.hidden_dim)  # 2 * self.n_rel + 1
                rela_id = id_bat * self.need_rel + id_rel
                message_rela = torch.index_select(relation.flatten(0,1), index=rela_id, dim=0)
            else:
                message_rela = self.rela_embed(id_rel)
        else:
            if type(rela) == torch.nn.modules.linear.Linear:
                assert self.rela_independent
                relation = rela(query).view(batch_size, -1, self.hidden_dim)
                rela_id = id_bat * self.need_rel + id_rel
                message_rela = torch.index_select(relation.flatten(0, 1), index=rela_id, dim=0)
            else:
                message_rela = rela(id_rel)

        message_tail = torch.index_select(all_ent_hidden, index=id_obj, dim=0)
        # message_quer = self.rela_embed(id_que)
        message_quer = torch.index_select(query, index=id_bat, dim=0)

        if self.message_func == 'transe':
            mess = message_head + message_rela  
        elif self.message_func == 'distmult':
            mess = message_head * message_rela

        # attn: head, rela, que
        if self.use_attn:
            alpha_2 = torch.sigmoid(self.W_attn(nn.ReLU()(self.Ws_attn(message_head) + self.Wr_attn(message_rela) + self.Wqr_attn(message_quer))))
            message = mess * alpha_2
        else:
            message = mess


        id_mess = id_obj
        agg_dim_size = all_ent_hidden.shape[0]  
        message_agg = scatter(message, index=id_mess, dim=0, dim_size=agg_dim_size, reduce=self.aggregate_func)
        unique_id_mess = torch.unique(id_mess)


        select_nodes_hidden = message_agg[unique_id_mess, :]
        select_nodes_hidden = self.mlp(select_nodes_hidden)
        
        new_hidden = torch.zeros_like(message_agg)
        new_hidden[unique_id_mess, :] = select_nodes_hidden
        new_hidden = new_hidden.view(batch_size, ent_num, dim_size)
        return new_hidden


class FrameWork(torch.nn.Module):
    def __init__(self, params, loader):
        super(FrameWork, self).__init__()
        self.specific = params.specific
        self.high_way = params.high_way
        self.mlp_num = 2
        
        self.task = params.task
        self.method = params.method
        self.n_layer = params.n_layer
        self.rela_independent = params.rela_independent

        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader

        # Highway Layer
        if self.high_way:
            all_rel_num = (2 * self.n_rel + 1) + params.type_num
            self.short_gnn = MultiConditionGNN(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, rela_independent=False, 
                                        total_rel=all_rel_num, use_attn=False, mlp_num=0)
            self.high_way_rel = 2 * self.n_rel + 1
        else:
            all_rel_num = 2 * self.n_rel + 1
        
        # Multi-Condition GNN
        self.layers = []
        for i in range(self.n_layer):
            self.layers.append(MultiConditionGNN(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.rela_independent, 
                                            total_rel=all_rel_num, use_attn=True, mlp_num=self.mlp_num))
        self.layers = nn.ModuleList(self.layers)
        self.rela_embed = nn.Embedding(all_rel_num, self.hidden_dim)
        nn.init.xavier_normal_(self.rela_embed.weight.data)
        self.relation_linear = nn.Linear(self.hidden_dim, all_rel_num * self.hidden_dim)    

        # Decoder
        self.W_final = nn.Linear(self.hidden_dim * 2, 1, bias=False)  # get score
        mlp = []
        num_mlp_layer = 2
        mlp_hidden = self.hidden_dim * 2
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(mlp_hidden, mlp_hidden))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(mlp_hidden, 1))
        self.mlp = nn.Sequential(*mlp)

        # Pre-Embedded GNN
        self.role_num = params.type_num
        self.topk = params.type_topk
        self.role_emb = nn.Embedding(self.role_num, self.hidden_dim)  # head, similar-head
        self.role_layer_n = 6
        self.role_layers = [PreEmbLayer(self.hidden_dim, self.n_rel) for i in range(self.role_layer_n)]
        self.role_layers = nn.ModuleList(self.role_layers)
        if self.specific:  # Query-dependent Linear
            self.query_map_w = nn.Linear(self.hidden_dim, self.hidden_dim * self.hidden_dim)
            self.query_map_b = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Starting Entities Selection
        score = []
        score.append(nn.Linear(self.hidden_dim * 3, self.hidden_dim * 1))
        score.append(nn.ReLU())
        score.append(nn.Linear(self.hidden_dim * 1, 1))
        self.score = nn.Sequential(*score)
        self.role_classify = nn.Linear(self.hidden_dim, self.role_num - 1)

        if params.dropout is not None:
            self.dropout = nn.Dropout(params.dropout)
        else:
            self.dropout = nn.Identity()


    def forward(self, subs, rels, objs, work_mode='train', mode='transductive'):
        device = self.rela_embed.weight.data.device
        batch_size = len(subs)
        n_ent = self.loader.n_ent if work_mode in ['train', 'valid'] else self.loader.n_ent_ind
        q_sub = torch.tensor(subs, dtype=torch.long, device=device)
        q_rel = torch.tensor(rels, dtype=torch.long, device=device)
        q_obj = torch.tensor(objs, dtype=torch.long, device=device)
        if q_sub.shape != q_obj.shape:
            q_obj = None

        query = self.rela_embed(q_rel)
        filter_edges, M_sub = self.loader.get_edges(q_sub, q_rel, q_obj, mode=mode)
        np_filter_edges = filter_edges.detach().cpu().numpy()


        # Pre-Embedded GNN
        stage1_emb = torch.zeros((batch_size, n_ent, self.hidden_dim), device=device)
        for i in range(self.role_layer_n):
            stage1_emb = self.role_layers[i](query, stage1_emb, q_sub, q_rel, filter_edges, n_ent, relation_linear=self.relation_linear)

        query_emb = self.rela_embed(q_rel).unsqueeze(1).repeat_interleave(n_ent, dim=1)
        head_emb = stage1_emb[torch.arange(batch_size, device=device), q_sub, :].unsqueeze(1).repeat_interleave(n_ent, dim=1)
        stage1_emb = torch.cat([stage1_emb], dim=-1)
        if self.specific:
            stage1_emb_w = self.query_map_w(query).view(batch_size, self.hidden_dim, self.hidden_dim)
            stage1_emb_b = self.query_map_b(query).unsqueeze(1)
            stage1_emb_linear = stage1_emb@stage1_emb_w + stage1_emb_b
            stage1_emb = stage1_emb_linear

        nodes_head = torch.cat([torch.arange(batch_size, device=device).unsqueeze(-1), q_sub.unsqueeze(-1)], dim=-1)
        init_hidden = torch.zeros(batch_size, n_ent, self.hidden_dim, device=device)

        # Starting Entities Selection
        def method_base():
            nodes = nodes_head
            return nodes, None

        def method_mstar():
            scores = self.score(torch.cat([stage1_emb, head_emb, query_emb], dim=-1))
            scores = scores.squeeze(-1)
            scores[nodes_head[:, 0], nodes_head[:, 1]] = -10000  # exclude head
            _, argtopk = torch.topk(scores, k=self.topk, dim=-1)
            bid = torch.arange(batch_size, device=device).repeat_interleave(self.topk).unsqueeze(-1)
            eid = argtopk.flatten().unsqueeze(-1)
            entities = torch.cat([bid, eid], dim=-1)
            entity_emb = stage1_emb[entities[:, 0], entities[:, 1], :].view(batch_size, -1, self.hidden_dim)  # (batch_size, topk, Dim)
            logits = self.role_classify(entity_emb)
            r_hard = torch.argmax(logits, dim=-1) + 1  # type0->head
            r = (r_hard.unsqueeze(-1) - logits).detach() + logits
            nodes = torch.cat([nodes_head, entities], dim=0)
            return nodes, entities, r_hard

        def method_random_query():
            if self.topk == 0:
                return method_base()
            pseu_score = torch.rand((batch_size, n_ent), dtype=stage1_emb.dtype, device=device)
            pseu_score[nodes_head[:, 0], nodes_head[:, 1]] = -1000
            topk_simi, topk_ent_id = torch.topk(pseu_score, k=self.topk, dim=-1)
            select_head = topk_ent_id[:, :]
            select_head = select_head.flatten().unsqueeze(1)  # b0 b0, b1 b1, b2 b2, ...
            bid = torch.arange(batch_size, device=device).repeat_interleave(self.topk).unsqueeze(1)
            select_head = torch.cat([bid, select_head], dim=-1)
            nodes = torch.cat([nodes_head, select_head], dim=0)
            return nodes, select_head
        
        def method_degree_query():
            one_degree = torch.ones(filter_edges.shape[0], dtype=torch.float, device=device)
            degree = scatter(one_degree, index=filter_edges[:, 0], dim=0, dim_size=n_ent, reduce='sum')
            topk_degree, topk_ent_id = torch.topk(degree, k=self.topk, dim=-1)
            select_head = topk_ent_id.repeat(batch_size).unsqueeze(1)
            bid = torch.arange(batch_size, device=device).repeat_interleave(self.topk).unsqueeze(1)  # 0,0,0,..,0,1,1,1,...
            select_head = torch.cat([bid, select_head], dim=-1)
            nodes = torch.cat([nodes_head, select_head], dim=0)
            return nodes, select_head
        
        if self.method == "mstar":
            nodes, entities, r_type = method_mstar()
        elif self.method == "None":
            nodes, entities  = method_base()
        elif self.method == 'random_query':
            nodes, entities  = method_random_query()
            r_type = torch.ones(entities.shape[0], dtype=torch.long, device=entities.device)
        elif self.method == 'degree_query':
            nodes, entities  = method_degree_query()
            r_type = torch.ones(entities.shape[0], dtype=torch.long, device=entities.device)
        else:
            assert False

        def reidx(batch_id, node_id):
            return batch_id * n_ent + node_id
        
        init_hidden = torch.zeros(batch_size, n_ent, self.hidden_dim, device=device)  # if not high_way
        if self.high_way:
            init_hidden = torch.zeros(batch_size, n_ent, self.hidden_dim, device=device)
            bid = entities[:, [0]]
            high_way_rel = torch.ones_like(bid) * self.high_way_rel
            high_way_rel = high_way_rel -1 + r_type.flatten().unsqueeze(-1)
            high_way_edges = torch.cat([bid, q_sub[bid], high_way_rel, entities[:, [1]]], dim=1)  # bid, h, r, entities
            high_way_edges[:, 1] = reidx(high_way_edges[:, 0], high_way_edges[:, 1])
            high_way_edges[:, 3] = reidx(high_way_edges[:, 0], high_way_edges[:, 3])
            nodes_heads_emb = self.role_emb(torch.ones(1, dtype=torch.long, device=device) * 0)

            init_hidden[nodes_head[:, 0], nodes_head[:, 1], :] = nodes_heads_emb
            init_hidden = self.short_gnn(query, q_sub, q_rel, init_hidden, high_way_edges, nodes, rela=self.rela_embed)
            init_hidden[nodes_head[:, 0], nodes_head[:, 1], :] = nodes_heads_emb
        hidden = init_hidden

        # Multi-Condition GNN
        for layer_id in range(self.n_layer):
            next_layer_nodes, selected_edges = self.loader.get_next_layer_nodes_edges(nodes, n_ent, M_sub, np_filter_edges)
            nodes = next_layer_nodes

            # batch_id, h,r,t
            selected_edges[:, 1] = reidx(selected_edges[:, 0], selected_edges[:, 1])
            selected_edges[:, 3] = reidx(selected_edges[:, 0], selected_edges[:, 3])
            curr_edges = selected_edges
            # gnn
            rela = self.relation_linear if self.rela_independent else self.rela_embed
            new_hidden = self.layers[layer_id](query, q_sub, q_rel, hidden, curr_edges, nodes, rela)
            hidden = new_hidden
            hidden = self.dropout(hidden)

        # Decoder
        scores_all = torch.zeros((batch_size, n_ent), device=device)  # (batch_size, ent_num)
        visited = torch.zeros((batch_size, n_ent), dtype=torch.bool, device=device)
        visited[nodes[:, 0], nodes[:, 1]] = 1
        hidden = torch.cat([hidden, head_emb], dim=-1)
        scores_all = self.mlp(hidden).squeeze(-1)

        return scores_all, visited, None
