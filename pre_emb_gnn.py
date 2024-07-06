import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
from torch_scatter import scatter

class PreEmbLayer(torch.nn.Module):
    def __init__(self, hidden_dim, n_rel):
        super(PreEmbLayer, self).__init__()
        self.n_rel = n_rel
        self.hidden_dim = hidden_dim

        self.rela_embed = nn.Embedding(2 * n_rel + 1, self.hidden_dim)
        nn.init.xavier_normal_(self.rela_embed.weight.data)
        self.relation_linear = nn.Linear(self.hidden_dim, (2 * n_rel + 1) * self.hidden_dim)
        nn.init.xavier_normal_(self.relation_linear.weight.data)


    def forward(self, query, hidden, q_sub, q_rel, edges, ent_num, rela_embed=None, relation_linear=None, bid=None, layer=None):
        batch_size = q_sub.shape[0]

        if rela_embed is None:
            rela_embed = self.rela_embed
        rela_emb = rela_embed(edges[:, 1])
        if relation_linear is None:
            relation_linear = self.relation_linear
        rela_linear = relation_linear(query).view(batch_size, -1, self.hidden_dim)
        
        # gnn
        hidden = hidden  # (batch_size, n_ent, dim)
        head_emb = hidden[:,edges[:, 0], :]
        relation = rela_linear[:, edges[:, 1], :]  # (batch_size, edge_num, dim)
        message = head_emb + relation  # (batch_size, edge_num, dim)
        new_hidden = scatter(message, index=edges[:, -1], dim=1, dim_size=ent_num, reduce='mean')

        return new_hidden
