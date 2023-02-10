import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from performer_pytorch import SelfAttention
from dgl.nn.pytorch.conv import GINConv
import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h, local_gnn_type, global_model_type, num_heads, act=nn.ReLU,
                 pna_degrees=None, dropout=0.0, attn_dropout=0.0, log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.activation = act

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type != 'Transformer':
            raise NotImplementedError(
                "Logging of attention weights is only supported for "
                "Transformer global attention model."
            )

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GIN':
            gin_nn = nn.Sequential(nn.Linear(dim_h, dim_h),
                                   self.activation,
                                   nn.Linear(dim_h, dim_h))
            self.local_model = dglnn.GINConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = dglnn.GATConv(dim_h,
                                             dim_h // num_heads,
                                             num_heads,
                                             feat_drop=dropout, 
                                             attn_drop=attn_dropout,
                                             residual=True)
        elif local_gnn_type == 'SAGE':
            self.local_model = dglnn.SAGEConv(dim_h,
                                              dim_h,
                                              aggregator_type='mean',
                                              feat_drop=dropout)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            delta = torch.log1p(torch.from_numpy(np.array(pna_degrees)))
            self.local_model = dglnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             delta=delta,
                                             edge_feat_size=1)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=dropout, causal=False)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type



        # Normalization for MPNN and Self-Attention representations.
        self.norm1_local = nn.LayerNorm(dim_h)
        self.norm1_attn = nn.LayerNorm(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.norm2 = nn.LayerNorm(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, graph, h, edge_weight=None):
        h_in1 = h  # for first residual connection
        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            if self.local_gnn_type in ['GIN', 'SAGE', 'PNA']:
                h_local = self.local_model(graph, h, edge_weight)
            else:
                h_local = self.local_model(graph, h)
            if self.local_gnn_type == 'GAT':
                h_local = torch.flatten(h_local, 1)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local  # Residual connection.
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h, None, None)
            elif self.global_model_type == 'Performer':
                h = h.unsqueeze(0)
                h_attn = self.self_attn(h)
                h_attn = h_attn.squeeze(0)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        h = self.norm2(h)
        return h

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights = True`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 use_bn=False, use_ln=False, dropout=0.5, activation='relu',
                 residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


# class GIN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
#                  use_bn=True, dropout=0.5, activation='relu'):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         if use_bn: self.bns = nn.ModuleList()
#         self.use_bn = use_bn
#         # input layer
#         update_net = MLP(in_channels, hidden_channels, hidden_channels, 2,
#                          use_bn=use_bn, dropout=dropout, activation=activation)
#         self.layers.append(GINConv(update_net))
#         # hidden layers
#         for i in range(n_layers - 2):
#             update_net = MLP(hidden_channels, hidden_channels, hidden_channels,
#                              2, use_bn=use_bn, dropout=dropout,
#                              activation=activation)
#             self.layers.append(GINConv(update_net))
#             if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
#         # output layer
#         update_net = MLP(hidden_channels, hidden_channels, out_channels, 2,
#                          use_bn=use_bn, dropout=dropout, activation=activation)
#         self.layers.append(GINConv(update_net))
#         if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, g, x):
#         for i, layer in enumerate(self.layers):
#             if i != 0:
#                 x = self.dropout(x)
#                 if self.use_bn:
#                     if x.ndim == 2:
#                         x = self.bns[i - 1](x)
#                     elif x.ndim == 3:
#                         x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
#                     else:
#                         raise ValueError('invalid x dim')
#             x = layer(g, x)
#         return x

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, use_bn=True, dropout=0.5, activation='relu'):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError('Invalid activation')
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net, 'sum'))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
            self.layers.append(GINConv(update_net, 'sum'))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net, 'sum'))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i-1](x)
                    elif x.ndim == 3:
                        x = self.bns[i-1](x.transpose(2,1)).transpose(2,1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(g, x)
        return x
    
class GINDeepSigns(nn.Module):
    """ Sign invariant neural network with MLP aggregation.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 k, dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, 
#                  k, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):
#         super(GINDeepSigns, self).__init__()
#         self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
#         rho_dim = out_channels * k
#         self.rho = MLP(rho_dim, hidden_channels, k, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
#         self.k = k
        
#     def forward(self, g, x):
#         N = x.shape[0]  # Total number of nodes in the batch.
#         x = x.transpose(0, 1) # N x K x In -> K x N x In
#         x = self.enc(g, x) + self.enc(g, -x)
#         x = x.transpose(0, 1).reshape(N, -1)  # K x N x Out -> N x (K * Out)
#         x = self.rho(x)  # N x dim_pe (Note: in the original codebase dim_pe is always K)
#         return x
    
    def forward(self, g, x):
        x = x.unsqueeze(-1)  # (Num nodes) x (Num Eigenvectors) x 1
        x = self.enc(g, x) + self.enc(g, -x)
        orig_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = self.rho(x)
        x = x.reshape(orig_shape[0], -1)
        return x
    
class SignNetNodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet
    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.
    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.
    Args:
        dim_emb: Size of final node embedding
    """

    def __init__(self, dim_pe=16, eigen_k=10, layers=3, post_layers=2, model_type='MLP'):
        super().__init__()
        # dim_pe Size of PE embedding
        if model_type not in ['MLP', 'DeepSet']:
            raise ValueError(f"Unexpected SignNet model {model_type}")
        self.model_type = model_type
        sign_inv_layers = layers  # Num. layers in \phi GNN part
        rho_layers = post_layers  # Num. layers in \rho MLP/DeepSet
        if rho_layers < 1:
            raise ValueError(f"Num layers in rho model has to be positive.")
        max_freqs = eigen_k  # Num. eigenvectors (frequencies)

        phi_hidden_dim = 64
        phi_out_dim = 4

        # Sign invariant neural network.
        if self.model_type == 'MLP':
            self.sign_inv_net = GINDeepSigns(
                in_channels=1,
                hidden_channels=phi_hidden_dim,
                out_channels=phi_out_dim,
                num_layers=sign_inv_layers,
                k=max_freqs,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        elif self.model_type == 'DeepSet':
            self.sign_inv_net = MaskedGINDeepSigns(
                in_channels=1,
                hidden_channels=phi_hidden_dim,
                out_channels=phi_out_dim,
                num_layers=sign_inv_layers,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        else:
            raise ValueError(f"Unexpected model {self.model_type}")

    def forward(self, g):
        
        eigvec = g.ndata['eigvec']
        
        # pos_enc = torch.cat((eigvec.unsqueeze(2), eigval), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc = self.sign_inv_net(g, eigvec)  # (Num nodes) x (pos_enc_dim)
        return pos_enc