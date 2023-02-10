from dgl.nn.pytorch.conv import GATv2Conv, GATConv, SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation, NormLayer, create_norm
from dgl.nn.pytorch.factory import KNNGraph
#from tsne_torch import TorchTSNE as TSNE
import dgl
import copy
from .layers import GPSLayer

#def create_layers(layers, num_layers, num_hidden, num_heads):
#    layers.append(SAGEConv(num_hidden, num_hidden, 'mean'))
#    for l in range(1, num_layers - 1):
#        layers.append(SAGEConv(num_hidden, num_hidden, 'mean'))
#    layers.append(SAGEConv(num_hidden, num_hidden, 'mean'))

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, nhead):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(in_dim, in_dim)
    
    def forward(self, h ):
        return self.layer(h)

def create_layers(layers, model_class, num_layers, num_hidden, num_heads):
    head_hid = int(num_hidden/num_heads)
    layers.append(model_class(num_hidden, head_hid, num_heads))
    for l in range(1, num_layers-1):
        layers.append(model_class(num_hidden, head_hid, num_heads))
    layers.append(model_class(num_hidden, head_hid, num_heads))

class GPS(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 nhead,
                 num_layers,
                 dropout,
                 activation,
                 attn_drop,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr = "SAGE",
                 pe=None,
                 cat_pe = False,
              ):
        super(GPS, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.pe = pe
        self.cat_pe = cat_pe
        
        if norm != 'groupnorm':
            self.norm0 = create_norm(norm)(num_hidden)
        else:
            self.norm0 = nn.GroupNorm(nhead, num_hidden)
        
        self.gpslayers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gpslayers.append(GPSLayer(num_hidden, aggr, 'Performer', nhead, act=nn.GELU(),
                 pna_degrees=None, dropout=dropout, attn_dropout=attn_drop, log_attn_weights=False))
        
        if cat_pe and pe is not None:
            num_emb = num_hidden // 2
        else:
            num_emb = num_hidden
        self.head = nn.Identity() if num_hidden == out_dim else nn.Linear(num_hidden, out_dim)
        self.emb = nn.Linear(in_dim, num_emb)
        self.act = create_activation(activation)
        
        if self.pe == 'lap':
            self.pe_enc = nn.Linear(10, num_emb)
        elif self.pe == 'mlp':
            self.pe_enc = nn.Linear(2, num_emb)
        elif self.pe == 'bin':
            self.pe_enc = nn.Embedding(10000, num_emb)
        elif self.pe == 'signnet':
            self.pe_enc = SignNetNodeEncoder(num_emb)
        
    def forward(self, g, inputs):
        
        if self.pe is None:
            h = self.dropout(self.norm0(self.act(self.emb(inputs))))
        else:
            if self.pe == 'lap':
                pe_input = g.ndata['eigvec'] * (torch.randint(0, 2, (g.ndata['eigvec'].shape[1], ), dtype=torch.float, device=inputs.device)[None, :]*2-1)
            elif self.pe == 'mlp':
                pe_input = g.ndata['pos']
            elif self.pe == 'bin':
                x = g.ndata['pos'][:, 0]
                y = g.ndata['pos'][:, 1]
                x = (x * 100).long()
                y = (y * 100).long()
                x[x==100] = 99
                y[y==100] = 99
                pe_input = x*100+y
            elif self.pe == 'signnet':
                pe_input = g
            if not self.cat_pe:
                h = self.dropout(self.norm0(self.act(self.emb(inputs) + self.act(self.pe_enc(pe_input)))))
            else:
                h = self.dropout(self.norm0(self.act(torch.cat([self.emb(inputs), self.pe_enc(pe_input)], 1))))
                
        theta_list = []
        for l in range(self.num_layers):
            h = h + self.gpslayers[l](g, h)
            
        return self.act(self.head(h))

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
