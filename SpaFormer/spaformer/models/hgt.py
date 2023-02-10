from dgl.nn.pytorch.conv import HGTConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation, NormLayer, create_norm

class HGT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 nhead,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(HGT, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        aggregator_type='mean'
        
        if num_layers == 1:
            if not last_norm:
                norm = create_norm(norm)(output_dim)
            else:
                norm = None
            self.layers.append(HGTConv(in_dim, nhead, out_dim, 1, 2))
            
        else:
            # input projection (no residual)
            self.layers.append(HGTConv(in_dim, nhead, num_hidden, 1, 2))
            
            # hidden layers
            for l in range(1, num_layers - 1):
                self.layers.append(HGTConv(num_hidden*nhead, nhead, num_hidden, 1, 2))
            
            # output projection
            self.layers.append(HGTConv(num_hidden*nhead, nhead, num_hidden, 1, 2))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        nt = torch.tensor([0]*g.num_nodes()).to(g.device)
        et = g.edata['type']
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h, nt, et, presorted=True)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
