from dgl.nn.pytorch.conv import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation, NormLayer, create_norm

class RGCN(nn.Module):
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
        super(RGCN, self).__init__()
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
            if not last_norm and (norm is not None):
                norm = True
            else:
                norm = False
            self.layers.append(RelGraphConv(in_dim, out_dim, 2, activation=last_activation, layer_norm=norm))
            
        else:
            if norm is not None:
                norm = True
            else:
                norm = False
            
            # input projection (no residual)
            self.layers.append(RelGraphConv(in_dim, num_hidden, 2, activation=create_activation(activation), layer_norm=norm))
            
            # hidden layers
            for l in range(1, num_layers - 1):
                self.layers.append(RelGraphConv(num_hidden, num_hidden, 2, activation=create_activation(activation), layer_norm=norm))
            
            # output projection
            self.layers.append(RelGraphConv(num_hidden, num_hidden, 2, activation=last_activation, layer_norm=norm))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        nt = torch.tensor([0]*g.num_nodes()).to(g.device)
        et = g.edata['type']
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h, et, presorted=True)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
