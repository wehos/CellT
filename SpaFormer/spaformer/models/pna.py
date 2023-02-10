from dgl.nn.pytorch.conv import PNAConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation, NormLayer, create_norm

class PNA(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(PNA, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            if not last_norm:
                self.norm = create_norm(norm)(output_dim)
            else:
                self.norm = create_norm(None)(output_dim)
            self.layers.append(PNAConv(in_dim, out_dim, ['mean', 'max', 'min', 'std'], ['identity', 'attenuation', 'amplification'], 10, residual=last_residual))
            
        else:
            # input projection (no residual)
            
            self.norm=create_norm(norm)(num_hidden)
            print(norm, self.norm)
            self.layers.append(PNAConv(in_dim, num_hidden, ['mean', 'max', 'min', 'std'], ['identity', 'attenuation', 'amplification'], 10, residual=residual))
            
            # hidden layers
            for l in range(1, num_layers - 1):
                self.layers.append(PNAConv(num_hidden, num_hidden, ['mean', 'max', 'min', 'std'], ['identity', 'attenuation', 'amplification'], 10, residual=residual, num_towers=5))
            
            # output projection
            self.layers.append(PNAConv(num_hidden, out_dim, ['mean', 'max', 'min', 'std'], ['identity', 'attenuation', 'amplification'], 10, residual=last_residual))
            
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.norm(self.layers[l](g, h))
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
