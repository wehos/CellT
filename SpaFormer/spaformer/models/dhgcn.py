from dgl.nn.pytorch.conv import GATv2Conv, GATConv, SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation, NormLayer, create_norm
from dgl.nn.pytorch.factory import KNNGraph
#from tsne_torch import TorchTSNE as TSNE
import dgl
import copy

#def create_layers(layers, num_layers, num_hidden, num_heads):
#    layers.append(SAGEConv(num_hidden, num_hidden, 'mean'))
#    for l in range(1, num_layers - 1):
#        layers.append(SAGEConv(num_hidden, num_hidden, 'mean'))
#    layers.append(SAGEConv(num_hidden, num_hidden, 'mean'))

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, nhead):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
    
    def forward(self, h ):
        return self.layer(h)

def create_layers(layers, model_class, num_layers, in_dim, out_dim, num_hidden, num_heads):
    head_hid = int(num_hidden/num_heads)
    layers.append(model_class(in_dim, head_hid, num_heads))
    for l in range(1, num_layers-1):
        layers.append(model_class(num_hidden, head_hid, num_heads))
    layers.append(model_class(num_hidden, out_dim, 1))

class DHGCN(nn.Module):
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
                 latent_dim = 20,
                 ):
        super(DHGCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        aggregator_type='mean'
        
        self.norm = nn.ModuleList()
        for i in range(num_layers-1):
            if norm is not None:    
                if norm != 'groupnorm':
                    self.norm.append(create_norm(norm)(num_hidden))
                else:
                    self.norm.append(nn.GroupNorm(nhead, num_hidden))
            else:
                self.norm.append(nn.Identity())
        if norm is not None:
            if norm != 'groupnorm':
                self.norm.append(create_norm(norm)(latent_dim))
            else:
                self.norm.append(nn.GroupNorm(nhead, latent_dim))
        else:
            self.norm.append(nn.Identity())
        # input projection (no residual)
        #self.layers1.append(GATv2Conv(num_hidden, num_hidden, num_heads=nhead, activation=create_activation(activation)))
        # hidden layers
        #for l in range(1, num_layers - 1):
        #    self.layers1.append(GATv2Conv(num_hidden, num_hidden, num_heads=nhead, activation=create_activation(activation)))
        # output projection
        #self.layers1.append(GATv2Conv(num_hidden, num_hidden, num_heads=nhead, activation=last_activation))
        
        # input projection (no residual)
        #self.layers2.append(GATv2Conv(num_hidden, num_hidden, num_heads=nhead, activation=create_activation(activation)))
        # hidden layers
        #for l in range(1, num_layers - 1):
        #    self.layers2.append(GATv2Conv(num_hidden, num_hidden, num_heads=nhead, activation=create_activation(activation)))
        # output projection
        #self.layers2.append(GATv2Conv(num_hidden, num_hidden, num_heads=nhead, activation=last_activation))
        
        create_layers(self.layers1, GATv2Conv, num_layers, in_dim, latent_dim, num_hidden, nhead)
        create_layers(self.layers2, GATConv,  num_layers, in_dim, latent_dim, num_hidden, nhead)
        create_layers(self.layers3, LinearLayer, num_layers, in_dim, latent_dim,  num_hidden, 1)
        
        self.head = nn.Identity()
        #self.head = nn.Linear(num_hidden, 15)
        self.emb = nn.Linear(in_dim, num_hidden)
        self.gate = nn.ModuleList([nn.Linear(i, 2) for i in ([num_hidden]*(num_layers-1)+[latent_dim])])
        self.act = create_activation(activation)
    
    def forward(self, g, inputs):
        
        #h = self.act(self.norm(self.emb(inputs)))
        h = inputs
        theta_list = []
        for l in range(self.num_layers):
            if True:#h.shape[1]>128:
                with torch.no_grad():
                    _, _, V = torch.pca_lowrank(inputs, q=64)
                    loc = inputs @ V[:, :64]
                    #loc = F.normalize(inputs, dim=0)#
                #loc = torch.cat([loc, h.detach()], 1)
                #loc += h.detach()
            #else:
            #    loc = h.detach()
            
            kg = KNNGraph(10)
            h0 = self.layers3[l](h)
            #h0 = h

            h = F.dropout(h, p=self.dropout, training=self.training)
            
            minl = 1
            maxl = 3
            if l>=minl and l<=maxl:
                h1 = self.layers1[l](g, h)
                new_g = kg(loc, algorithm='bruteforce-sharemem', dist='cosine')
                h2 = self.layers2[l](new_g, h)
                if len(h1.shape)==3:
                    h1 = torch.flatten(h1, 1)
                    h2 = torch.flatten(h2, 1)
            #h1 = self.act(self.norm(h1))
            #h2 = self.act(self.norm(h2))

            theta = torch.softmax(self.gate[l](h0), 1)
            #theta = h0
            if l>=minl and l<=maxl:
                h = h0 #+ h2
                #h = h1 + h2
                #h = h0 + theta[:, 0:1]*h1 + theta[:, 1:2]*h2
                #h = h0 + h1 + h2
            else:
                #h = theta[:, 2:3] * h0 + theta[:, 0:1]*h1 + theta[:, 1:2]*h2
                h = h0
                #h =   h0 + h1 + h2
            h = self.act(self.norm[l](h))

            theta_list.append(theta.detach())
            
        # output projection
        return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
