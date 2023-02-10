import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation, NormLayer, create_norm
from performer_pytorch import SelfAttention
from .layers import SignNetNodeEncoder, positionalencoding2d
import dgl
import copy

class Performer(nn.Module):
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
                 aggr="sum",
                 pe=None,
                 cat_pe = False,
                 pe_aug = False,
              ):
        super(Performer, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.pe = pe
        self.cat_pe = cat_pe
        self.pe_aug = pe_aug

        if norm != 'groupnorm':
            self.norm0 = create_norm(norm)(num_hidden)
        else:
            self.norm0 = nn.GroupNorm(nhead, num_hidden)
        
        self.attlayers = nn.ModuleList()
        self.fflayers = nn.ModuleList() 
        self.norm1 = nn.ModuleList() 
        self.norm2 = nn.ModuleList() 
        for i in range(num_layers):
            self.attlayers.append(
                SelfAttention(
                    dim=num_hidden, heads=nhead,
                    dropout=dropout, causal=False)
            )
            
            self.fflayers.append(nn.Sequential(
                nn.Linear(num_hidden, num_hidden*4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(num_hidden*4, num_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

            if norm != 'groupnorm':
                self.norm1.append(create_norm(norm)(num_hidden))
                self.norm2.append(create_norm(norm)(num_hidden))
            else:
                self.norm1.append(nn.GroupNorm(nhead, num_hidden))
                self.norm2.append(nn.GroupNorm(nhead, num_hidden))
        
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
            self.pe_enc = nn.Embedding(11000, num_emb)
        elif self.pe == 'signnet':
            self.pe_enc = SignNetNodeEncoder(num_emb)
        elif self.pe == 'sinu':
            self.pe_enc = nn.Embedding.from_pretrained(positionalencoding2d(num_emb, 110, 100).flatten(1).T)
    
    def forward(self, g, inputs, no_pe=False):
        
        if self.pe is None:
            h = self.dropout(self.norm0(self.act(self.emb(inputs))))
        else:
            if self.pe_aug:
                if self.training:
                    g.ndata['pos'] += torch.rand((1,2), device=g.ndata['pos'].device)/10
                    g.ndata['pos'] += torch.randn(g.ndata['pos'].shape, device=g.ndata['pos'].device)/200
                else:
                    g.ndata['pos'] += 0.05
            if self.pe == 'lap':
                pe_input = g.ndata['eigvec'] * (torch.randint(0, 2, (g.ndata['eigvec'].shape[1], ), dtype=torch.float, device=inputs.device)[None, :]*2-1)
            elif self.pe == 'mlp':
                pe_input = g.ndata['pos']
            elif self.pe in ['bin', 'sinu']:
                x = g.ndata['pos'][:, 0]
                y = g.ndata['pos'][:, 1]
                x = (x * 100).long()
                y = (y * 100).long()
                x[x>=110] = 109
                y[y>=100] = 99
                x[x<0] = 0
                y[y<0] = 0
                pe_input = x*100+y
            elif self.pe == 'signnet':
                pe_input = g

            if no_pe:
                if not self.cat_pe:
                    h = self.norm0(self.dropout(self.emb(inputs)))
                else:
                    h = self.norm0(self.dropout(self.emb(torch.cat([inputs, torch.zeros_like(inputs)], 1))))
            else:
                if not self.cat_pe:
                    h = self.norm0(self.dropout(self.emb(inputs) + self.pe_enc(pe_input)))
                else:
                    h = self.norm0(self.dropout(torch.cat([self.emb(inputs), self.pe_enc(pe_input)], 1)))
                
        theta_list = []
        for l in range(self.num_layers):
            h = self.norm1[l](h + self.dropout(self.attlayers[l](h.unsqueeze(0))).squeeze(0))
            h = self.norm2[l](h + self.fflayers[l](h))
            
        return self.act(self.head(h))

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
