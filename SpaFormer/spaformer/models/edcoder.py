from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import pickle
#from sklearn.cluster import KMeans
#from sklearn.metrics.cluster import normalized_mutuscore
#from sklearn.metrics import adjusted_rand_score
#from sklearn.preprocessing import LabelEncoder

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .pna import PNA
from .sage import SAGE
from .rgcn import RGCN
from .dhgcn import DHGCN
from .gps import GPS
from .performer import Performer
from .hgt import HGT
from .dot_gat import DotGAT
from .loss_func import sce_loss
from graphmae.utils import create_norm
from graphmae.evaluation import comprehensive_evaluate
from dgl import DropNode
from .relative_performer import RelativePerformer

class MLP(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout, nhead, norm):
        super().__init__()
        self.layers = nn.ModuleList()
        assert num_layers > 1, 'At least two layers for encoder.'
        for i in range(num_layers-1):
            layer_in = in_dim if i==0 else num_hidden
            self.layers.append(nn.Sequential(
                nn.Linear(layer_in, num_hidden),
                nn.PReLU(),
                nn.Dropout(dropout),
                create_norm(norm)(num_hidden) if norm!='groupnorm' else nn.GroupNorm(nhead, num_hidden)
            ))
        self.out_layer = nn.Sequential(
            nn.Linear(num_hidden*(num_layers-1), out_dim),
            nn.PReLU(),
        )

    def forward(self, g, x):
        hist = []
        for layer in self.layers:
            x = layer(x)
            hist.append(x)
        return self.out_layer(torch.cat(hist, 1))
    
def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, norm, nhead, nhead_out, attn_drop, pe=None, negative_slope=0.2, concat_out=True, latent_dim=20, cat_pe=False, pe_aug=False, aggr='SAGE') -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == 'pna':
        mod = PNA(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == 'sage':
        mod = SAGE(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "hgt":
        mod = HGT(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            nhead=nhead,
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            norm=norm,
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "dhgcn":
        mod = DHGCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            nhead=nhead,
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            norm=norm,
            encoding=(enc_dec == "encoding"),
            latent_dim = latent_dim
        )
    elif m_type in ["gps", "graphgps"]:
        mod = GPS(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            nhead=nhead,
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            attn_drop=attn_drop,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            pe = pe,
            cat_pe = cat_pe,
            aggr = aggr,
        )
    elif m_type == "performer":
        mod = Performer(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            nhead=nhead,
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            attn_drop=attn_drop,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            pe = pe,
            cat_pe = cat_pe,
            pe_aug = pe_aug,
        )
    elif m_type == "relative_performer":
        mod = RelativePerformer(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            nhead=nhead,
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            attn_drop=attn_drop,
            norm=norm,
        )
    elif m_type == "rgcn":
        mod = RGCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            nhead=nhead,
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = MLP(in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            nhead=nhead,
            norm=create_norm(norm),
            )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod

class MeanAct(nn.Module):
    """Mean activation class."""

    def __init__(self, softmax, standardscale):
        super().__init__()
        self.standardscale = standardscale
        self.softmax = softmax

    def forward(self, x):
        if not self.softmax:
            return torch.clamp(torch.exp(x), min=1e-5, max=1e6)
        else:
            return torch.softmax(x, 1) * self.standardscale

class DispAct(nn.Module):
    """Dispersion activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

class ZINB(nn.Module):
    """ZINB Decoder.
    Parameters
    ----------
    input_dim : int
        dimension of input feature.
    n_z : int
        dimension of latent embedding.
    n_dec_1 : int optional
        number of nodes of decoder layer 1.
    n_dec_2 : int optional
        number of nodes of decoder layer 2.
    """

    def __init__(self, n_z, input_dim, standardscale, n_dec_1=128, n_dec_2=128, softmax=True, disp='gene_cell'):
        super().__init__()
        self.input_dim = input_dim
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_mean = nn.Sequential(nn.Linear(n_dec_2, self.input_dim), MeanAct(softmax, standardscale))
        self.dec_pi = nn.Sequential(nn.Linear(n_dec_2, self.input_dim), nn.Sigmoid())
        self.disp = disp
        if disp == 'gene':
            self.dec_disp = nn.Parameter(torch.ones(input_dim))
        else:
            self.dec_disp = nn.Sequential(nn.Linear(n_dec_2, self.input_dim), DispAct())

    def forward(self, z):
        """Forward propagation.
        Parameters
        ----------
        z :
            embedding.
        Returns
        -------
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB4
        """
        
        h = F.relu(self.dec_1(z))
#         h = F.relu(self.dec_2(h))
        _mean = self.dec_mean(h)
        if self.disp == 'gene':
            _disp = self.dec_disp.repeat(z.shape[0], 1)
        else:
            _disp = self.dec_disp(h)
        _pi = self.dec_pi(h)
        return _mean, _disp, _pi

class LatentModel(nn.Module):
    def __init__(self, h_dim, z_dim, kl_weight=1e-6, warmup_step=10000):
        super().__init__()
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        self.kl = 0
        self.kl_weight = kl_weight
        self.step_count = 0
        self.warmup_step = warmup_step
    
    def kl_schedule_step(self):
        self.step_count += 1
        if self.step_count < self.warmup_step:
            self.kl_weight = self.kl_weight + (1e-2 - 1e-6) / self.warmup_step
        elif self.step_count == self.warmup_step:
            pass
#             self.step_count = 0
#             self.kl_weight = 1e-6
        
    def forward(self, h):
        mu = self.hid_2mu(h)
        log_var = self.hid_2sigma(h)
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        
        if self.training:
            z = mu + sigma*epsilon
            self.kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum() / z.shape[0]  * self.kl_weight
            self.kl_schedule_step()
        else:
            z = mu
        return z
    
class ZINBLoss(nn.Module):
    """ZINB loss class."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mean, disp, pi, scale_factor, ridge_lambda=0.0):
        """Forward propagation.
        Parameters
        ----------
        x :
            input features.
        mean :
            data mean.
        disp :
            data dispersion.
        pi :
            data dropout probability.
        scale_factor : list
            scale factor of mean.
        ridge_lambda : float optional
            ridge parameter.
        Returns
        -------
        result : float
            ZINB loss.
        """
        eps = 1e-10
        scale_factor = scale_factor.unsqueeze(-1)
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result
    
class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            norm: Optional[str],
            mask_node_rate: float = 0.5,
            mask_feature_rate: float = 0.5,
            encoder_type: str = "gat",
            decoder_type: str = "zinb",
            loss_fn: str = "sce",
            alpha_l: float = 2,
            concat_hidden: bool = False,
            latent_dim : int = 20,
            pe = None,
            drop_node_rate : float = 0.,
            objective : str = 'mask', 
            standardscale : int = 250, 
            cat_pe = False,
            pe_aug = False,
            aggr = 'SAGE',
         ):
        super(PreModel, self).__init__()
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        self._drop_node_rate = drop_node_rate
        self._drop_model = DropNode(drop_node_rate)
        self._objective = objective
        
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat", 'hgt'):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        elif encoder_type == 'dhgcn':
            enc_num_hidden = num_hidden
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = nhead

        dec_in_dim = latent_dim
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 
        if objective in ['vae', 'maskvae']:
            enc_out_dim = enc_num_hidden
        else:
            enc_out_dim = latent_dim
            
        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_out_dim,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            norm=norm,
            latent_dim = latent_dim,
            pe = pe,
            cat_pe = cat_pe,
            pe_aug = pe_aug,
            aggr = aggr,
        )


        if decoder_type == 'zinb':
            self.decoder = ZINB(dec_in_dim, in_dim, standardscale=standardscale)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(dec_in_dim, num_hidden),
                nn.PReLU(),
                nn.Dropout(feat_drop),
                nn.Linear(num_hidden, in_dim),
                nn.ReLU()
            )
            
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        
        if objective in ['mask', 'ae']:
            self.encoder_to_decoder = nn.Identity()
        elif objective in ['maskvae', 'vae']:
            self.encoder_to_decoder = LatentModel(enc_num_hidden, dec_in_dim)
        
#         if concat_hidden:
#             self.encoder_to_decoder = nn.Linear(num_hidden * num_layers, dec_in_dim, bias=False)
#         else:
#             self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.zinb_loss = ZINBLoss()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def weighted_mse_loss(self, input, target):
        temp = (input - target) ** 2
        return torch.mean( (target==0) * temp * 0.5 + (target>0) * temp)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "wmse":
            criterion = self.weighted_mse_loss#nn.MSELoss()
        elif loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, renorm = 'post'):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(self._mask_node_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if renorm == 'post':
            out_x = F.dropout(x, p=self._mask_feature_rate) 
            out_x[keep_nodes] = x[keep_nodes]
        elif renorm == 'raw':
            out_x = F.dropout(g.ndata['raw'], p=self._mask_feature_rate) 
            out_x[keep_nodes] = g.ndata['raw'][keep_nodes]
            out_x = out_x * ((torch.exp(x)-1).sum() / out_x.sum()) 
            out_x = torch.log(out_x+1)
        else:
            out_x = F.dropout(x, p=self._mask_feature_rate) * (1-self._mask_feature_rate)
            out_x[keep_nodes] = x[keep_nodes]

        use_g = g.clone()
        use_g.ndata['input'] = out_x
        use_g.ndata['masked'] = torch.zeros((x.shape[0],), device=x.device)
        use_g.ndata['masked'][mask_nodes] = 1
        use_g = self._drop_model(use_g)
        out_x = use_g.ndata['input']
        mask_nodes = torch.nonzero(use_g.ndata['masked']==1, as_tuple=True)[0]
        keep_nodes = torch.nonzero(use_g.ndata['masked']==0, as_tuple=True)[0]
        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x):
        if self._objective in ['mask', 'maskvae']:
            use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x)
        else:
            use_g, use_x = g, x
        enc_rep = self.encoder(use_g, use_x)
            
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)
        if self._decoder_type == 'mlp':
            recon = self.decoder(rep)
            mse = nn.MSELoss()
            if self._objective == 'mask':
                loss = mse(recon[mask_nodes], use_g.ndata['feat'][mask_nodes])
            elif self._objective == 'maskvae':
                loss = mse(recon[mask_nodes], use_g.ndata['feat'][mask_nodes]) + self.encoder_to_decoder.kl
            elif self._objective == 'ae':
                loss = mse(recon, use_g.ndata['feat'])
            elif self._objective == 'vae':
                loss = mse(recon, use_g.ndata['feat']) + self.encoder_to_decoder.kl
            
        elif self._decoder_type == 'zinb':
            mean, disp, pi = self.decoder(rep)    
            if self._objective == 'mask':
                loss = self.zinb_loss(use_g.ndata['raw'][mask_nodes], mean[mask_nodes], disp[mask_nodes], pi[mask_nodes], use_g.ndata['size_factor'][mask_nodes])
            elif self._objective == 'maskvae':
                loss = self.zinb_loss(use_g.ndata['raw'][mask_nodes], mean[mask_nodes], disp[mask_nodes], pi[mask_nodes], use_g.ndata['size_factor'][mask_nodes]) + self.encoder_to_decoder.kl
            elif self._objective == 'ae':
                loss = self.zinb_loss(use_g.ndata['raw'], mean, disp, pi, use_g.ndata['size_factor'])
            elif self._objective == 'vae':
                loss = self.zinb_loss(use_g.ndata['raw'], mean, disp, pi, use_g.ndata['size_factor']) + self.encoder_to_decoder.kl 
        return loss
         
    def evaluate(self, g, x, mode='valid', mask=None, no_pe=False):
        assert mode in ['valid', 'test', 'infer'], 'Invalid evaluation mode.'
            
        if mode == 'valid' and self._objective in ['mask', 'maskvae']:
            use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x)   
        else:
            use_g, use_x = g, x
            mask_nodes = np.arange(g.num_nodes())

        if self._encoder_type == 'performer':
            enc_rep = self.encoder(use_g, use_x, no_pe)
        else:
            assert not no_pe, '"no_pe" only supported for performers.'
            enc_rep = self.encoder(use_g, use_x)
            
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type == "zinb":
            mu, _, pi = self.decoder(rep)
            recon = (1 - pi) * mu 
            recon = torch.log(recon+1)
        else:
            recon = self.decoder(rep)
        
        if mode=='infer':
            print('Overall RMSE', math.sqrt(F.mse_loss(recon, gt)))
            return recon, rep
        elif mode=='test':
            loss = math.sqrt(F.mse_loss(recon[mask], use_g.ndata['gt'][mask]))
        elif mode=='valid':
            if self._objective in ['mask', 'maskvae']:
                loss = math.sqrt(F.mse_loss(recon[mask_nodes], use_g.ndata['feat'][mask_nodes]))
            elif self._objective in ['vae', 'ae']:
                loss = math.sqrt(F.mse_loss(recon, use_g.ndata['feat']))
        return loss
    
    def batch_test(self, graphs, mask, lbl, logger=None):
        with torch.no_grad():
            self.eval()
            device = next(self.parameters()).device
            res = []
            encres = []
            gound_truths = []
            for i, g in enumerate(graphs):
                g = g.to(device)
                x = g.ndata['input']
                gt = g.ndata['gt'].cpu()
                gound_truths.append(gt)
                m = mask[i]
                enc_rep = self.encoder(g, x)
                rep = self.encoder_to_decoder(enc_rep)
                encres.append(rep.cpu())
                
                if self._decoder_type == "zinb":
                    mu, _, pi = self.decoder(rep)
                    recon = (1 - pi) * mu
                    recon = torch.log(recon+1).cpu()
                else:
                    recon = self.decoder(rep).cpu()
                res.append(recon)
            
            nres = torch.cat(res).numpy()
            comprehensive_evaluate(nres, torch.cat(gound_truths).numpy(), np.concatenate(mask), lbl, logger)
        
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
