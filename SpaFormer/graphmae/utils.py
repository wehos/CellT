import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np
import anndata as ad
import scanpy as sc
import dgl
import torch
import torch.nn as nn
from torch import optim as optim
from tensorboardX import SummaryWriter
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Lung9_Rep1")
    parser.add_argument("--standardscale", type=int, default=250)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=3000,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--dropout", type=float, default=.2,
                        help="network dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default='layernorm', choices=[None, 'layernorm', 'batchnorm', 'groupnorm'])
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--mask_node_rate", type=float, default=0.5)
    parser.add_argument("--mask_feature_rate", type=float, default=0.5)
    parser.add_argument("--drop_node_rate", type=float, default=0.3)

    parser.add_argument("--encoder", type=str, default="mlp")
    parser.add_argument("--decoder", type=str, default="mlp", choices=["mlp", "zinb"])
    parser.add_argument("--aggr", type=str, default="SAGE", choices=["GAT", "SAGE"])
    parser.add_argument("--loss_fn", type=str, default="mse", choices=["mse", "wmse"])
    parser.add_argument("--optimizer", type=str, default="adamw")
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--cat_pe", action="store_true", default=False)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--data_path", type=str, default='../data/cosmx')
    parser.add_argument("--log_path", type=str, default="./result/run_20220126")
    parser.add_argument("--cache_path", type=str, default='./preprocessed')
    # ablation study
    parser.add_argument("--pe", type=str, default=None, choices=['lap', 'signnet', 'bin', 'mlp', 'sinu', 'None', None])
    parser.add_argument("--test_pe", action="store_true")
    parser.add_argument("--pe_aug", action="store_true")
    parser.add_argument("--objective", type=str, default='mask', choices=['mask', 'vae', 'maskvae', 'ae'])
    parser.add_argument("--noise_pc", type=int, default=30)
    args = parser.parse_args()
    return args


def prepare_dataset(dataset_name, pc, data_path='../../data/cosmx', cache_path='./preprocessed'):
    spatial = []
    loc = []
    adj = []
    I = []
    gt = []
    ps = []
    msk = []
    ratio = 0.2
    
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    if os.path.exists(f'{cache_path}/{dataset_name}_{pc}pc.pkl'):
        with open(f'{cache_path}/{dataset_name}_{pc}pc.pkl', 'rb') as f:
            spa, gex, masks, masked, fovlist, label = pickle.load(f)
    else:
        adata = ad.read_h5ad(f'{data_path}/cosmx_{dataset_name}.h5ad')
        adata = adata[adata.obs['fov'].isin(adata.obs['fov'].value_counts()[adata.obs['fov'].value_counts()>1000].index)]
        
        sc.pp.filter_genes(adata, min_counts=5)
        sc.pp.filter_cells(adata, min_counts=5)
        
        gex = adata.X
        spa = adata.obs[['x_FOV_px', 'y_FOV_px']].values
        masks = (np.random.rand(adata.shape[0], adata.shape[1]) > (pc/100)).astype(float)
        masked = np.multiply(gex.todense(), masks)
        all_zero = (np.sum(masked, 1)==0).astype(float)
        masks = np.multiply(masks, 1-all_zero) +  np.multiply(np.ones_like(masks), all_zero)
        masked = np.multiply(gex.todense(), masks)
        masked = csc_matrix(masked)
        masks = csc_matrix(1-masks)
        fovlist = adata.obs['fov']
        label = adata.obs['cellType'].astype('str')

        with open(f'{cache_path}/{dataset_name}_{pc}pc.pkl', 'wb') as f:
            pickle.dump([spa, gex, masks, masked, fovlist, label], f)
    
    gex = gex.todense().astype(float)
    masked = masked.todense().astype(float)
    masks = masks.todense().astype(bool)

    for fov in tqdm(fovlist.unique()):
        x = pairwise_distances(spa[fovlist==fov], metric='euclidean')
        #x = 1-(x-x.min())/(x.max()-x.min())
        np.fill_diagonal(x, 1e6)
        spatial.append(x)
        loc.append(spa[fovlist==fov])

        groundtruth_fov = gex[fov==fovlist]
        maskedgex_fov = masked[fov==fovlist]
        mask_fov = masks[fov==fovlist]

        gt.append(groundtruth_fov)
        ps.append(maskedgex_fov)
        msk.append(mask_fov)

        # GEX Correlation
        corr = np.corrcoef(maskedgex_fov).astype('float')#, rowvar=False)
        np.fill_diagonal(corr, 0)
        corr = np.nan_to_num(corr)
        adj.append(corr)

    with open(f'{cache_path}/{dataset_name}_{pc}pc_cache.pkl', 'wb') as f:
        pickle.dump([gt, ps, adj, spatial, msk, loc, label], f) 
    return gt, ps, adj, spatial, msk, loc, label


def graph_construction(ground_truth, masked, similarity_graph, spatial_graph, graph_type, standardscale, mask, pos, label, dataset_name, cache_path):
    
    total_sum = 0
    edge_count = 0
    graphs = []
    mask_new = []
    lbl = []
    
    logging.info("Loading preprocessed datasets.")
    for i in tqdm(range(len(ground_truth))):
        
        raw_X = torch.from_numpy(masked[i]).float()
        gt = torch.from_numpy(ground_truth[i]).float()
        nc = torch.sum(raw_X, 1)
        X_data = raw_X/(nc.unsqueeze(-1))*standardscale
        X_data = torch.log(X_data+1)
        spa = spatial_graph[i]
#         adj_0 = similarity_graph[i]
#         adj_0 = np.abs(similarity_graph[i])
        
        mask_new.append(torch.from_numpy(mask[i]).bool())
        
        # Create adjacency matrix
#         adj_0 = (adj_0 > 0.8) * (spa>0.8)
#         adj_0 = (((adj_0 > 0.85) + (spa>0.99))>0).astype('float')
#         adj_0 = (spa>0.98).astype('float')
#         adj_0 = (adj_0 > 0.84).astype('float')
#         adj_0 = ((adj_0+spa)>1.8).astype('float')
#         adj_0 = (tsne > 0.9) * (spa>0.85)
        
        adj_I = np.eye(X_data.shape[0])

        if graph_type == 'homo':
            # args.lambda_I = 0.6
            # adj = (1 - args.lambda_I) * adj_0 + args.lambda_I * adj_I
            if dataset_name.find('Liver') != -1:
                adj_0 = (spa<150).astype('float')
            else:
                adj_0 = (spa<95).astype('float')
            adj = adj_I + adj_0
            edge_count += adj_0.sum()
            total_sum += adj_0.shape[0]
            graph = dgl.from_scipy(sparse.csr_matrix(adj))
        elif graph_type == 'hete':
            pass
        else:
            print('Unsuported graph type.')
            exit(-1)
            
        coordinates = torch.from_numpy(pos[i]).float()
        scale = max(coordinates[:, 0].max() - coordinates[:, 0].min(), coordinates[:, 1].max() - coordinates[:, 1].min())
        coordinates[:, 0] = (coordinates[:, 0] - coordinates[:, 0].min()) / scale
        coordinates[:, 1] = (coordinates[:, 1] - coordinates[:, 1].min()) / scale
        
        graph.ndata["feat"] = X_data
        graph.ndata["input"] = X_data
#         graph.ndata['label'] = label
        graph.ndata['raw'] = raw_X
        graph.ndata['raw_gt'] = gt
        graph.ndata['gt'] = torch.log(gt/(torch.sum(gt, 1).unsqueeze(-1))*standardscale+1)
        graph.ndata['size_factor'] = nc/standardscale
        graph.ndata['pos'] = coordinates
        
        if os.path.exists(f'{cache_path}/lap/{dataset_name}_{i}.pkl'):
            with open(f'{cache_path}/lap/{dataset_name}_{i}.pkl', 'rb') as f:
                lpe = pickle.load(f)
        else:
            if not os.path.exists(f'{cache_path}/lap'):
                os.mkdir(f'{cache_path}/lap')
            logging.info(f'Laplacian positional encoding not found for fov {i}. Start calculatng.')
            lpe = dgl.laplacian_pe(graph, 10)
            with open(f'{cache_path}/lap/{dataset_name}_{i}.pkl', 'wb') as f:
                pickle.dump(lpe, f)
        lpe[torch.isnan(lpe)] = 0 
        graph.ndata['eigvec'] = lpe
        
        num_features = graph.ndata["feat"].shape[1]
        num_classes = num_features
        graphs.append(graph)
        
    print(edge_count/total_sum)
    return graphs, num_features, num_classes, mask_new, label

def load_dataset(dataset_name, pc, standardscale=250, data_path='../../data/cosmx', graph_type='homo', cache_path='./preprocessed'):
    if os.path.exists(f'{cache_path}/{dataset_name}_{pc}pc_cache.pkl'):
        with open(f'{cache_path}/{dataset_name}_{pc}pc_cache.pkl', 'rb') as f:
            gt, ps, adj, spatial, msk, loc, label = pickle.load(f)
    else:
        logging.info('Cache files not found. Preprocessing dataset.')
        gt, ps, adj, spatial, msk, loc, label = prepare_dataset(dataset_name, pc, data_path, cache_path)

    return graph_construction(gt, ps, adj, spatial, graph_type, standardscale, msk, loc, label, dataset_name, cache_path)

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
#         return None
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", setting='default', seed=None):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        self.seed = seed
        name = os.path.join(log_path, setting, f'seed_{seed}')
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()

class Logger(object):
    def __init__(self, args):
        super(Logger, self).__init__()

        log_path = args.log_path
        
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        self.setting = f'{args.dataset}_{args.noise_pc}pc_{args.encoder}'
        self.seed = args.seed
        self.args = args

    def note(self, metrics_dict):
        metrics_dict['seed'] = self.seed
        metrics_dict['num_hidden'] = self.args.num_hidden
        metrics_dict['decoder'] = self.args.decoder
        metrics_dict['objective'] = self.args.objective
        metrics_dict['num_layers'] = self.args.num_layers
        metrics_dict['num_heads'] = self.args.num_heads
        metrics_dict['norm'] = self.args.norm
        metrics_dict['aggr'] = self.args.aggr
        metrics_dict['mask_node_rate'] = self.args.mask_node_rate
        metrics_dict['mask_feature_rate'] = self.args.mask_feature_rate
        metrics_dict['drop_node_rate'] = self.args.drop_node_rate
        metrics_dict['latent_dim'] = self.args.latent_dim
        metrics_dict['pe'] = self.args.pe
        metrics_dict['cat_pe'] = self.args.cat_pe
        metrics_dict['lr'] = self.args.lr
        metrics_dict['weight_decay'] = self.args.weight_decay
        
        for k in metrics_dict:
            metrics_dict[k] = [metrics_dict[k]]
        if not os.path.exists(os.path.join(self.log_path, self.setting+'.csv')):
            df = pd.DataFrame(metrics_dict)
        else:
            df = pd.read_csv(os.path.join(self.log_path, self.setting+'.csv'))
            df = pd.concat([df, pd.DataFrame(metrics_dict)])
        df.to_csv(os.path.join(self.log_path, self.setting+'.csv'), index=False)


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
