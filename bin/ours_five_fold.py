import os
import sys
import logging
import argparse
import itertools
import numpy as np
import scipy.spatial
import collections
import matplotlib.pyplot as plt
from mpl_scatter_density import ScatterDensityArtist

import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.utils.data as Data
import sklearn.metrics as metrics
from astropy.visualization.mpl_normalize import ImageNormalize
import random
from astropy.visualization import LogStretch
from typing import *
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data,DataLoader
from sklearn.model_selection import train_test_split

import scanpy as sc
import pandas as pd

# To ensure that the output is fixed when the same input
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scMOG"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

MODELS_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)


import both_GAN_1_new
import anndata as ad
import activations
import utils
import lossfunction
import losses
from pytorchtools import EarlyStopping
from utils import plot_loss_history,plot_auroc, plot_prc,rmse_value,plot_scatter_with_r
import scipy.sparse as sp
from scipy.sparse import issparse
from torch_geometric.utils import to_dense_adj,subgraph

from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T
    adata.X = normalized_data
    return adata
    
def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    https://github.com/YosefLab/scVI/blob/6c9f43e3332e728831b174c1c1f0c9127b77cba0/scvi/models/log_likelihood.py#L206
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)  # Found above
        + torch.lgamma(x + theta)  # Found above
        - torch.lgamma(theta)  # Found above
        - torch.lgamma(x + 1)  # Found above
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return -res.mean()

class PairedGraphDataset(Dataset):
    def __init__(self, rna_graph, atac_graph):
        self.rna_features = rna_graph.x  # RNA 
        self.rna_edge_index = rna_graph.edge_index  # RNA

        self.atac_features = atac_graph.x  # ATAC 
        self.atac_edge_index = atac_graph.edge_index  # ATAC

    def __len__(self):
        return len(self.rna_features)

    def __getitem__(self, idx):
        if idx >= len(self.atac_features):
            raise IndexError(f"Index {idx} out of bounds for ATAC features with size {len(self.atac_features)}")
        
        # Construct RNA Data
        rna_data = Data(
            x=self.rna_features[idx].unsqueeze(0),  # RNA feature
            edge_index=self.rna_edge_index  # RNA edge_index
        )

        # Construct ATAC Data
        atac_data = Data(
            x=self.atac_features[idx].unsqueeze(0),  # ATAC feature
            edge_index=self.atac_edge_index  # ATAC edge_index
        )

        return rna_data, atac_data


logging.basicConfig(level=logging.INFO)

SAVEFIG_DPI = 1200
CUDA_VISIBLE_DEVICES=2


# sc_rna_train_dataset_save=ad.read_h5ad('D:\\project\\scACT-main\\scACT-main\\data\\mouse_embryo\\mouse_rna_common.h5ad')
# sc_atac_train_dataset_save=ad.read_h5ad('D:\\project\\scACT-main\\scACT-main\\data\\mouse_embryo\\mouse_atac_common.h5ad')


    
    
# sc_atac_train_dataset_copy_1 = sc_atac_train_dataset_save.copy()


#sc_rna_test_dataset=ad.read_h5ad('/mnt/5468e/twang/WBT/scMOG-main/scMOG-main/scMOG_code/mouse_embryo/valid_rna.h5ad')
#sc_atac_test_dataset=ad.read_h5ad('/mnt/5468e/twang/WBT/scMOG-main/scMOG-main/scMOG_code/mouse_embryo/valid_atac.h5ad')

def build_parser():
    """Building a parameter parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--outdir", "-o", required=True, type=str, help="Directory to output to"
    )
    parser.add_argument(
        "--hidden", type=int, nargs="*", default=[16], help="Hidden dimensions"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument(
        "--lr", "-l", type=float, default=[0.0001], nargs="*", help="Learning rate"
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, nargs="*", default=[256], help="Batch size"
    )
    parser.add_argument(
        "--seed", type=int, nargs="*", default=[2024], help="Random seed to use"
    )
    parser.add_argument("--device", default=3, type=int, help="Device to train on")
    parser.add_argument(
        "--ext",
        type=str,
        choices=["png", "pdf", "jpg"],
        default="pdf",
        help="Output format for plots",
    )
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    return parser

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy's sparse matrix to torch's sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adata_to_graph(adata, k=5):
    # 
    coordinates = torch.tensor(
        list(zip(adata.obs['x'], adata.obs['y'])), dtype=torch.float32
    )
    
    # 
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)

    # 
    edge_index = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # 
    X = adata.X.tocoo() if hasattr(adata.X, "tocoo") else adata.X
    features = torch.tensor(X.toarray(), dtype=torch.float32)

    # 
    return Data(x=features, edge_index=edge_index, coordinates = coordinates)

def build_adjacency_from_rna(features, k=5, metric='cosine'):
    
    if metric == 'cosine':
        similarity = cosine_similarity(features)
        distances = 1 - similarity  # 
    else:
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(features)
        distances, indices = nbrs.kneighbors(features)
    
    edge_index = []
    for i in range(features.shape[0]):
        # 
        neighbors = distances[i].argsort()[1:k+1]
        for neighbor in neighbors:
            edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_index

def build_adjacency_from_atac(features, k=5):
    similarity = features @ features.T  # 
    distances = 1 / (similarity + 1e-8)  # 
    
    edge_index = []
    for i in range(features.shape[0]):
        neighbors = distances[i].argsort()[:k]
        for neighbor in neighbors:
            if i != neighbor:  # 
                edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return edge_index


def regularization_loss(emb, graph_nei, graph_neg):
    #print(f"emb device: {emb.device}, graph_nei device: {graph_nei.device}, graph_neg device: {graph_neg.device}")
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    # mat = pd.DataFrame(mat.cpu().detach().numpy()).values

    # graph_neg = torch.ones(graph_nei.shape) - graph_nei

    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss

 

def cosine_similarity(emb):
    norm = torch.norm(emb, dim=1, keepdim=True)  # 
    emb_normalized = emb / (norm + 1e-8)  # 
    similarity = torch.mm(emb_normalized, emb_normalized.T)
    return similarity


def split_graph_data(graph_rna, graph_atac, test_size=0.1, val_size=0.2):
    # 
    num_nodes = graph_rna.x.shape[0]
    
    # 
    train_idx, temp_idx = train_test_split(range(num_nodes), test_size=test_size + val_size, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (test_size + val_size), random_state=42)
    
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    # 
    train_rna = graph_rna.subgraph(train_idx)
    val_rna = graph_rna.subgraph(val_idx)
    test_rna = graph_rna.subgraph(test_idx)
    
    train_atac = graph_atac.subgraph(train_idx)
    val_atac = graph_atac.subgraph(val_idx)
    test_atac = graph_atac.subgraph(test_idx)

    return train_rna,train_atac, test_rna, test_atac, val_rna, val_atac
    
def split_graph_data_two(graph_rna, graph_atac, test_size=0.1, val_size=0.3, seed=42):
    np.random.seed(seed)
    num_nodes = graph_rna.x.shape[0]  
    
    # 
    train_idx, temp_idx = train_test_split(range(num_nodes), test_size=test_size + val_size, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (test_size + val_size), random_state=42)
    
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    # 
    train_x_rna = graph_rna.x.clone()
    train_x_rna[val_idx] = 0  # 
    train_x_rna[test_idx] = 0  #

    train_x_atac = graph_atac.x.clone()
    train_x_atac[val_idx] = 0  # 
    train_x_atac[test_idx] = 0  #

    # 
    train_rna = Data(x=train_x_rna, edge_index=graph_rna.edge_index, coordinates=graph_rna.coordinates)
    train_atac = Data(x=train_x_atac, edge_index=graph_atac.edge_index, coordinates=graph_atac.coordinates)
    
    val_rna = graph_rna.subgraph(val_idx)
    test_rna = graph_rna.subgraph(test_idx)
    
    val_atac = graph_atac.subgraph(val_idx)
    test_atac = graph_atac.subgraph(test_idx)

    return train_rna, train_atac, val_rna, val_atac, test_rna, test_atac,train_idx, test_idx, val_idx

def filter_edges(edge_index, node_mask):
    src, dst = edge_index[0], edge_index[1]
    edge_mask = node_mask[src] & node_mask[dst]
    return edge_index[:, edge_mask]

def split_graph_data_old(graph_rna, graph_atac, test_size=0.2, val_size=0.1):
    def create_subgraph(graph, node_idx):
        node_mask = torch.zeros(graph.x.size(0), dtype=torch.bool)
        node_mask[node_idx] = True

        edge_index_filtered = filter_edges(graph.edge_index, node_mask)
        edge_index_spatial_filtered = filter_edges(graph.edge_index_spatial, node_mask)
        edge_index_feature_filtered = filter_edges(graph.edge_index_feature, node_mask)

        return Data(
            x=graph.x[node_mask],
            edge_index=edge_index_filtered,
            edge_index_spatial=edge_index_spatial_filtered,
            edge_index_feature=edge_index_feature_filtered,
            coordinates=graph.coordinates[node_mask]
        )

    num_nodes = graph_rna.x.shape[0]
    train_idx, temp_idx = train_test_split(range(num_nodes), test_size=test_size + val_size, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (test_size + val_size), random_state=42)

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    train_rna = create_subgraph(graph_rna, train_idx)
    val_rna = create_subgraph(graph_rna, val_idx)
    test_rna = create_subgraph(graph_rna, test_idx)

    train_atac = create_subgraph(graph_atac, train_idx)
    val_atac = create_subgraph(graph_atac, val_idx)
    test_atac = create_subgraph(graph_atac, test_idx)

    return train_rna, train_atac, test_rna, test_atac, val_rna, val_atac
    
import torch
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from sklearn.model_selection import KFold

def gaussian_kernel(distance_matrix, sigma=1.0):
    return np.exp(-distance_matrix**2 / (2 * sigma**2))

def preprocess_combined_graph_Guss(adata, feature_type, sigma=1.0, threshold=0.1):
    coordinates = torch.tensor(list(zip(adata.obs['x'], adata.obs['y'])), dtype=torch.float32)

    distance_matrix = cdist(coordinates, coordinates, metric='euclidean')

    adjacency_matrix = gaussian_kernel(distance_matrix, sigma=sigma)

    edge_index_spatial = np.argwhere(adjacency_matrix > threshold).T
    edge_index_spatial = torch.tensor(edge_index_spatial, dtype=torch.long)

    X = adata.X.tocoo() if hasattr(adata.X, "tocoo") else adata.X
    if issparse(X):
        features = torch.tensor(X.toarray(), dtype=torch.float32)
    else:
        features = torch.tensor(X, dtype=torch.float32)

    if feature_type == 'rna':
        edge_index_feature = build_adjacency_from_rna(features, k=4, metric='cosine')
    elif feature_type == 'atac':
        edge_index_feature = build_adjacency_from_atac(features, k=4)
    else:
        raise ValueError("Invalid feature_type. Use 'rna' or 'atac'.")

    return Data(x=features, edge_index=edge_index_spatial, coordinates=coordinates)

def create_spot_folds_bool_matrix(sc_atac_train_dataset, n_splits=5):
    
    all_spots = np.array(sc_atac_train_dataset.obs_names)  
    all_genes = np.array(sc_atac_train_dataset.var_names)  
    
    
    kf = KFold(n_splits=n_splits, shuffle=True)

    #
    spot_folds_bool = {}

    #
    for spot in all_spots:
        spot_index = np.where(all_spots == spot)[0][0]  
        
        # 
        spot_folds_bool[spot] = np.zeros((n_splits, len(all_genes)), dtype=bool)
        
        # 
        for i, (train_index, test_index) in enumerate(kf.split(all_genes)):
            # 
            spot_folds_bool[spot][i, train_index] = True  # 
            spot_folds_bool[spot][i, test_index] = False   # 
    
    return spot_folds_bool



def preprocess_combined_graph(adata, feature_type, k=14, metric='cosine'):
    coordinates = torch.tensor(list(zip(adata.obs['x'], adata.obs['y'])), dtype=torch.float32)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)

    edge_index_spatial = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                edge_index_spatial.append([i, neighbor])
    edge_index_spatial = torch.tensor(edge_index_spatial, dtype=torch.long).t()

    # 
    print("1232445555")
    print(adata)
    X = adata.X.tocoo() if hasattr(adata.X, "tocoo") else adata.X
    if issparse(X):
        features = torch.tensor(X.toarray(), dtype=torch.float32)
    else:
        features = torch.tensor(X, dtype=torch.float32)

    # 
    if feature_type == 'rna':
        edge_index_feature = build_adjacency_from_rna(features, k=k, metric=metric)
    elif feature_type == 'atac':
        edge_index_feature = build_adjacency_from_atac(features, k=k)
    else:
        raise ValueError("Invalid feature_type. Use 'rna' or 'atac'.")

    # 
    return Data(x=features, edge_index=edge_index_spatial, coordinates=coordinates)

def preprocess_self_loop_graph(adata):
    num_nodes = adata.shape[0]
    
    # 
    edge_index_self_loop = torch.arange(num_nodes).repeat(2, 1)  # [ [0,1,2,...], [0,1,2,...] ]

    # 
    X = adata.X.tocoo() if hasattr(adata.X, "tocoo") else adata.X
    features = torch.tensor(X.toarray(), dtype=torch.float32)

    # 
    coordinates = torch.tensor(list(zip(adata.obs['x'], adata.obs['y'])), dtype=torch.float32)

    return Data(x=features, edge_index=edge_index_self_loop, coordinates=coordinates)



def predict_atac(truth,generator,truth_iter, outdir_name, rna2atac, atac2rna, sc_atac_train_dataset_save,mask, ininverted_mask):
    logging.info("....................................Evaluating ATAC ")
    def predict1(generator,truth_iter):
        generator.eval()
        first = 1
        for i in range(1):
            
            # RNA graph
            rna_features = truth_iter.x # RNA node_feature
            rna_edge_index = truth_iter.edge_index  # RNA_edges
            rna_coord = truth_iter.coordinates  # RNA_edges
                    
            with torch.no_grad():
                z, y_pred, mu, logstad, adj_pred = generator(rna_features, rna_edge_index, rna_coord)  
                #mu, logstad, z = generator.encoder(rna_features, rna_edge_index, rna_coord) 
                #rna2atac_latent = rna2atac(z)
                #rna_latent_recon = atac2rna(rna2atac_latent)
                #y_pred, adj_pred = generator.decoder(rna_latent_recon,rna_coord)inverted_mask
                ret = y_pred
                
        return ret
    
    sc_rna_atac_truth_preds = predict1(generator,truth_iter)
    # print(sc_rna_atac_truth_preds.shape)
    # print(sc_atac_train_dataset_save)
    
    # print(truth.shape)
    
    # ddd
    #print(sc_rna_atac_truth_preds.shape)
            # 
    if isinstance(sc_rna_atac_truth_preds, torch.Tensor):
        tt = sc_rna_atac_truth_preds.cpu().numpy()  # 
            
            
    if isinstance(sc_rna_atac_truth_preds, np.ndarray):
        tt = sc_rna_atac_truth_preds.astype(np.float32)
    
    # if isinstance(ininverted_mask, torch.Tensor):
    #     ininverted_mask = ininverted_mask.cpu().numpy()  #
    
    if isinstance(ininverted_mask, torch.Tensor):
        ininverted_mask = ininverted_mask.bool()  #
    
    # 3
    # gene_names = sc_atac_train_dataset_save.var_names
    # cell_names = sc_atac_train_dataset_save.obs_names

    # # 4.
    # expression_df = pd.DataFrame(tt, index=cell_names, columns=gene_names)
    #expression_df_2 = pd.DataFrame(truth, index=cell_names, columns=gene_names)

    # 
    # expression_df.to_csv('D:\\project\\scACT-main\\scACT-main\\data\\mouse_embryo\\mouse_embryo_expression_matrix.csv')
    #expression_df_2.to_csv('D:\\project\\scACT-main\\scACT-main\\data\\mouse_embryo\\mouse_embryo_expression_matrix_orign.csv')

    # print(sum(ininverted_mask))
    # print(tt * ininverted_mask)
    # print(tt[ininverted_mask].shape)
    print(sum(mask))
    sc_atac_train_dataset_save.X = tt[:, ininverted_mask]
    #print(sc_atac_train_dataset_save.X)
    #DDDD
    print(type(truth))
    print(type(sc_rna_atac_truth_preds))
    sc_atac_train_dataset_save.write_h5ad("/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/data_p22/predict_fold5.h5ad")
    # print("111111111")
    # print(sc_rna_atac_truth_preds[:, ininverted_mask].shape)
    # print(sc_rna_atac_truth_preds[:, ininverted_mask].shape)
    #fig = plot_auroc(
    #    truth,
    #    sc_rna_atac_truth_preds,
    #    title_prefix="RNA > ATAC",
    #    fname=os.path.join(outdir_name, f"rna_atac_auroc.pdf"),
    #)  
    
    fig = plot_scatter_with_r(
        truth,
        sc_rna_atac_truth_preds,
        one_to_one=True,
        logscale=False,
        density_heatmap=True,
        title="RNA > ATAC",
        fname=os.path.join(outdir_name, f"rna_atac_auroc.pdf"),
    )
    rmse_value(truth,sc_rna_atac_truth_preds)
    plt.close(fig)

def predict_rna(truth,generator,truth_iter, outdir_name, rna2atac, atac2rna, sc_rna_train_dataset_save):
    logging.info(".........................................Evaluating  RNA")

    def predict2(generator,truth_iter, rna2atac, atac2rna):
        generator.eval()
        first = 1
        truth = []
        for i in range(1):
        
            # RNA graph
            atac_features = truth_iter.x  # RNA node_feature
            atac_edge_index = truth_iter.edge_index  # RNA_edges
            atac_coord = truth_iter.coordinates
           
            with torch.no_grad():
                z_atac, retval1, retval2, retval3, mu, logstd, adj_pred = generator(atac_features, atac_edge_index, atac_coord)
                #mu, logstd, z = generator.encoder(atac_features, atac_edge_index, atac_coord)
                #atac2rna_latent = atac2rna(z)
                #atac_latent_recon = rna2atac(atac2rna_latent)
                #retval1, retval2, retval3, adj_pred =  generator.decoder(atac_latent_recon,atac_coord)
                ret = retval1
                    
        return ret
    
    sc_rna_truth_preds = predict2(generator,truth_iter, rna2atac, atac2rna)
    #print(sc_rna_test_dataset)
    #print(sc_rna_truth_preds.shape)
            # 
    if isinstance(sc_rna_truth_preds, torch.Tensor):
        tt = sc_rna_truth_preds.cpu().numpy()  # 
            
            
    if isinstance(sc_rna_truth_preds, np.ndarray):
        tt = sc_rna_truth_preds.astype(np.float32)
    sc_rna_train_dataset_save.X = tt
    sc_rna_train_dataset_save.write_h5ad("/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/mouse_mebryo/predict_rna_fold_1.h5ad")
    print("8888888")
    fig = plot_scatter_with_r( 
        truth,
        sc_rna_truth_preds,
        one_to_one=True,
        logscale=False,
        density_heatmap=True,
        title="ATAC > RNA (test set)",
        fname=os.path.join(outdir_name, f"atac_rna_scatter_log.pdf"),
    )
    rmse_value(truth,sc_rna_truth_preds)
    plt.close(fig)

def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata
    
def clean_data(adata):
    if hasattr(adata.X, 'toarray'):  
        adata.X = adata.X.toarray()
    print("Before cleaning:")
    print(f"NaN count: {np.isnan(adata.X).sum()}")
    print(f"Infinity count: {(np.isinf(adata.X)).sum()}")
    
    # 
    adata.X = np.nan_to_num(adata.X, nan=0, posinf=0, neginf=0)

    # 
    print("After cleaning:")
    print(f"NaN count: {np.isnan(adata.X).sum()}")
    print(f"Infinity count: {(np.isinf(adata.X)).sum()}")
    
    return adata







def main():
    """Run Script"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}_training.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")

    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")
        
    loss_bce = losses.MSELoss()
    loss_rna = lossfunction.loss
   
        
    def loss_D(fake, real, discriminator, edge_index, spatial_coords):
        loss2_1 = -torch.mean(discriminator(real, edge_index, spatial_coords))  # 
        if isinstance(fake, tuple):
            loss2_2 = torch.mean(discriminator(fake[0].detach(), edge_index, spatial_coords))  # 
        else: 
            loss2_2 = torch.mean(discriminator(fake.detach(), edge_index, spatial_coords))  #
        loss2 = loss2_1 + loss2_2
        return loss2

    def loss_rna_G(fake, discriminator, edge_index, spatial_coords):
        loss1 = -torch.mean(discriminator(fake, edge_index, spatial_coords))  
        return loss1
    
    def loss_atac_G(fake, discriminator, edge_index, spatial_coords):
        loss1 = -torch.mean(discriminator(fake, edge_index, spatial_coords))  #
        return loss1 

    sc_rna_train_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/data_p22/mouse_rna_common.h5ad')
    sc_atac_train_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/data_p22/mouse_atac_common.h5ad')
    print("INFO----RNA and ATAC")
    print(sc_rna_train_dataset)
    print(sc_atac_train_dataset)
    #sc_atac_train_dataset.X[sc_atac_train_dataset.X>0] = 1

    #sc.pp.normalize_total(sc_rna_train_dataset)
    #sc.pp.log1p(sc_rna_train_dataset)
    
    #sc.pp.normalize_total(sc_atac_train_dataset)
    #sc.pp.log1p(sc_atac_train_dataset)
    sc_atac_train_dataset = scale(sc_atac_train_dataset)
    
    
    
    # #
    all_genes = np.array(sc_atac_train_dataset.var_names)

    # # 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

   
    cross_val_splits = []

    for train_index, test_index in kf.split(all_genes):
    #   
         train_genes = all_genes[train_index]
         test_genes = all_genes[test_index]
        
    #    
         cross_val_splits.append((train_genes, test_genes))

    # 
    for fold, (train_genes, test_genes) in enumerate(cross_val_splits, start=1):
         print(f"Fold {fold}:")
         print(f"Training Genes: {len(train_genes)}")
         print(f"Test Genes: {len(test_genes)}")
         print(f"Example Test Genes: {test_genes[:5]}")  # 
         print()
    
    all_genes_find = np.array(sc_atac_train_dataset.var_names)
    genes_to_zero_check = cross_val_splits[0][1] # 0 1 2 3
    print(len(genes_to_zero_check))
    indices_not_zero = (~np.isin(all_genes_find, genes_to_zero_check)).astype(bool)
    print("ddddd")
    print(sum(indices_not_zero))
    genes_to_zero = cross_val_splits[0][1]
    genes_in_test_set = np.isin(all_genes_find, genes_to_zero_check).astype(bool)
    print(sum(genes_in_test_set))
    subset_expression = sc_atac_train_dataset[:, genes_in_test_set]
    sc_atac_train_dataset_save = sc.AnnData(X=subset_expression.X, obs=subset_expression.obs, var=subset_expression.var)
    #sc_atac_train_dataset_save = sc_atac_train_dataset
    print(type(sc_atac_train_dataset))
    print(type(sc_atac_train_dataset_save))
   
    print(sc_atac_train_dataset_save)
 
    sc_rna_train_dataset_save = sc_rna_train_dataset
    
    
    #sc_atac_train_dataset.X[:,genes_in_test_set] = 0
    # print(sc_atac_train_dataset.X[:,genes_to_zero])
    # print(np.max(sum(sc_atac_train_dataset.X[:,genes_to_zero])))
    
    #sc_atac_train_dataset = scale(sc_atac_train_dataset)
    #sc_rna_train_dataset = scale(sc_rna_train_dataset)
    
    #sc.pp.normalize_total(sc_atac_train_dataset, target_sum=1e4)
    #sc.pp.log1p(sc_atac_train_dataset)
    # Preprocess data
    #graph_rna = preprocess_adata_to_graph(sc_rna_train_dataset)
    #graph_atac = preprocess_adata_to_graph(sc_atac_train_dataset)
    
    #
    #sc_rna_train_dataset, test_indices_rna, val_indices_rna = mask_spots(sc_rna_train_dataset, test_fraction=0.2, val_fraction=0.1, seed=1234)
    #sc_atac_train_dataset, test_indices_atac, val_indices_atac = mask_spots(sc_atac_train_dataset, test_fraction=0.2, val_fraction=0.1, seed=1234)
    
    graph_rna = preprocess_combined_graph(sc_rna_train_dataset, feature_type='rna', k=14, metric='cosine' )
    graph_atac = preprocess_combined_graph(sc_atac_train_dataset, feature_type='atac', k=14, metric='cosine')
    graph_atac_valid = preprocess_combined_graph(sc_atac_train_dataset, feature_type='rna', k=14, metric='cosine' )
    #graph_rna = preprocess_self_loop_graph(sc_rna_train_dataset)
    #graph_atac = preprocess_self_loop_graph(sc_atac_train_dataset)

    # Split data
    #train_rna, train_atac, valid_rna, valid_atac,test_rna, test_atac, train_idx, test_idx, valid_idx = split_graph_data_two(graph_rna, graph_atac)
    print("Evalulate numbers:")
    #print(torch.sum(train_atac.x == 0))
    #print(torch.sum(test_atac.x == 0))
    
    train_rna =  graph_rna
    train_atac = graph_atac
    test_rna = graph_rna
    test_atac = graph_atac
    valid_rna = graph_rna
    valid_atac = graph_atac
   
    
    
    train_dataset = PairedGraphDataset(train_rna, train_atac)
    test_dataset = PairedGraphDataset(test_rna, test_atac)
    valid_dataset = PairedGraphDataset(valid_rna, valid_atac)
    
    batch_size = 32
    
    # 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    truth_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model and optimizer
    in_channels_rna = graph_rna.x.size(1)  # RNA
    in_channels_atac = graph_atac.x.size(1)  # ATAC 
    hidden_channels = 128 #  32
    latent_channels = 64 #  16
    
    hidden_channels_2 = 128  # 128  32
    latent_channels_2 = 64 # 64  15
  
    num_outputs_rna = graph_atac.x.size(1)  # 
    num_outputs_atac = graph_rna.x.size(1)  # ATAC 
    
    
    generatorATAC = both_GAN_1_new.VGAEModel_rna(in_channels_rna, hidden_channels_2, latent_channels_2, num_outputs_rna)  # ATACDecoder
    #input_dim2 = chrom_counts.values.tolist()
    generatorRNA = both_GAN_1_new.VGAEModel_atac(in_channels_atac, hidden_channels, latent_channels, num_outputs_atac)  # RNADecoder
    
    
    
    
    #rna2atac = both_GAN_1.AffineTransform(16, 256, affine_num=20)
    #atac2rna = both_GAN_1.AffineTransform(16, 256, affine_num=20)
    
    rna2atac = both_GAN_1_new.AffineTransform(16, 256, affine_num=9)
    atac2rna = both_GAN_1_new.AffineTransform(64, 256, affine_num=9)
    
    atac_input_dim = (sc_atac_train_dataset.X.shape[1] - sum(genes_in_test_set))
    print(sc_atac_train_dataset.X.shape[1])
    print(atac_input_dim)

    
    RNAdiscriminator = both_GAN_1_new.Discriminator1(input_dim=sc_rna_train_dataset.X.shape[1] )
    ATACdiscriminator = both_GAN_1_new.Discriminator1(input_dim=atac_input_dim)
    
    cuda = True if torch.cuda.is_available() else False
    device_ids = range(torch.cuda.device_count())  
    
    #chrom_counts = sc_atac_train_dataset.var['chrom'].value_counts()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(np.max(sc_rna_train_dataset.X), np.min(sc_rna_train_dataset.X))
    print(np.max(sc_atac_train_dataset.X), np.min(sc_atac_train_dataset.X))
    
    dd = 0.0001
    dd2 = 0.0001
    weight_decay = 1e-5
    #optimizer_rna_1 = torch.optim.Adam(generatorRNA.parameters(), lr=dd, betas=(args.b1, args.b2))
    #optimizer_atac_1 = torch.optim.Adam(generatorATAC.parameters(), lr=dd, betas=(args.b1, args.b2))
    optimizer_atac_1 = torch.optim.RMSprop(generatorATAC.parameters(), lr=dd2, weight_decay=weight_decay)
    optimizer_rna_1 = torch.optim.RMSprop(generatorRNA.parameters(), lr=dd , weight_decay=weight_decay)
    optimizer_A = torch.optim.Adam(generatorRNA.parameters(), lr=dd , weight_decay=weight_decay)
    optimizer_B = torch.optim.Adam(generatorATAC.parameters(), lr=dd2, weight_decay=weight_decay)
    #optimizer_A_2 = torch.optim.Adam(generatorRNA.parameters(), lr=dd, betas=(args.b1, args.b2), weight_decay=weight_decay)
    #optimizer_B_2 = torch.optim.Adam(generatorATAC.parameters(), lr=dd, betas=(args.b1, args.b2), weight_decay=weight_decay)
    #optimizer_rna = torch.optim.Adam(generatorRNA.parameters(), lr=dd, betas=(args.b1, args.b2), weight_decay=weight_decay)
    #optimizer_atac = torch.optim.Adam(generatorATAC.parameters(), lr=dd, betas=(args.b1, args.b2), weight_decay=weight_decay)
    
    
    optimizer_A_2 = torch.optim.Adam(generatorRNA.parameters(), lr=dd,weight_decay=weight_decay)
    optimizer_B_2 = torch.optim.Adam(generatorATAC.parameters(), lr=dd2, weight_decay=weight_decay )
    optimizer_rna = torch.optim.Adam(generatorRNA.parameters(), lr=dd, weight_decay=weight_decay)
    optimizer_atac = torch.optim.Adam(generatorATAC.parameters(), lr=dd2, weight_decay=weight_decay)
   
    optimizer_D_rna = torch.optim.RMSprop(RNAdiscriminator.parameters(), lr=dd, weight_decay=weight_decay)
    optimizer_D_atac = torch.optim.RMSprop(ATACdiscriminator.parameters(), lr=dd, weight_decay=weight_decay)
# Model
    if isinstance(args.hidden, float) or isinstance(args.hidden, int):
        args.hidden = [int(args.hidden)]  # 
    param_combos = list(
        itertools.product(
            args.hidden,  args.lr,args.seed
        )
    )
    for h_dim, lr, rand_seed in param_combos:
        outdir_name = (
            f"{args.outdir}_hidden_{h_dim}_lr_{lr}_seed_{rand_seed}"
            if len(param_combos) > 1
            else args.outdir
        )
        if not os.path.isdir(outdir_name):
            assert not os.path.exists(outdir_name)
            os.makedirs(outdir_name)
        assert os.path.isdir(outdir_name)
        
        
        #generatorRNA.load_state_dict(torch.load(os.path.join(outdir_name, "RNAgenerator.pth")))
        #generatorATAC.load_state_dict(torch.load(os.path.join(outdir_name, "ATACgenerator.pth")))
        
        def train_vgae(model, optimizer, data, epochs, device, truth_atac, discriminator, updaterD, indices_not_zero):
            device = 'cuda:0'
            discriminator.train()
            model.train()
            discriminator.to(device)
            model = model.to(device)
            data = data.to(device)
            #indices_not_zero = indices_not_zero.to(device)
            loss_history = []
            trainD_losses=[]
            #print(data)
            y_hat = truth_atac.x.to(device)
            print(y_hat)
            
    
            graph_nei = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0]
            graph_neg = 1 - graph_nei # 
            graph_neg.fill_diagonal_(0)  # 
            
            for epoch in range(epochs):
                #torch.autograd.set_detect_anomaly(True)
                updaterD.zero_grad()
                z, y, mu, logstd, adj_pred = model(data.x.to(device), data.edge_index.to(device),data.coordinates.to(device))
                y_max = y.max().item() 

                print(f"Maximum value of y: {y_max}")
                #loss2 = loss_D(y, y_hat, discriminator)
                #ddddd
                loss2 = loss_D(y[:,indices_not_zero], y_hat[:, indices_not_zero], discriminator, data.edge_index, data.coordinates.to(device))
                #print(y[:, indices_not_zero].shape)
                loss2.backward()
                updaterD.step()
                trainD_losses.append(loss2.item())
                # for p in discriminator.parameters():
                #     p.data.clamp_(-args.clip_value, args.clip_value)
                print(f"Epoch {epoch + 1}/{epochs}, Loss_2: {loss2.item():.4f}")
                if epoch % 1 == 0:
                    optimizer.zero_grad()
                    z, y, mu, logstd, adj_pred = model(data.x.to(device), data.edge_index.to(device),data.coordinates.to(device))
                    #reg_loss = model.recon_loss(adj_pred, graph_nei.to(device))
                    #loss = loss_atac_G(y, discriminator)
                    loss = loss_atac_G(y[:, indices_not_zero], discriminator, data.edge_index, data.coordinates)
                    #reg_loss = regularization_loss(z, graph_nei, graph_neg)
                    kl_divergence = model.kl_loss(mu, logstd)
                    print("ddd")
                    print(loss * 10)
                    # print(reg_loss * 0.1)
                    print(f"ATAC KL Loss: {kl_divergence.item():.4f}")
                    
                    loss = loss * 10 #+ reg_loss * 0.1  + kl_divergence 
                    loss.backward()
                    

                    optimizer.step()
                    loss_history.append(loss.item())
                    
                    
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
            return loss_history
        
        def train_vgae_atac(model, optimizer, data, epochs, device, truth_rna, discriminator, updaterD):
            device = 'cuda:0'
            model.train()
            discriminator.train()
            model = model.to(device)
            
            #data = data.x[:, genes_not_zero].to(device)
            data = data.to(device)
            
            discriminator.to(device)
            loss_history = []
            trainD_losses=[]
            dd = 0
            size_factors_rna = torch.ones(data.x.size(0)).to(device)
            #print(data)
            y_hat = truth_rna.x[:, indices_not_zero].to(device)
            print(indices_not_zero)
            print(sum(indices_not_zero))
            print(y_hat.shape)
            
            from torch_geometric.utils import to_dense_adj
    
            graph_nei = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0]  # 
            graph_neg = 1 - graph_nei  # 
            graph_neg.fill_diagonal_(0)  # 
            
            
            for epoch in range(epochs):
                
                z, retval1, retval2, retval3, mu, logstd, adj_pred = model(
                data.x.to(device), data.edge_index.to(device),data.coordinates.to(device)
            )
                updaterD.zero_grad()
                dd = max(dd, retval1.detach().cpu().numpy().max())
                
                loss2 = loss_D(retval1[:, indices_not_zero], y_hat, discriminator, data.edge_index, data.coordinates)
                loss2.backward()
                updaterD.step()
                trainD_losses.append(loss2.item())
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)
                
                if epoch % 1 == 0:
                    optimizer.zero_grad()
                    z, retval1, retval2, retval3, mu, logstd, adj_pred = model(
                    data.x.to(device), data.edge_index.to(device),data.coordinates.to(device)
                    )
                    loss = loss_rna_G(retval1[:, indices_not_zero], discriminator, data.edge_index, data.coordinates)
                    #reg_loss = model.recon_loss(adj_pred, graph_nei.to(device))
                    reg_loss = regularization_loss(z, graph_nei, graph_neg)
                    
                    kl_divergence = model.kl_loss(mu, logstd)
                    print("ddd")
                    print(loss )
                    print(kl_divergence)
                    print(reg_loss)
                    #print(kl_divergence )
                    loss = loss + kl_divergence #+ reg_loss #
                    #loss = loss_bce(y_hat, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()
                    
                    loss_history.append(loss.item())
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
                    print(dd)
        
            return loss_history
          
            
        def train_cycle_consistency(generator_A, generator_B, num_epochs, train_iter, test_iter,truth_iter,
            updaterG_A, updaterG_B, lambda_cycle, discriminator, cuda=True, ISRNA=True,sc_atac_train_dataset_save= sc_atac_train_dataset_save,
            sc_rna_train_dataset_save= sc_rna_train_dataset_save, mask=indices_not_zero, inverted_mask = genes_in_test_set):
            # 
            device_rna = 'cuda:0'
            device_atac = 'cuda:0'
            loss1_history, loss2_history, loss1_test_history = [], [], []
            generator_A.to(device_atac)
            generator_B.to(device_rna)
            
            valid_atac.to(device_rna)
            valid_rna.to(device_rna)
            
            #sc_atac_train_dataset_save.to(device_rna)
            
            from torch_geometric.utils import to_dense_adj
            
            graph_nei = to_dense_adj(train_rna.edge_index, max_num_nodes=train_rna.x.size(0))[0]  # 
            graph_neg = 1 - graph_nei  # 
            graph_neg.fill_diagonal_(0)  # 
            graph_nei = graph_nei.float()
            graph_nei = graph_nei.to(device_rna)
            graph_neg = graph_neg.to(device_rna)      
            
            graph_nei_test = to_dense_adj(test_rna.edge_index, max_num_nodes=test_rna.x.size(0))[0]  # 
            graph_neg_test = 1 - graph_nei_test  # 
            graph_neg_test.fill_diagonal_(0)  # 
            graph_nei_test = graph_nei_test.float()
            graph_nei_test = graph_nei_test.to(device_rna)
            graph_neg_test = graph_neg_test.to(device_rna)        
            
            kk = 0
            # EarlyStopping
            early_stopping = EarlyStopping(patience=7, verbose=True)
            # 
            for epoch in range(num_epochs):
                generator_A.train()
                generator_B.train()
                train_losses_1 = []
                train_losses_2 = []
               
                for i in range(1):
                    
                    # RNA 
                    rna_features = train_rna.x.to(device_rna)
                    rna_edge_index = train_rna.edge_index.to(device_rna) #RNA 
                    rna_edge_coord = train_rna.coordinates.to(device_rna) #RNA 
                    
                    
                    # ATAC
                    atac_features = train_atac.x.to(device_atac)  # ATAC
                    atac_edge_index = train_atac.edge_index.to(device_atac)  # ATAC
                    atac_edge_coord = train_atac.coordinates.to(device_atac) #RNA 
                   
                    updaterG_A.zero_grad()
                    if(ISRNA): 
                        z_rna, y, mu_rna, logstad_rna, adj_pred_rna = generator_A(rna_features, rna_edge_index,rna_edge_coord)
                        z_atac, retval1, retval2, retval3, mu_atac, logstad_atac, adj_pred_atac = generator_B(atac_features, atac_edge_index,atac_edge_coord)
                    else :
                        z_atac, retval1, retval2, retval3, mu_rna, logstad_rna, adj_pred_atac = generator_A(atac_features, atac_edge_index,atac_edge_coord)
                        z_rna, y, mu_atac, logstad_atac, adj_pred_rna = generator_B(rna_features, rna_edge_index,rna_edge_coord)
                    
                    
                    
                    if(ISRNA) :
                        #rna_recon_cycle_loss = loss_rna(preds=retval1, theta=retval2, truth=rna_features,)
                        rna_recon_cycle_loss_1 = loss_rna(preds=retval1, theta=retval2, truth=rna_features,)
                        rna_recon_cycle_loss, recon, kl = generator_B.compute_elbo(mu_rna, logstad_rna, rna_recon_cycle_loss_1,)
                        reg_loss_rna = generator_A.recon_loss(graph_nei, adj_pred_rna.float().detach())
                        kl_rna = generator_B.kl_loss(mu_rna, logstad_rna)
                        #rna_recon_cycle_loss = log_zinb_positive(rna_features, retval1, retval3, retval3).mean()
                        atac_recon_cycle_loss = loss_bce(y * mask, atac_features)
                        reg_loss_atac = generator_B.recon_loss(graph_nei, adj_pred_atac.float().detach())
                        kl_atac = generator_B.kl_loss(mu_atac, logstad_atac)
                        
                        #retval1_train, retval2_train, retval3_train, adj_pred_train = generator_B.decoder(rna_latent_recon, rna_edge_coord)
                        #rna_recon_cycle_loss_train = loss_rna(preds=retval1_train, theta=retval2_train, truth=rna_features,)
                        
                    else :
                        #rna_recon_cycle_loss = loss_rna(preds=retval1, theta=retval2, truth=rna_features)
                        rna_recon_cycle_loss_1 = loss_rna(preds=retval1, theta=retval2, truth=rna_features,)
                        rna_recon_cycle_loss, recon, kl = generator_B.compute_elbo(mu_rna, logstad_rna, rna_recon_cycle_loss_1,)
                        reg_loss_rna = generator_A.recon_loss(graph_nei, adj_pred_rna.float().detach())
                        kl_rna = generator_B.kl_loss(mu_rna, logstad_rna)
                        #rna_recon_cycle_loss = log_zinb_positive(rna_features, retval1, retval3, retval3).mean()
                        #atac_recon_cycle_loss = loss_bce(y, atac_features)
                        atac_recon_cycle_loss_1 = loss_bce(y[:, indices_not_zero], atac_features[:,indices_not_zero])
                        atac_recon_cycle_loss, recon, kl = generator_B.compute_elbo(mu_atac, logstad_atac, atac_recon_cycle_loss_1)
                        
                        
                        reg_loss_atac = generator_B.recon_loss(graph_nei, adj_pred_atac.float().detach())
                        kl_atac = generator_B.kl_loss(mu_atac, logstad_atac)
                        
                        #y_pred_train, adj_pred_train= generator_B.decoder(atac_latent_recon, atac_edge_coord)
                        #atac_recon_cycle_loss_train = loss_bce(y_pred_train, atac_features)
                
                    print("Generator_B")
                    if(ISRNA):
                        print(f"RNA Reconstruction Loss: {rna_recon_cycle_loss.item():.4f}")
                    # else:
                    #     print(f"ATAC Reconstruction Loss: {atac_recon_cycle_loss_1.item():.4f}")
                        #print(f"rna_recon_cycle_loss_train: {rna_recon_cycle_loss_train.item():.4f}")
                    print(f"RNA Reconstruction Loss: {rna_recon_cycle_loss.item():.4f}")
                    print(f"ATAC Reconstruction Loss: {atac_recon_cycle_loss.item():.4f}")
                    print(f"rna reg Loss: {reg_loss_rna.item():.4f}")
                    print(f"ATAC reg Loss: {reg_loss_atac.item():.4f}")
                    print(f"RNA KL Loss: {kl_rna.item():.4f}")
                    print(f"ATAC KL Loss: {kl_atac.item():.4f}")
    
                    #print(f"Cycle_loss_rna: {cycle_loss_rna.item():.4f}")
                    #print(f"Cycle_loss_atac: {cycle_loss_atac.item():.4f}")
                    
                    
                    # 
                    if ISRNA:
                        #loss_A = rna_recon_cycle_loss + reg_loss_rna * 0.01 + lambda_cycle * (cycle_loss_rna )#+ cycle_loss_atac) # + kl_rna
                        loss_A = rna_recon_cycle_loss  #+ 0.01 * (cycle_loss_rna) + rna_recon_cycle_loss_train # 10 + reg_loss_rna * 0.01 + lambda_cycle * (cycle_loss_rna )#+ cycle_loss_atac) # + kl_rna
                    else :
                        #loss_A = atac_recon_cycle_loss * 0.01 + reg_loss_atac  + lambda_cycle * ( cycle_loss_atac) #+ kl_atac
                        loss_A =  atac_recon_cycle_loss #* 10 + reg_loss_atac * 0.01 #1 * ( cycle_loss_atac) + atac_recon_cycle_loss_train *  0.01# + reg_loss_atac * 0.01 + lambda_cycle * ( cycle_loss_atac) #+ kl_atac
                    print(f"loss_A Loss: {loss_A.item():.4f}")
                    
                    loss_A.backward()
                    updaterG_A.step()
                    train_losses_1.append(loss_A.item())
                   
                if ((epoch + 1) % 10== 0):
                    if ISRNA:
                        truth = valid_rna.x#.to(device)
                        predict_rna(truth, generator_B, train_atac, outdir_name,  rna2atac, atac2rna, sc_rna_train_dataset_save)
                    else:
                        truth = valid_atac.x.to(device_rna)
                        print(truth.shape)
                        predict_atac(truth, generator_B, valid_rna, outdir_name, rna2atac, atac2rna, sc_atac_train_dataset_save,indices_not_zero, inverted_mask)
                # 
                loss1_history.append(np.mean(train_losses_1))
                logging.info(f"Epoch [{epoch + 1}/{num_epochs}], loss_A: {loss1_history[-1]:.4f}")

                # 
                if test_iter:
                    generator_A.eval()
                    generator_B.eval()
                    valid_losses_1 = []

                    with torch.no_grad():
                        rna_features_test = test_rna.x.to(device_rna)
                        rna_edge_index_test = test_rna.edge_index.to(device_rna) #RNA 
                        rna_coord_test = test_rna.coordinates.to(device_rna)  #RNA 
                    
                        # ATAC
                        atac_features_test = test_atac.x.to(device_atac)  # ATAC
                        atac_edge_index_test = test_atac.edge_index.to(device_atac)  # ATAC
                        atac_coord_test = test_atac.coordinates.to(device_atac)  #RNA 
                            
                        if(ISRNA) :
                            z_rna_test, y_test, mu_rna_test, logstad_rna_test, adj_pred_rna = generator_A(rna_features_test, rna_edge_index_test, rna_coord_test)
                            z_atac_test, retval1_test, retval2_test, retval3_test, mu_atac_test, logstd_atac_test, adj_pred_atac = generator_B(atac_features_test, atac_edge_index_test, atac_coord_test)
                        else: 
                            print("eeee")
                            z_atac_test, retval1_test, retval2_test, retval3_test, mu_rna_test, logstad_rna_test, adj_pred_atac = generator_A(atac_features_test, atac_edge_index_test, atac_coord_test)
                            z_rna_test, y_test, mu_atac_test, logstad_atac_test, adj_pred_rna = generator_B(rna_features_test, rna_edge_index_test, rna_coord_test)    
                        print()
                   
                        
                        
                        if(ISRNA):
                            #rna_recon_cycle_loss_test = loss_rna(preds=retval1_test, theta=retval2_test, truth=rna_features_test)
                            rna_recon_cycle_loss_test_1 = loss_rna(preds=retval1_test, theta=retval2_test, truth=rna_features_test)
                            rna_recon_cycle_loss_test , recon, kl= generator_B.compute_elbo(mu_atac_test, logstad_atac_test, rna_recon_cycle_loss_test_1,)
                            #reg_loss_rna = generator_A.recon_loss(graph_nei_test, adj_pred_rna.float().detach())
                            #kl_rna = generator_A.kl_loss(mu_rna, logstad_rna)
                            #rna_recon_cycle_loss_test = log_zinb_positive(rna_features_test, retval1_test, retval2_test, retval3_test).mean()
                            atac_recon_cycle_loss_test = loss_bce(y_test, atac_features_test)
                            
                            #retval1_test, retval2_test, retval3_test, adj_pred_test = generator_B.decoder(rna_latent_recon_test, rna_coord_test)
                            #rna_recon_cycle_loss_test_2 = loss_rna(preds=retval1_test, theta=retval2_test, truth=rna_features_test,)
                        else :
                            #rna_recon_cycle_loss_test = loss_rna(preds=retval1_test, theta=retval2_test, truth=rna_features_test)
                            #reg_loss_rna = generator_A.recon_loss(graph_nei_test, adj_pred_rna.float().detach())
                            #kl_rna = generator_A.kl_loss(mu_rna, logstad_rna)
                            #rna_recon_cycle_loss_test = log_zinb_positive(rna_features_test, retval1_test, retval2_test, retval3_test).mean()
                                
                            #atac_recon_cycle_loss_test = loss_bce(y_test,atac_features_test) 
                            
                            atac_recon_cycle_loss_test_1 = loss_bce(y_test[:, indices_not_zero],atac_features_test[:,indices_not_zero])
                            atac_recon_cycle_loss_test, recon, kl = generator_B.compute_elbo(mu_atac_test, logstad_atac_test, atac_recon_cycle_loss_test_1,)
                            #reg_loss_atac = generator_B.recon_loss(graph_nei_test, adj_pred_atac.float().detach())
                            #kl_atac = generator_B.kl_loss(mu_atac, logstd_atac)
                            
                            #y_pred_test, adj_pred_test= generator_B.decoder(atac_latent_recon_test, atac_coord_test)
                            #atac_recon_cycle_loss_test_2 = loss_bce(y_pred_test, atac_features_test)
                            
                            
                        if ISRNA:
                            #loss_test = rna_recon_cycle_loss + reg_loss_rna * 0.01 + lambda_cycle * (cycle_loss_rna ) #  + kl_rna
                            loss_test = rna_recon_cycle_loss_test # + 0.01 * (cycle_loss_rna_test) + rna_recon_cycle_loss_test_2  #* 10 + reg_loss_rna * 0.01 + lambda_cycle * (cycle_loss_rna ) #  + kl_rna
                        else :
                            #loss_test = atac_recon_cycle_loss  + reg_loss_atac + lambda_cycle * (cycle_loss_atac) #  + kl_atac
                            loss_test = atac_recon_cycle_loss_test #* 10 + reg_loss_atac * 0.01 # 1 * (cycle_loss_atac_test) + atac_recon_cycle_loss_test_2 *  0.01 #* 0.1 + reg_loss_atac * 0.01 + lambda_cycle * (cycle_loss_atac) #  + kl_atac
                        print(f"loss_teset Loss: {loss_test.item():.4f}")
                        #loss_test = atac_recon_cycle_loss_test + rna_recon_cycle_loss_test #+ lambda_cycle * (cycle_loss_rna_test + cycle_loss_atac_test)
                        valid_losses_1.append(loss_test.item())

                    # 
                    loss1_test_history.append(np.mean(valid_losses_1))
                    logging.info(f"Test loss: {loss1_test_history[-1]:.4f}")

                    # 
                    early_stopping(loss1_test_history[-1], generator_B)
                    if early_stopping.early_stop:
                        logging.info("Early stopping triggered!")
                        if ISRNA:
                            truth = valid_rna.x#.to(device_rna)
                            predict_rna(truth, generator_B, train_atac, outdir_name, rna2atac, atac2rna,sc_rna_train_dataset_save)
                        else:
                            truth = valid_atac.x.to(device_rna)
                            print(truth.shape)
                            predict_atac(truth, generator_B, valid_rna, outdir_name, rna2atac, atac2rna,sc_atac_train_dataset_save, mask, inverted_mask)
                        break

                    # 
                    #fig = plot_loss_history(loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossATAC-RNA.{args.ext}"))
                    #plt.close(fig)

            return loss1_history, loss2_history, loss1_test_history
        
        
        
        #
        
        
        train_vgae(model = generatorATAC, optimizer = optimizer_atac_1, data = train_rna, epochs = 200, device = 'cuda:4', truth_atac = train_atac,discriminator =  ATACdiscriminator, updaterD = optimizer_D_atac, indices_not_zero = indices_not_zero)
        #torch.save(generatorATAC.state_dict(),os.path.join(outdir_name, f"ATACgenerator.pth"))
        #torch.save(generatorATAC.state_dict(),os.path.join(outdir_name, f"ATACgenerator.pth"))
        #torch.save(generatorATAC.state_dict(),os.path.join(outdir_name, f"ATACgenerator.pth"))
        #train_vgae_atac(model = generatorRNA, optimizer =  optimizer_rna_1, data = train_atac, epochs = 300, device = 'cuda:0', truth_rna = train_rna, discriminator = RNAdiscriminator, updaterD = optimizer_D_rna)
        #
        #torch.save(generatorRNA.state_dict(),os.path.join(outdir_name, f"RNAgenerator.pth"))
            
    
        #generatorRNA.load_state_dict(torch.load(os.path.join(outdir_name, "RNAgenerator.pth")))
        #generatorATAC.load_state_dict(torch.load(os.path.join(outdir_name, "ATACgenerator.pth")))
        
        # logging.info("training ATAC -> RNA with cycle_consistency")
        # loss1_history, loss2_history, loss1_test_history = train_cycle_consistency(generator_A=generatorATAC, generator_B=generatorRNA, num_epochs=500, train_iter=train_loader, 
        # test_iter=test_loader,truth_iter = truth_loader,updaterG_A=optimizer_A, updaterG_B=optimizer_B, lambda_cycle=0.01, discriminator = RNAdiscriminator, cuda=True, ISRNA=True)
        
        
        # loss visualization
        # fig = plot_loss_history(
        #     loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossATAC-RNA.{args.ext}")
        # )
        # plt.close(fig)
      
        logging.info("training RNA -> ATAC with cycle_consistency")
        loss2_history, loss2_history, loss2_test_history = train_cycle_consistency(generator_A=generatorRNA, generator_B=generatorATAC, num_epochs=500, train_iter=train_loader,
        test_iter=test_loader,truth_iter = truth_loader,updaterG_A=optimizer_B_2, updaterG_B=optimizer_A_2, lambda_cycle=1, discriminator =  ATACdiscriminator, cuda=True, ISRNA=False,sc_atac_train_dataset_save= sc_atac_train_dataset_save,sc_rna_train_dataset_save= sc_rna_train_dataset_save,mask=indices_not_zero, inverted_mask = genes_in_test_set)
        
        # loss visualization
        #fig = plot_loss_history(
        #    loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossATAC-RNA.{args.ext}")
        #)
        #plt.close(fig)
         
        
        logging.info("training ATAC -> RNA")
        #loss1_history,loss2_history,loss1_test_history=train(generator=generatorRNA, discriminator=None,num_epochs=60, train_iter=train_iter2, test_iter=test_iter2,truth_iter=truth_iter_atac, truth=sc_rna_truth, updaterG=optimizer_rna_1, updaterD=None,ISRNA=True)
        #loss visualization
        #fig = plot_loss_history(
        #    loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossATAC-RNA.{args.ext}")
        #)
        #plt.close(fig)
        

        logging.info("........................................................................................................................................................")
        logging.info("training RNA -> ATAC")
        #loss1_history, loss2_history, loss1_test_history = train(generator=generatorATAC, discriminator=None, num_epochs=60, train_iter=train_iter1, test_iter=test_iter1, truth_iter=truth_iter_rna, truth=sc_atac_truth, updaterG=optimizer_atac_1, updaterD=None, ISRNA=False)
        # loss visualization
        #fig = plot_loss_history(
        #    loss1_history, loss2_history, loss1_test_history, os.path.join(outdir_name, f"lossRNA-ATAC.{args.ext}")
        #)
        #plt.close(fig)
        torch.save(generatorRNA.state_dict(),os.path.join(outdir_name, f"RNAgenerator_2.pth"))
        #torch.save(RNAdiscriminator.state_dict(), os.path.join(outdir_name, f"RNAdiscriminator.pth"))
        #torch.save(generatorATAC.state_dict(),os.path.join(outdir_name, f"ATACgenerator_2.pth"))
        #torch.save(ATACdiscriminator.state_dict(), os.path.join(outdir_name, f"ATACdiscriminator.pth"))
        #torch.save(rna2atac.state_dict(),os.path.join(outdir_name, f"rna2atac.pth"))
        #torch.save(atac2rna.state_dict(),os.path.join(outdir_name, f"atac2rna.pth"))


if __name__ == "__main__":
    main()