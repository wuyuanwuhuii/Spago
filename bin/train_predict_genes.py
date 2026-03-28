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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Spago"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

MODELS_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)


import both_GAN_1_ours
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

def cosine_similarity(emb):
    norm = torch.norm(emb, dim=1, keepdim=True)  # 
    emb_normalized = emb / (norm + 1e-8)  # 
    similarity = torch.mm(emb_normalized, emb_normalized.T)
    return similarity

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
    parser.add_argument("--task", default='predict_rna', type=str, help="task to predict")
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
    
    #
    kf = KFold(n_splits=n_splits, shuffle=True)

    # 
    spot_folds_bool = {}

    # 
    for spot in all_spots:
        spot_index = np.where(all_spots == spot)[0][0]  
        
        spot_folds_bool[spot] = np.zeros((n_splits, len(all_genes)), dtype=bool)
        
        for i, (train_index, test_index) in enumerate(kf.split(all_genes)):
            
            spot_folds_bool[spot][i, train_index] = True 
            spot_folds_bool[spot][i, test_index] = False   
    
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



def predict_atac(truth,generator,truth_iter, outdir_name, sc_atac_train_dataset_save,mask, ininverted_mask, save_path):
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
                ret = y_pred
                
        return ret
    
    sc_rna_atac_truth_preds = predict1(generator,truth_iter)
    
    if isinstance(sc_rna_atac_truth_preds, torch.Tensor):
        tt = sc_rna_atac_truth_preds.cpu().numpy()  # 
            
            
    if isinstance(sc_rna_atac_truth_preds, np.ndarray):
        tt = sc_rna_atac_truth_preds.astype(np.float32)
    
    # if isinstance(ininverted_mask, torch.Tensor):
    #     ininverted_mask = ininverted_mask.cpu().numpy()  #
    
    if isinstance(ininverted_mask, torch.Tensor):
        ininverted_mask = ininverted_mask.bool()  # 
    
    sc_atac_train_dataset_save.X = np.where(ininverted_mask, tt, sc_atac_train_dataset_save.X)
    
    full_path = os.path.join(save_path, "predict_atac_fold_1.h5ad")
    sc_atac_train_dataset_save.write_h5ad(full_path)

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

def predict_rna(truth,generator,truth_iter, outdir_name, sc_rna_train_dataset_save, mask, inverted_mask, save_path):
    logging.info(".........................................Evaluating  RNA")

    def predict2(generator,truth_iter):
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
                ret = retval1
                    
        return ret
    
    sc_rna_truth_preds = predict2(generator,truth_iter)
    if isinstance(sc_rna_truth_preds, torch.Tensor):
        tt = sc_rna_truth_preds.cpu().numpy()  # 
            
            
    if isinstance(sc_rna_truth_preds, np.ndarray):
        tt = sc_rna_truth_preds.astype(np.float32)
    print(sum(inverted_mask[0,:]))
    sc_rna_train_dataset_save.X = np.where(inverted_mask, tt, sc_rna_train_dataset_save.X)
    full_path = os.path.join(save_path, "predict_rna_fold_1.h5ad")
    sc_rna_train_dataset_save.write_h5ad(full_path)
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


def save_spot_folds_to_csv(sc_atac_train_dataset, spot_folds_bool, save_dir="spot_folds_csv"):
    
    os.makedirs(save_dir, exist_ok=True)  

    for spot, bool_matrix in spot_folds_bool.items():
        #
        df = pd.DataFrame(bool_matrix, columns=np.array(sc_atac_train_dataset.var_names), 
                          index=[f"Fold_{i+1}" for i in range(bool_matrix.shape[0])])
        
        #
        spot_name = str(spot).replace("/", "_").replace("\\", "_")  # 
        file_path = os.path.join(save_dir, f"{spot_name}_folds.csv")
        df.to_csv(file_path)
        print(f"save {spot_name}_folds.csv to {save_dir}")

def uniform_grid_mask(x, y, keep_ratio=0.2, tol=0.01, max_iter=25, random_state=0, pick='first'):
   
    rng = np.random.default_rng(random_state)
    n = len(x)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    # 
    diag = np.hypot(xmax - xmin, ymax - ymin)
    s_lo, s_hi = diag / (n * 10), diag  

    best_idx = None
    for _ in range(max_iter):
        s = 0.5 * (s_lo + s_hi)
        gx = np.floor((x - xmin) / s).astype(np.int64)
        gy = np.floor((y - ymin) / s).astype(np.int64)
        # =
        cell2idx = {}
        if pick == 'first':
            for i, c in enumerate(zip(gx, gy)):
                if c not in cell2idx:
                    cell2idx[c] = i
        elif pick == 'random':
            order = rng.permutation(n)
            for i in order:
                c = (gx[i], gy[i])
                if c not in cell2idx:
                    cell2idx[c] = i
        keep_idx = np.fromiter(cell2idx.values(), dtype=np.int64)
        keep_ratio_now = keep_idx.size / n

        # 
        if best_idx is None or abs(keep_ratio_now - keep_ratio) < abs(best_idx.size / n - keep_ratio):
            best_idx = keep_idx

        # 
        if keep_ratio_now > keep_ratio:   
            s_lo = s
        else:                             
            s_hi = s

        if abs(keep_ratio_now - keep_ratio) <= tol:
            best_idx = keep_idx
            break

    target_k = int(round(keep_ratio * n))
    if best_idx.size > target_k:
        best_idx = np.sort(best_idx)[:target_k]
    elif best_idx.size < target_k:
        #
        remain = np.setdiff1d(np.arange(n), np.sort(best_idx), assume_unique=True)
        add = rng.choice(remain, size=target_k - best_idx.size, replace=False)
        best_idx = np.sort(np.concatenate([best_idx, add]))

    mask = np.zeros(n, dtype=bool)
    mask[best_idx] = True
    return mask


def save_random_mask_to_csv(sc_atac_train_dataset, mask_ratio=0.5, output_dir="masked_data"):
    
    all_spots = sc_atac_train_dataset.obs_names
    all_genes = sc_atac_train_dataset.var_names

    os.makedirs(output_dir, exist_ok=True)

    mask_matrix = []

    for spot in all_spots:
        num_genes = len(all_genes)
        num_masked_genes = int(np.floor(mask_ratio * num_genes))
        
        masked_genes_idx = np.random.choice(num_genes, num_masked_genes, replace=False)
        mask_vector = np.zeros(num_genes, dtype=bool)
        
        mask_vector[masked_genes_idx] = True
        
        mask_matrix.append(mask_vector)

    mask_df = pd.DataFrame(mask_matrix, index=all_spots, columns=all_genes)

    output_file_path = os.path.join(output_dir, "random_mask_matrix.csv")
    
    mask_df.to_csv(output_file_path)
    print(f"save mask matrix to {output_file_path}")
from scipy.stats import spearmanr

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
    
    
    # for gene score matrix
    loss_bce = losses.MSELoss()
    # for chrom count matrix
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

    sc_rna_train_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/data_p21/adata_common_rna.h5ad')
    sc_atac_train_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/data_p21/adata_common_atac.h5ad')
    #sc_atac_train_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/data_p21/adata_p21_save.h5ad')
    #sc_atac_old = ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/data_p21/predict_atac_fold_1_ours_new.h5ad')
    print("INFO----RNA and ATAC")
    print(sc_rna_train_dataset)
    print(sc_atac_train_dataset)
    print(np.max(sc_rna_train_dataset.X), np.min(sc_rna_train_dataset.X))
    print(np.max(sc_atac_train_dataset.X), np.min(sc_atac_train_dataset.X))
    
    
    # RNA preprocess
    sc.pp.normalize_total(sc_rna_train_dataset)
    sc.pp.log1p(sc_rna_train_dataset)
    
    # chrom count matrix preprocess preprocess
    #sc.pp.normalize_total(sc_atac_train_dataset)
    #sc.pp.log1p(sc_atac_train_dataset)
    sc_atac_train_dataset = scale(sc_atac_train_dataset)
    
    # gene score matrix preprocess
    #sc_atac_train_dataset.X[sc_atac_train_dataset.X>0] = 1 
    
    
    n_spots, n_genes = sc_atac_train_dataset.shape
    n_folds = 5
     
    fold_masks = [np.ones((n_spots, n_genes), dtype=bool) for _ in range(n_folds)]
    
    rng = np.random.default_rng(seed=42) 
    
    for i in range(n_spots):
        gene_indices = np.arange(n_genes)
        rng.shuffle(gene_indices)
        split_indices = np.array_split(gene_indices, n_folds)
        
        for fold_idx in range(n_folds): 
            test_genes = split_indices[fold_idx]
            fold_masks[fold_idx][i, test_genes] = False  # 
    
    mask_value = fold_masks[1]
    
    inverted_mask_value = ~mask_value 
    
    mask = mask_value.astype(bool).astype(int)
    inverted_mask = inverted_mask_value.astype(bool).astype(int)
    # 
    genes_to_zero = torch.tensor(inverted_mask, dtype=torch.bool)
    indices_not_zero = torch.tensor(mask, dtype=torch.bool)
    
    sc_atac_train_dataset_save = sc_atac_train_dataset
    sc_rna_train_dataset_save = sc_rna_train_dataset
    

    
    graph_rna = preprocess_combined_graph(sc_rna_train_dataset, feature_type='rna', k=10, metric='cosine' )
    graph_atac = preprocess_combined_graph(sc_atac_train_dataset, feature_type='atac', k=10, metric='cosine')
    graph_atac_valid = preprocess_combined_graph(sc_atac_train_dataset, feature_type='rna', k=10, metric='cosine' )
   
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
    hidden_channels = 128 
    latent_channels = 64 
    
    hidden_channels_2 = 128 
    latent_channels_2 = 64 
    
    num_outputs_rna = graph_atac.x.size(1)  # 
    num_outputs_atac = graph_rna.x.size(1)  # ATAC 
    
    
    generatorATAC = both_GAN_1_ours.VGAEModel_rna(in_channels_rna, hidden_channels_2, latent_channels_2, num_outputs_rna)  # ATACDecoder
    #input_dim2 = chrom_counts.values.tolist()
    generatorRNA = both_GAN_1_ours.VGAEModel_atac(in_channels_atac, hidden_channels, latent_channels, num_outputs_atac)  # RNADecoder
    
    
    RNAdiscriminator = both_GAN_1_ours.Discriminator1(input_dim=sc_rna_train_dataset.X.shape[1] )
    ATACdiscriminator = both_GAN_1_ours.Discriminator1(input_dim=sc_atac_train_dataset.X.shape[1])
    
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
    
    optimizer_atac_1 = torch.optim.RMSprop(generatorATAC.parameters(), lr=dd2, weight_decay=weight_decay)
    optimizer_rna_1 = torch.optim.RMSprop(generatorRNA.parameters(), lr=dd , weight_decay=weight_decay)
    optimizer_A = torch.optim.Adam(generatorRNA.parameters(), lr=dd , weight_decay=weight_decay)
    optimizer_B = torch.optim.Adam(generatorATAC.parameters(), lr=dd2, weight_decay=weight_decay)
    
    
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
        
        
        
        def train_vgae(model, optimizer, data, epochs, device, truth_atac, discriminator, updaterD, indices_not_zero):
            device = 'cuda:0'
            discriminator.train()
            model.train()
            discriminator.to(device)
            model = model.to(device)
            data = data.to(device)
            
            loss_history = []
            trainD_losses=[]
            
            y_hat = truth_atac.x.to(device)
            
            indices_not_zero = indices_not_zero.to(device)
    
            graph_nei = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0]
            graph_neg = 1 - graph_nei 
            graph_neg.fill_diagonal_(0)  
            
            for epoch in range(epochs):
                
                updaterD.zero_grad()
                z, y, mu, logstd, adj_pred = model(data.x.to(device), data.edge_index.to(device),data.coordinates.to(device))
              
                loss2 = loss_D(y * indices_not_zero, y_hat * indices_not_zero, discriminator, data.edge_index, data.coordinates.to(device))
                
                loss2.backward()
                updaterD.step()
                trainD_losses.append(loss2.item())
                #for p in discriminator.parameters():
                #    p.data.clamp_(-args.clip_value, args.clip_value)
                print(f"Epoch {epoch + 1}/{epochs}, Loss_2: {loss2.item():.4f}")
                if epoch % 1 == 0:
                    optimizer.zero_grad()
                    z, y, mu, logstd, adj_pred = model(data.x.to(device), data.edge_index.to(device),data.coordinates.to(device))
                    
                    loss = loss_atac_G(y * indices_not_zero, discriminator, data.edge_index, data.coordinates)
                    loss.backward()

                    optimizer.step()
                    loss_history.append(loss.item())
                    
                    
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
            return loss_history
        
        def train_vgae_atac(model, optimizer, data, epochs, device, truth_rna, discriminator, updaterD, indices_not_zero):
            device = 'cuda:0'
            model.train()
            discriminator.train()
            model = model.to(device)
            
            indices_not_zero = indices_not_zero.to(device)
            data = data.to(device)
            
            discriminator.to(device)
            loss_history = []
            trainD_losses=[]
            
            size_factors_rna = torch.ones(data.x.size(0)).to(device)
            y_hat = truth_rna.x.to(device)
            
            
            from torch_geometric.utils import to_dense_adj
    
            graph_nei = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0]  # 
            graph_neg = 1 - graph_nei  # 
            graph_neg.fill_diagonal_(0)  # 
            
            
            for epoch in range(epochs):
                
                z, retval1, retval2, retval3, mu, logstd, adj_pred = model(
                data.x.to(device), data.edge_index.to(device),data.coordinates.to(device)
            )
                updaterD.zero_grad()
                
                loss2 = loss_D(retval1 * indices_not_zero, y_hat * indices_not_zero, discriminator, data.edge_index, data.coordinates)
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
                    loss = loss_rna_G(retval1 * indices_not_zero, discriminator, data.edge_index, data.coordinates)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()
                    
                    loss_history.append(loss.item())
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
            return loss_history
          
            
        def train_cycle_consistency(generator_A, generator_B, num_epochs, train_iter, test_iter,truth_iter,
            updaterG_A, updaterG_B, lambda_cycle, discriminator, cuda=True, ISRNA=True,sc_atac_train_dataset_save= sc_atac_train_dataset_save,
            sc_rna_train_dataset_save= sc_rna_train_dataset_save, mask=indices_not_zero, inverted_mask = genes_to_zero, save_path = args.outdir):
            # 
            device_rna = 'cuda:0'
            device_atac = 'cuda:0'
            loss1_history, loss2_history, loss1_test_history = [], [], []
            generator_A.to(device_atac)
            generator_B.to(device_rna)
            
            valid_atac.to(device_rna)
            valid_rna.to(device_rna)
            indices_not_zero = mask.to(device_atac)
            
        
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
                        z_rna, y, mu_rna, logstad_rna, adj_pred_rna = generator_A(rna_features * indices_not_zero, rna_edge_index,rna_edge_coord)
                        z_atac, retval1, retval2, retval3, mu_atac, logstad_atac, adj_pred_atac = generator_B(atac_features * indices_not_zero, atac_edge_index,atac_edge_coord)
                    else :
                        z_atac, retval1, retval2, retval3, mu_atac, logstad_atac, adj_pred_atac = generator_A(atac_features * indices_not_zero, atac_edge_index,atac_edge_coord)
                        z_rna, y, mu_rna, logstad_rna, adj_pred_rna = generator_B(rna_features * indices_not_zero, rna_edge_index,rna_edge_coord)
                    
                    
                    if(ISRNA) :
                        rna_recon_cycle_loss_1 = loss_rna(preds=retval1* indices_not_zero, theta=retval2 * indices_not_zero, truth=rna_features* indices_not_zero,)
                        rna_recon_cycle_loss, recon, kl = generator_B.compute_elbo(mu_rna, logstad_rna, rna_recon_cycle_loss_1,)
                    else :
                        atac_recon_cycle_loss_1 = loss_bce(y * indices_not_zero, atac_features * indices_not_zero)
                        atac_recon_cycle_loss, recon, kl = generator_B.compute_elbo(mu_atac, logstad_atac, atac_recon_cycle_loss_1)
                    if ISRNA:
                        loss_A = rna_recon_cycle_loss 
                    else :
                        loss_A =  atac_recon_cycle_loss
                    print(f"loss_A Loss: {loss_A.item():.4f}")
                    
                    loss_A.backward()
                    updaterG_A.step()
                    train_losses_1.append(loss_A.item())
                   
                if ((epoch + 1) % 10== 0 and epoch > 350):
                    if ISRNA:
                        truth = valid_rna.x#.to(device)
                        predict_rna(truth, generator_B, train_atac, outdir_name, sc_rna_train_dataset_save, mask, inverted_mask, save_path)
                    else:
                        truth = valid_atac.x.to(device_rna)
                        predict_atac(truth, generator_B, valid_rna, outdir_name, sc_atac_train_dataset_save,mask, inverted_mask, save_path)
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
                            z_rna_test, y_test, mu_rna_test, logstad_rna_test, adj_pred_rna = generator_A(rna_features_test * indices_not_zero, rna_edge_index_test, rna_coord_test)
                            z_atac_test, retval1_test, retval2_test, retval3_test, mu_atac_test, logstd_atac_test, adj_pred_atac = generator_B(atac_features_test, atac_edge_index_test, atac_coord_test)
                        else: 
                            z_atac_test, retval1_test, retval2_test, retval3_test, mu_atac_test, logstad_atac_test, adj_pred_atac = generator_A(atac_features_test * indices_not_zero, atac_edge_index_test, atac_coord_test)
                            z_rna_test, y_test, mu_rna_test, logstad_rna_test, adj_pred_rna = generator_B(rna_features_test, rna_edge_index_test, rna_coord_test)    
                        

                        if(ISRNA):
                            rna_recon_cycle_loss_test_1 = loss_rna(preds=retval1_test * indices_not_zero, theta=retval2_test * indices_not_zero, truth=rna_features_test * indices_not_zero)
                            rna_recon_cycle_loss_test , recon, kl= generator_B.compute_elbo(mu_atac_test, logstd_atac_test, rna_recon_cycle_loss_test_1,)
                        else :
                            atac_recon_cycle_loss_test_1 = loss_bce(y_test * indices_not_zero,atac_features_test * indices_not_zero)
                            atac_recon_cycle_loss_test, recon, kl = generator_B.compute_elbo(mu_atac_test, logstd_atac_test, atac_recon_cycle_loss_test_1,)
                            
                        if ISRNA:
                            loss_test = rna_recon_cycle_loss_test
                        else :
                            loss_test = atac_recon_cycle_loss_test
                        print(f"loss_teset Loss: {loss_test.item():.4f}")
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
                            predict_rna(truth, generator_B, train_atac, outdir_name, sc_rna_train_dataset_save, mask, inverted_mask, save_path)
                        else:
                            truth = valid_atac.x.to(device_rna)
                            predict_atac(truth, generator_B, valid_rna, outdir_name, sc_atac_train_dataset_save, mask, inverted_mask, save_path)
                        break

            return loss1_history, loss2_history, loss1_test_history
        
        # pre_training
        train_vgae(model = generatorATAC, optimizer = optimizer_atac_1, data = train_rna, epochs = 200, device = 'cuda:4', truth_atac = train_atac,discriminator =  ATACdiscriminator, updaterD = optimizer_D_atac, indices_not_zero = indices_not_zero)
        
        train_vgae_atac(model = generatorRNA, optimizer =  optimizer_rna_1, data = train_atac, epochs = 300, device = 'cuda:0', truth_rna = train_rna, discriminator = RNAdiscriminator, updaterD = optimizer_D_rna, indices_not_zero = indices_not_zero)
        
        
        # logging.info("training ATAC -> RNA with cycle_consistency")
        # loss1_history, loss2_history, loss1_test_history = train_cycle_consistency(generator_A=generatorATAC, generator_B=generatorRNA, num_epochs=500, train_iter=train_loader, 
        # test_iter=test_loader,truth_iter = truth_loader,updaterG_A=optimizer_A, updaterG_B=optimizer_B, lambda_cycle=0.01, discriminator = RNAdiscriminator, cuda=True, ISRNA=True)
        
        
        
        # train atac
        if args.task == 'predict_atac':
            loss2_history, loss2_history, loss2_test_history = train_cycle_consistency(generator_A=generatorRNA, generator_B=generatorATAC, num_epochs=400, train_iter=train_loader,
            test_iter=test_loader,truth_iter = truth_loader,updaterG_A=optimizer_B_2, updaterG_B=optimizer_A_2, lambda_cycle=1, discriminator =  ATACdiscriminator, cuda=True, ISRNA=False,sc_atac_train_dataset_save= sc_atac_train_dataset_save,sc_rna_train_dataset_save= sc_rna_train_dataset_save,mask=indices_not_zero, inverted_mask = genes_to_zero, save_path = args.outdir)
        
        if args.task == 'predict_rna':
            loss2_history, loss2_history, loss2_test_history = train_cycle_consistency(generator_A=generatorATAC, generator_B=generatorRNA, num_epochs=400, train_iter=train_loader,
            test_iter=test_loader,truth_iter = truth_loader,updaterG_A=optimizer_A, updaterG_B=optimizer_B, lambda_cycle=1, discriminator =  ATACdiscriminator, cuda=True, ISRNA=True,sc_atac_train_dataset_save= sc_atac_train_dataset_save,sc_rna_train_dataset_save= sc_rna_train_dataset_save,mask=indices_not_zero, inverted_mask = genes_to_zero, save_path = args.outdir)
         
        torch.save(generatorRNA.state_dict(),os.path.join(outdir_name, f"RNAgenerator.pth"))
        #torch.save(RNAdiscriminator.state_dict(), os.path.join(outdir_name, f"RNAdiscriminator.pth"))
        torch.save(generatorATAC.state_dict(),os.path.join(outdir_name, f"ATACgenerator.pth"))
        #torch.save(ATACdiscriminator.state_dict(), os.path.join(outdir_name, f"ATACdiscriminator.pth"))
        #torch.save(rna2atac.state_dict(),os.path.join(outdir_name, f"rna2atac.pth"))
        #torch.save(atac2rna.state_dict(),os.path.join(outdir_name, f"atac2rna.pth"))


if __name__ == "__main__":
    main()
