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


sc_rna_test_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/spleen1_protein/adata_RNA_2.h5ad')
sc_atac_test_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/spleen1_protein/adata_ADT_2.h5ad')

# 
#obs_names = sc_rna_test_dataset.obs_names
#var_names = sc_atac_test_dataset.var_names

#
#zero_matrix = np.zeros((len(obs_names), len(var_names)))

# 
#adata_save = ad.AnnData(X=zero_matrix, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
#adata_save.obs = sc_rna_test_dataset.obs
#print(adata_save)


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
    parser.add_argument("--task", default='protein', type=str, help="task to predict")
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

    return train_rna,train_atac, val_rna, val_atac, test_rna, test_atac
    
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
    train_x_rna[val_idx,:] = 0  # 
    train_x_rna[test_idx,:] = 0  #

    train_x_atac = graph_atac.x.clone()
    train_x_atac[val_idx,:] = 0  # 
    train_x_atac[test_idx,:] = 0  #

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





def preprocess_combined_graph(adata, feature_type, k=4, metric='cosine'):
    #coordinates = torch.tensor(list(zip(adata.obs['x'], adata.obs['y'])), dtype=torch.float32)
    coordinates = torch.tensor(adata.obsm['spatial'], dtype=torch.float32)
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
    elif feature_type == 'protein':
        edge_index_feature = build_adjacency_from_atac(features, k=k)
    else:
        raise ValueError("Invalid feature_type. Use 'rna' or 'protein'.")

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



def predict_protein(truth,generator,truth_iter, outdir_name, rna2atac, atac2rna,):
    logging.info("....................................Evaluating Protein ")
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
                
        return ret,z
    sc_rna_atac_truth_preds,z = predict1(generator,truth_iter)
    if isinstance(sc_rna_atac_truth_preds, torch.Tensor):
        tt = sc_rna_atac_truth_preds.cpu().numpy()  # 
    if isinstance(sc_rna_atac_truth_preds, np.ndarray):
        tt = sc_rna_atac_truth_preds.astype(np.float32)
        
    #sc_atac_test_dataset.X = tt
    #sc_atac_test_dataset.write_h5ad("/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/spleen1_protein/predict_pro_ours_mouse_spleen.h5ad")
    
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
                ret = retval1    
        return ret
    
    sc_rna_truth_preds = predict2(generator,truth_iter, rna2atac, atac2rna)
    if isinstance(sc_rna_truth_preds, torch.Tensor):
        tt = sc_rna_truth_preds.cpu().numpy()  # 
    if isinstance(sc_rna_truth_preds, np.ndarray):
        tt = sc_rna_truth_preds.astype(np.float32)
    #sc_rna_train_dataset_save.X = tt
    #sc_rna_train_dataset_save.write_h5ad("/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/spleen1_protein/predict_rna_ours_mouse_spleen.h5ad")
 
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
  
def clr_transform(x: np.ndarray, add_pseudocount: bool = True) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = x.toarray()
    assert isinstance(x, np.ndarray)
    if add_pseudocount:
        x = x + 1.0
    if len(x.shape) == 1:
        denom = scipy.stats.mstats.gmean(x)
        retval = np.log(x / denom)
    elif len(x.shape) == 2:
        # Assumes that each row is an independent observation
        # and that columns denote features
        per_row = []
        for i in range(x.shape[0]):
            denom = scipy.stats.mstats.gmean(x[i])
            row = np.log(x[i] / denom)
            per_row.append(row)
        assert len(per_row) == x.shape[0]
        retval = np.stack(per_row)
        assert retval.shape == x.shape
    else:
        raise ValueError(f"Cannot CLR transform array with {len(x.shape)} dims")
    return retval


def main():
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
    
    def loss_protein_G(fake, discriminator, edge_index, spatial_coords):
        loss1 = -torch.mean(discriminator(fake, edge_index, spatial_coords))  #
        return loss1 

    sc_rna_train_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/spleen1_protein/adata_RNA_1.h5ad')
    sc_protein_train_dataset=ad.read_h5ad('/mnt/datab/home/zhaohui/WBT/scMOG-main/scMOG_code/spleen1_protein/adata_ADT_1.h5ad')
    sc_rna_train_dataset_save = sc_rna_train_dataset
    
  
    sc.pp.normalize_total(sc_rna_train_dataset) 
    sc.pp.log1p(sc_rna_train_dataset)
    
    
    sc_protein_train_dataset.X = clr_transform(sc_protein_train_dataset.X)
   
    
    
    
    graph_rna = preprocess_combined_graph(sc_rna_train_dataset, feature_type='rna', k=14, metric='cosine' )
    graph_protein = preprocess_combined_graph(sc_protein_train_dataset, feature_type='protein', k=14, metric='cosine')
  
    # Split data
    #train_rna, train_protein, valid_rna, valid_protein,test_rna, test_protein = split_graph_data(graph_rna, graph_protein)
    print("Evalulate numbers:")
    #print(torch.sum(train_protein.x == 0))
    #print(torch.sum(test_protein.x == 0))
    train_rna =  graph_rna
    train_protein = graph_protein
    test_rna = graph_rna
    test_protein = graph_protein
    valid_rna = graph_rna
    valid_protein = graph_protein
    
    
    
   
    train_dataset = PairedGraphDataset(train_rna, train_protein)
    test_dataset = PairedGraphDataset(test_rna, test_protein)
    valid_dataset = PairedGraphDataset(valid_rna, valid_protein)
    
    batch_size = 32
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    truth_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define model and optimizer
    in_channels_rna = graph_rna.x.size(1)  # RNA
    in_channels_protein = graph_protein.x.size(1)  # protein 
    hidden_channels = 128 
    latent_channels = 64
    
    hidden_channels_2 = 64  
    latent_channels_2 = 32 
  
    num_outputs_rna = graph_protein.x.size(1)  
    num_outputs_protein = graph_rna.x.size(1)  # protein 
    
    
    generatorprotein = both_GAN_1_ours.VGAEModel_rna_protein(in_channels_rna, hidden_channels_2, latent_channels_2, num_outputs_rna)  # Protein
    #input_dim2 = chrom_counts.values.tolist()
    generatorRNA = both_GAN_1_ours.VGAEModel_protein_rna(in_channels_protein, hidden_channels, latent_channels, num_outputs_protein)  # RNADecoder
    
    
    
    rna2protein = both_GAN_1_ours.AffineTransform(64, 128, affine_num=9)
    protein2rna = both_GAN_1_ours.AffineTransform(64, 128, affine_num=9)
    
    RNAdiscriminator = both_GAN_1_ours.Discriminator1(input_dim=sc_rna_train_dataset.X.shape[1])
    proteindiscriminator = both_GAN_1_ours.Discriminator1(input_dim=sc_protein_train_dataset.X.shape[1])
    
    cuda = True if torch.cuda.is_available() else False
    device_ids = range(torch.cuda.device_count())  
   
    #chrom_counts = sc_protein_train_dataset.var['chrom'].value_counts()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(np.max(sc_rna_train_dataset.X), np.min(sc_rna_train_dataset.X))
    print(np.max(sc_protein_train_dataset.X), np.min(sc_protein_train_dataset.X))
    
    dd = 0.0001
    dd2 = 0.0001
    weight_decay = 1e-5
    
    optimizer_protein_1 = torch.optim.RMSprop(generatorprotein.parameters(), lr=dd2, weight_decay=weight_decay)
    optimizer_rna_1 = torch.optim.RMSprop(generatorRNA.parameters(), lr=dd , weight_decay=weight_decay)
    optimizer_A = torch.optim.Adam(generatorRNA.parameters(), lr=dd , weight_decay=weight_decay)
    optimizer_B = torch.optim.Adam(generatorprotein.parameters(), lr=dd2, weight_decay=weight_decay)
    optimizer_A_2 = torch.optim.Adam(generatorRNA.parameters(), lr=dd,weight_decay=weight_decay)
    optimizer_B_2 = torch.optim.Adam(generatorprotein.parameters(), lr=dd2, weight_decay=weight_decay )
    optimizer_rna = torch.optim.Adam(generatorRNA.parameters(), lr=dd, weight_decay=weight_decay)
    optimizer_protein = torch.optim.Adam(generatorprotein.parameters(), lr=dd2, weight_decay=weight_decay)
   
    optimizer_D_rna = torch.optim.RMSprop(RNAdiscriminator.parameters(), lr=dd, weight_decay=weight_decay)
    optimizer_D_protein = torch.optim.RMSprop(proteindiscriminator.parameters(), lr=dd, weight_decay=weight_decay)
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
        

        
        def train_vgae(model, optimizer, data, epochs, device, truth_protein, discriminator, updaterD):
            discriminator.train()
            model.train()
            discriminator.to(device)
            model = model.to(device)
            data = data.to(device)
            
    
            loss_history = []
            trainD_losses=[]
            #print(data)
            y_hat = truth_protein.x.to(device)
            print(y_hat)
            
    
            graph_nei = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0))[0]
            graph_neg = 1 - graph_nei # 
            graph_neg.fill_diagonal_(0)  # 
            
            for epoch in range(epochs):
                #torch.autograd.set_detect_anomaly(True)
                updaterD.zero_grad()
                z, y, mu, logstd, adj_pred = model(data.x.to(device), data.edge_index.to(device),data.coordinates.to(device))
                
                loss2 = loss_D(y, y_hat, discriminator, data.edge_index, data.coordinates)
                loss2.backward()
                updaterD.step()
                trainD_losses.append(loss2.item())
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)
                print(f"Epoch {epoch + 1}/{epochs}, Loss_2: {loss2.item():.4f}")
                if epoch % 1 == 0:
                    optimizer.zero_grad()
                    z, y, mu, logstd, adj_pred = model(data.x.to(device), data.edge_index.to(device),data.coordinates.to(device))
                    reg_loss = model.recon_loss(adj_pred, graph_nei.to(device))
                    #loss = loss_protein_G(y, discriminator)
                    loss = loss_protein_G(y, discriminator, data.edge_index, data.coordinates)     
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
            return loss_history
        
        def train_vgae_protein(model, optimizer, data, epochs, device, truth_rna, discriminator, updaterD):
            model.train()
            discriminator.train()
            model = model.to(device)
            
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
                loss2 = loss_D(retval1, y_hat, discriminator, data.edge_index, data.coordinates)
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
                    loss = loss_rna_G(retval1, discriminator, data.edge_index, data.coordinates)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()
                    
                    loss_history.append(loss.item())
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            return loss_history
          
            
        def train_cycle_consistency(generator_A, generator_B, num_epochs, train_iter, test_iter,truth_iter,
                            updaterG_A, updaterG_B, lambda_cycle, discriminator, cuda=True, ISRNA=True, sc_rna_train_dataset_save = sc_rna_train_dataset_save):
            # 
            device_rna = 'cuda:1'
            device_protein = 'cuda:1'
            loss1_history, loss2_history, loss1_test_history = [], [], []
            generator_A.to(device_protein)
            generator_B.to(device_rna)
            rna2protein.to(device_rna)
            protein2rna.to(device_rna)
            valid_protein.to(device_rna)
            valid_rna.to(device_rna)
            
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
                    
                    
                    # protein
                    protein_features = train_protein.x.to(device_protein)  # protein
                    protein_edge_index = train_protein.edge_index.to(device_protein)  # protein
                    protein_edge_coord = train_protein.coordinates.to(device_protein) #RNA 
                   
                    updaterG_A.zero_grad()
                    if(ISRNA): 
                        z_rna, y, mu_rna, logstad_rna, adj_pred_rna = generator_A(rna_features, rna_edge_index,rna_edge_coord)
                        z_protein, retval1, retval2, retval3, mu_protein, logstd_protein, adj_pred_protein = generator_B(protein_features, protein_edge_index,protein_edge_coord)
                    else :
                        z_protein, retval1, retval2, retval3, mu_protein, logstd_protein, adj_pred_protein = generator_A(protein_features, protein_edge_index,protein_edge_coord)
                        z_rna, y, mu_rna, logstad_rna, adj_pred_rna = generator_B(rna_features, rna_edge_index,rna_edge_coord)
                    
                    if(ISRNA) :
                        rna_recon_cycle_loss_1 = loss_rna(preds=retval1, theta=retval2, truth=rna_features,)
                        rna_recon_cycle_loss = loss_rna(preds=retval1, theta=retval2, truth=rna_features,)
                    else : 
                        protein_recon_cycle_loss_1 = loss_bce(y, protein_features)
                        protein_recon_cycle_loss, recon, kl = generator_B.compute_elbo(mu_protein, logstd_protein, protein_recon_cycle_loss_1)
                
                    if ISRNA:
                        loss_A = rna_recon_cycle_loss
                    else :
                        loss_A =  protein_recon_cycle_loss
                    print(f"loss_A Loss: {loss_A.item():.4f}")
                    
                    loss_A.backward()
                    updaterG_A.step()
                    train_losses_1.append(loss_A.item())
                   
                if ((epoch + 1) % 10== 0 and epoch > 450):
                    if ISRNA:
                        truth = valid_rna.x#.to(device)
                        predict_rna(truth, generator_B, valid_protein, outdir_name,  rna2protein, protein2rna, sc_rna_train_dataset_save)
                    else:
                        truth = valid_protein.x#.to(device)
                        print(truth.shape)
                        print(valid_rna.x.shape)
                        predict_protein(truth, generator_B, valid_rna, outdir_name, rna2protein, protein2rna)
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
                    
                        # protein
                        protein_features_test = test_protein.x.to(device_protein)  # protein
                        protein_edge_index_test = test_protein.edge_index.to(device_protein)  # protein
                        protein_coord_test = test_protein.coordinates.to(device_protein)  #RNA 
                            
                        if(ISRNA) :
                            z_rna_test, y_test, mu_rna_test, logstad_rna_test, adj_pred_rna = generator_A(rna_features_test, rna_edge_index_test, rna_coord_test)
                            z_protein_test, retval1_test, retval2_test, retval3_test, mu_protein_test, logstd_protein_test, adj_pred_protein = generator_B(protein_features_test, protein_edge_index_test, protein_coord_test)
                        else: 
                            z_protein_test, retval1_test, retval2_test, retval3_test, mu_protein_test, logstd_protein_test, adj_pred_protein = generator_A(protein_features_test, protein_edge_index_test, protein_coord_test)
                            z_rna_test, y_test, mu_rna_test, logstad_rna_test, adj_pred_rna = generator_B(rna_features_test, rna_edge_index_test, rna_coord_test)    
                        
                        if(ISRNA):
                            rna_recon_cycle_loss_test_1 = loss_rna(preds=retval1_test, theta=retval2_test, truth=rna_features_test,)
                            rna_recon_cycle_loss_test , recon, kl= generator_B.compute_elbo(mu_rna_test, logstad_rna_test, rna_recon_cycle_loss_test_1,) 
                        else :
                            protein_recon_cycle_loss_test_1 = loss_bce(y_test,protein_features_test)
                            protein_recon_cycle_loss_test, recon, kl = generator_B.compute_elbo(mu_protein_test, logstd_protein_test, protein_recon_cycle_loss_test_1,)
                            
                        if ISRNA:
                            loss_test = rna_recon_cycle_loss_test 
                        else :
                            loss_test = protein_recon_cycle_loss_test
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
                            predict_rna(truth, generator_B, valid_protein, outdir_name, rna2protein, protein2rna, sc_rna_train_dataset_save)
                        else:
                            truth = valid_protein.x#.to(device_rna)
                            predict_protein(truth, generator_B,valid_rna, outdir_name, rna2protein, protein2rna)
                        break

                   

            return loss1_history, loss2_history, loss1_test_history
        
        
        
        train_vgae(model = generatorprotein, optimizer = optimizer_protein_1, data = train_rna, epochs = 200, device = 'cuda:0', truth_protein = train_protein,discriminator =  proteindiscriminator, updaterD = optimizer_D_protein)

        train_vgae_protein(model = generatorRNA, optimizer =  optimizer_rna_1, data = train_protein, epochs = 200, device = 'cuda:0', truth_rna = train_rna, discriminator = RNAdiscriminator, updaterD = optimizer_D_rna)
        
            
        # load_model to predict
        if task == 'predict_rna':
            generatorRNA.load_state_dict(torch.load(os.path.join(outdir_name, "RNAgenerator.pth")))
            truth_rna = valid_rna.x#.to(device_rna)
            predict_rna(truth_rna, generatorRNA,valid_protein, outdir_name, rna2protein, protein2rna, sc_rna_train_dataset_save)
        
        if task == 'predict_protein':
            generatorprotein.load_state_dict(torch.load(os.path.join(outdir_name, "proteingenerator.pth")))
            truth = valid_protein.x#.to(device_rna)
            predict_protein(truth, generatorprotein,valid_rna, outdir_name, rna2protein, protein2rna)
            
        if args.task == 'rna':
            logging.info("........................................................................................................................................................")
            logging.info("training protein -> RNA with cycle_consistency")
            loss1_history, loss2_history, loss1_test_history = train_cycle_consistency(generator_A=generatorprotein, generator_B=generatorRNA, num_epochs=500, train_iter=train_loader, 
            test_iter=test_loader,truth_iter = truth_loader,updaterG_A=optimizer_A, updaterG_B=optimizer_B, lambda_cycle=0.01, discriminator = RNAdiscriminator, cuda=True, ISRNA=True, sc_rna_train_dataset_save = sc_rna_train_dataset_save)
        
        if args.task == 'protein':
            logging.info("........................................................................................................................................................")
            logging.info("training RNA -> protein with cycle_consistency")
            loss2_history, loss2_history, loss2_test_history = train_cycle_consistency(generator_A=generatorRNA, generator_B=generatorprotein, num_epochs=300, train_iter=train_loader, 
            test_iter=test_loader,truth_iter = truth_loader,updaterG_A=optimizer_B_2, updaterG_B=optimizer_A_2, lambda_cycle=1, discriminator =  proteindiscriminator, cuda=True, ISRNA=False)
            
      
        torch.save(generatorRNA.state_dict(),os.path.join(outdir_name, f"RNAgenerator.pth"))
        #torch.save(RNAdiscriminator.state_dict(), os.path.join(outdir_name, f"RNAdiscriminator.pth"))
        torch.save(generatorprotein.state_dict(),os.path.join(outdir_name, f"proteingenerator.pth"))
        #torch.save(proteindiscriminator.state_dict(), os.path.join(outdir_name, f"proteindiscriminator.pth"))


if __name__ == "__main__":
    main()
