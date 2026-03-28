import os
import sys

from typing import List, Tuple, Union, Callable

from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn
from torch_geometric.nn import VGAE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import activations
from torch_geometric.nn import GCNConv, Sequential, BatchNorm,GATConv,SGConv,TAGConv, SAGEConv,TransformerConv,GraphSAGE, GATv2Conv, GraphSAGE
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution_old(Module):
    

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

device='cpu'
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        adj = torch.sparse_coo_tensor(
            adj,
            torch.ones(adj.shape[1]),
            (input.size(0), input.size(0))
        ).to_dense().to(device)
        
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

cuda = True if torch.cuda.is_available() else False
import numpy as np

class Decoder(Module):
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        
        return x   

def random_uniform_init(input_dim, output_dim, seed):
    np.random.seed(seed)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    torch.manual_seed(seed)
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, seed, input_dim, output_dim, activation=torch.sigmoid):
        super(GraphConvSparse, self).__init__()
        torch.manual_seed(seed)
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, inputs, edge_index):
        adj_sparse = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.shape[1]), 
            (inputs.size(0), inputs.size(0))
        )
        adj_sparse = adj_sparse.to(inputs.device)
        x = torch.mm(inputs, self.weight)  
        x = torch.sparse.mm(adj_sparse, x)  
        outputs = self.activation(x)
        return outputs


class AffineTransform(nn.Module):
    def __init__(self, input_dim, z_dim, affine_num=9, affine_layer_num=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.affine_layer_num = affine_layer_num

        # affine matrix init
        self.affine_matrices = nn.ParameterList(
            [nn.Parameter(torch.stack([torch.randn(input_dim, input_dim).flatten() for _ in range(affine_num)])) for _ in range(affine_layer_num)]
        )
        self.affine_offsets = nn.ParameterList(
            [nn.Parameter(torch.randn(affine_num, input_dim)) for _ in range(affine_layer_num)]
        )

        # regressor for the affine transform selection
        self.fc_loc = nn.Sequential(
            nn.Linear(input_dim, z_dim),
            nn.ReLU(True),
            nn.Linear(z_dim, affine_num)
        )
        self.act = nn.PReLU()

    def forward(self, x):
        soft_idx = F.softmax(self.fc_loc(x), dim=-1)

        output = x.unsqueeze(-1)
        for i in range(self.affine_layer_num):
            affine_matrix = torch.mm(soft_idx, self.affine_matrices[i])
            affine_matrix = affine_matrix.view(-1, self.input_dim, self.input_dim)  # [b, d, d]
            affine_offset = torch.mm(soft_idx, self.affine_offsets[i])
            affine_offset = affine_offset.unsqueeze(-1)  # [b, d, 1]

            # do affine transform
            output = torch.bmm(affine_matrix, output) + affine_offset

            # do activation until output
            if i < self.affine_layer_num - 1:
                output = self.act(output)

        output = output.squeeze(-1)
        return output


class ATACEncoder(nn.Module):
    def __init__(self, num_inputs: int, num_units=32,seed: int = 182822,):
        super().__init__()
        torch.manual_seed(seed)  ##Seed the CPU to generate random numbers so that the results are deterministic

        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode0 = nn.Linear(self.num_inputs, 512)
        nn.init.xavier_uniform_(self.encode0.weight)
        self.bn0 = nn.BatchNorm1d(512)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        self.encode1 = nn.Linear(512, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.encode2 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act0(self.bn0(self.encode0(x)))
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return x
        


class SplitEnc(nn.Module):
    def __init__(self, input_dim, z_dim, SUB_1=64, SUB_2=32, seed: int = 182822):
        super(SplitEnc, self).__init__()
        torch.manual_seed(seed)  # Set the seed for reproducibility

        self.input_dim = input_dim  # List of input dimensions for each part
        self.split_layer = nn.ModuleList()

        # Construct layers for each split part
        for n in self.input_dim:
            # Layer 1
            layer1 = nn.Linear(n, SUB_1)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(SUB_1)
            act1 = nn.PReLU()

            # Layer 2
            layer2 = nn.Linear(SUB_1, SUB_2)
            nn.init.xavier_uniform_(layer2.weight)
            bn2 = nn.BatchNorm1d(SUB_2)
            act2 = nn.PReLU()

            self.split_layer.append(
                nn.ModuleList([layer1, bn1, act1, layer2, bn2, act2])
            )

        # Final encoding layer
        self.enc2 = nn.Linear(SUB_2 * len(input_dim), z_dim)
        self.bn2 = nn.BatchNorm1d(z_dim)

        # Initialize the final layers
        nn.init.xavier_uniform_(self.enc2.weight)

    def forward(self, x):
        # Split the input tensor based on the input dimensions
        xs = torch.split(x, self.input_dim, dim=1)

        # Process each split part through its layers
        assert len(xs) == len(self.input_dim)
        enc_chroms = []
        for init_mod, chrom_input in zip(self.split_layer, xs):
            for f in init_mod:
                chrom_input = f(chrom_input)  # Apply layers sequentially
            enc_chroms.append(chrom_input)

        # Concatenate all encoded parts
        enc1 = torch.cat(enc_chroms, dim=1)

        # Final encoding (latent space)
        chromosome_enc = self.bn2(self.enc2(enc1))

        # Return the latent variable z (chromosome_enc is z)
        return chromosome_enc


class RNAEncoder_sdss(nn.Module):
    def __init__(self, num_inputs: int, num_units=32,seed: int = 182822,):
        super().__init__()
        torch.manual_seed(seed)  ##Seed the CPU to generate random numbers so that the results are deterministic
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode0 = nn.Linear(self.num_inputs, 256)
        nn.init.xavier_uniform_(self.encode0.weight)
        self.bn0 = nn.BatchNorm1d(256)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        self.encode1 = nn.Linear(256, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.encode2 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act0(self.bn0(self.encode0(x)))
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return x
        
class VGAEEncoder_protein(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, spatial_coord_dim=2, dropout_rate=0, K= 3):
        super(VGAEEncoder_protein, self).__init__()

        # Define the linear encoder
        self.linear_encoder = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            #nn.PReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(dropout_rate),
            #nn.Linear(512,256),
            #nn.BatchNorm1d(256),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.PReLU(),
            #nn.Dropout(dropout_rate)
        )

        self.conv1 = GATv2Conv(in_channels, 32)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = GATv2Conv(32,hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATv2Conv(128,hidden_channels)
        self.fusion_layer = nn.Linear(in_channels + spatial_coord_dim, in_channels + spatial_coord_dim)
        
        
        self.conv_mu = GATv2Conv(hidden_channels, latent_channels)
        self.conv_logstd = GATv2Conv(hidden_channels, latent_channels)
        
        self.elu1 = nn.ELU()  # 定义ELU激活层
        self.elu2 = nn.ELU()  # 定义ELU激活层
        self.dropout = nn.Dropout(dropout_rate)
        self.K = K  # Number of Gaussians in GMM
   
    def encode(self, x, edge_index, spatial_coord=None):
        #if spatial_coord is not None:
        #    x = torch.cat([x, spatial_coord], dim=1)
        #    x = self.fusion_layer(x)

        # Pass through the linear encoder
        #x = self.linear_encoder(x)

        # Apply the first graph convolution layer
        x = F.elu(self.conv1(x, edge_index))
        #x = self.conv1(x, edge_index).relu()
        #x = self.bn1(x)  # Apply batch normalization after GCN
        #x = self.dropout(x)  # Apply dropout for regularization
        
        #x = self.conv2(x, edge_index).relu()
        x = F.elu(self.conv2(x, edge_index))
        #x = self.bn2(x)
        #x = self.conv3(x, edge_index).relu()
       

        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        

        return mu, logstd

    def reparameterize(self, mu, logstd, noise_factor=0.1, min_std=-5.0, max_mu=10.0, max_std=10.0):
        # Compute standard deviation from log variance
        #std = torch.exp(logstd)
    
        # Apply size restrictions (clamp) on the standard deviation and mean
        #std = torch.clamp(std, min=min_std, max=max_std)  # Ensure std stays within a valid range
        #mu = torch.clamp(mu, max=max_mu)  # Limit mu to a maximum value
        
        # Generate noise
        #eps = torch.randn_like(std)  # Standard normal noise
        #z = eps * std + mu  # Reparameterized latent variable z
        std = torch.exp(logstd)  # Compute standard deviation from logstd
        eps = torch.randn_like(std)  # Generate noise
        z = eps * std + mu  # Reparameterized latent variable z
        return z
        
        return z

    def forward(self, x, edge_index, spatial_coord=None):
        # Encoding step to obtain the mean and log variance
        mu, logstd = self.encode(x, edge_index, spatial_coord)

        # Sample the latent variable using the reparameterization trick
        z = self.reparameterize(mu, logstd)

        return mu, logstd, z

class VGAEEncoder_rna_protein(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, spatial_coord_dim=2, dropout_rate=0, K= 3):
        super(VGAEEncoder_rna_protein, self).__init__()

        # Define the linear encoder
        self.linear_encoder = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            #nn.PReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(dropout_rate),
            #nn.Linear(512,256),
            #nn.BatchNorm1d(256),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.PReLU(),
            #nn.Dropout(dropout_rate)
        )

        self.conv1 = GATv2Conv(in_channels, 128)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = GATv2Conv(128,hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATv2Conv(128,hidden_channels)
        self.fusion_layer = nn.Linear(in_channels + spatial_coord_dim, in_channels + spatial_coord_dim)
        
        
        self.conv_mu = GATv2Conv(hidden_channels, latent_channels)
        self.conv_logstd = GATv2Conv(hidden_channels, latent_channels)
        
        self.elu1 = nn.ELU()  # 定义ELU激活层
        self.elu2 = nn.ELU()  # 定义ELU激活层
        self.dropout = nn.Dropout(dropout_rate)
        self.K = K  # Number of Gaussians in GMM
   
    def encode(self, x, edge_index, spatial_coord=None):
        #if spatial_coord is not None:
        #    x = torch.cat([x, spatial_coord], dim=1)
        #    x = self.fusion_layer(x)

        # Pass through the linear encoder
        #x = self.linear_encoder(x)

        # Apply the first graph convolution layer
        x = F.elu(self.conv1(x, edge_index))
        #x = self.conv1(x, edge_index).relu()
        #x = self.bn1(x)  # Apply batch normalization after GCN
        #x = self.dropout(x)  # Apply dropout for regularization
        
        x = self.conv2(x, edge_index).relu()
        #x = F.elu(self.conv2(x, edge_index))
        #x = self.bn2(x)
        #x = self.conv3(x, edge_index).relu()
       

        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        

        return mu, logstd

    def reparameterize(self, mu, logstd, noise_factor=0.1, min_std=-5.0, max_mu=10.0, max_std=10.0):
        # Compute standard deviation from log variance
        std = torch.exp(logstd)
    
        # Apply size restrictions (clamp) on the standard deviation and mean
        std = torch.clamp(std, min=min_std, max=max_std)  # Ensure std stays within a valid range
        mu = torch.clamp(mu, max=max_mu)  # Limit mu to a maximum value
        
        # Generate noise
        eps = torch.randn_like(std)  # Standard normal noise
        z = eps * std + mu  # Reparameterized latent variable z
        
        return z

    def forward(self, x, edge_index, spatial_coord=None):
        # Encoding step to obtain the mean and log variance
        mu, logstd = self.encode(x, edge_index, spatial_coord)

        # Sample the latent variable using the reparameterization trick
        z = self.reparameterize(mu, logstd)

        return mu, logstd, z

class VGAEEncoder_old(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, spatial_coord_dim = 2, dropout_rate=0.5):
        super(VGAEEncoder_old, self).__init__()
        
        # Define the encoder as the variational encoder
        self.conv1 = GraphConvolution(in_channels, 128, k = 4)
        self.conv2 = GraphConvolution(128, hidden_channels, k = 4)
        self.conv_mu = GraphConvolution(hidden_channels, latent_channels, k = 4)
        self.conv_logstd = GraphConvolution(hidden_channels, latent_channels, k = 4)
        self.fusion_layer = nn.Linear(in_channels + spatial_coord_dim, in_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def encode(self, x, edge_index, spatial_coord = None):
        
        if spatial_coord is not None:
            x = torch.cat([x, spatial_coord], dim=1)
            x = self.fusion_layer(x)
        
        # Apply the first SGConv layer with ReLU activation
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        # Get the mean and log variance of the latent space
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        
        return mu, logstd

    def reparameterize(self, mu, logstd):
     
        std = torch.exp(logstd)  # Compute standard deviation from logstd
        eps = torch.randn_like(std)  # Generate noise
        z = eps * std + mu  # Reparameterized latent variable z
        return z

    def forward(self, x, edge_index, spatial_coord = None):
        # Encoding step to obtain the mean and log variance
        mu, logstd = self.encode(x, edge_index, spatial_coord)
        
        # Sample the latent variable using the reparameterization trick
        z = self.reparameterize(mu, logstd)
        
        return mu, logstd, z
    
class VGAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, spatial_coord_dim=2, dropout_rate=0, K= 3):
        super(VGAEEncoder, self).__init__()

        # Define the linear encoder
        self.linear_encoder = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            #nn.PReLU(),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(dropout_rate),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.PReLU(),
            #nn.Dropout(dropout_rate)
        )

        self.conv1 = GCNConv(512 ,hidden_channels)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128,hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(128,hidden_channels)
        self.fusion_layer = nn.Linear(in_channels + spatial_coord_dim, in_channels + spatial_coord_dim)
        
        
        self.conv_mu = GCNConv(hidden_channels, latent_channels)
        self.conv_logstd = GCNConv(hidden_channels, latent_channels)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.K = K  # Number of Gaussians in GMM
   
    def encode(self, x, edge_index, spatial_coord=None):
        #if spatial_coord is not None:
            #x = torch.cat([x, spatial_coord], dim=1)
        #    x = self.fusion_layer(x)

        # Pass through the linear encoder
        x = self.linear_encoder(x)

        # Apply the first graph convolution layer
        x = self.conv1(x, edge_index).relu()
        #x = self.bn1(x)  # Apply batch normalization after GCN
        #x = self.dropout(x)  # Apply dropout for regularization
        
        #x = self.conv2(x, edge_index).relu()
       

        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        

        return mu, logstd

    def reparameterize(self, mu, logstd, noise_factor=0.1):
        std = torch.exp(logstd)  # Compute standard deviation from log variance
        eps = torch.randn_like(std)  # Generate noise
        z = eps * std + mu  # Reparameterized latent variable z
        noise = torch.randn_like(z) * noise_factor
        z_noisy = z + noise  
        #z_transformed, _ = self.flow(z)  
        #return z_transformed
        return z

    def forward(self, x, edge_index, spatial_coord=None):
        # Encoding step to obtain the mean and log variance
        mu, logstd = self.encode(x, edge_index, spatial_coord)

        # Sample the latent variable using the reparameterization trick
        z = self.reparameterize(mu, logstd)

        return mu, logstd, z



class VGAEEncoder_rna(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(VGAEEncoder_rna, self).__init__()
        
        # Replace the encoder with the new GraphConvolution
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.conv_mu = GraphConvolution(hidden_channels, latent_channels)
        self.conv_logstd = GraphConvolution(hidden_channels, latent_channels)

    def encode(self, x, adj):
        # Apply the first GCN layer with ReLU activation
        x = self.conv1(x, adj).relu()  
        
        # Get the mean and log variance of the latent space
        mu = self.conv_mu(x, adj)
        logstd = self.conv_logstd(x, adj)
        
        return mu, logstd

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)  # Compute standard deviation from logstd
        eps = torch.randn_like(std)  # Generate noise
        z = eps * std + mu  # Reparameterized latent variable z
        return z

    def forward(self, x, adj):
        adj = torch.sparse_coo_tensor(
            adj,
            torch.ones(adj.shape[1]),
            (x.size(0), x.size(0))
        ).to_dense().cuda()
        # Encoding step to obtain the mean and log variance
        mu, logstd = self.encode(x, adj)
        
        # Sample the latent variable using the reparameterization trick
        z = self.reparameterize(mu, logstd)
        return mu, logstd, z

        
class VGAEEncoder_atac_pro(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, spatial_coord_dim=2, dropout_rate=0, K=3):
        super(VGAEEncoder_atac_pro, self).__init__()

        # Define the linear encoder for initial feature transformation
        self.linear_encoder = nn.Sequential(
            nn.Linear(in_channels,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #nn.init.xavier_uniform_(self.linear_encoder[0].weight)  
        #nn.init.xavier_uniform_(self.linear_encoder[3].weight)  
        # Define the graph convolution layers
        self.conv1 = TAGConv(in_channels,512)
        self.norm1 = torch.nn.LayerNorm(128)
        self.conv2 = TAGConv(512, hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.conv3 = TAGConv(256, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.act1 = nn.ReLU()
        #self.conv2 = GCN(512, 256)
        #self.conv3 = GCN(256, hidden_channels)
        
        # Define the layers for the GMM parameters
        
        self.conv_mu = TAGConv(hidden_channels, latent_channels)
        self.norm3 = torch.nn.LayerNorm(latent_channels)
        self.conv_logstd = TAGConv(hidden_channels, latent_channels)
        self.norm4 = torch.nn.LayerNorm(latent_channels)
        #self.conv_mu = nn.Linear(hidden_channels, latent_channels)
        #self.conv_logstd = nn.Linear(hidden_channels, latent_channels)
        #self.conv_mu = SGConv(hidden_channels, latent_channels )  # K Gaussians for each latent dimension
        #self.conv_logstd = SGConv(hidden_channels, latent_channels )  # K Gaussians for logstd
        self.conv_pi = SGConv(hidden_channels, latent_channels )  # K Gaussians for mixing coefficients (pi)

        self.K = K  # Number of Gaussians in the GMM

    def encode(self, x, edge_index, spatial_coord=None):
          
        #if spatial_coord is not None:
            # Concatenate spatial coordinates with the input features
        #    x = torch.cat([x, spatial_coord], dim=1)
        
        #x = self.linear_encoder(x)
        # Apply the graph convolution layers
        x = self.conv1(x, edge_index).relu()
        #x1 = self.norm1(x1)
        #x1 = F.relu(x1)
        
        x = self.conv2(x, edge_index).relu()
        #x2 = self.norm2(x2)
        #x2 = F.relu(x2)
        #x = x2
        #x = self.conv3(x, edge_index).relu()
        #x = self.conv3(x, edge_index).relu()

        # Compute the parameters for the GMM: K Gaussians
        mu = self.conv_mu(x, edge_index)  # mu for all K Gaussians
        #mu = self.norm3(mu)
        logstd = self.conv_logstd(x, edge_index)  # logstd for all K Gaussians
        #logstd = self.norm4(logstd)
        
        return mu, logstd


    def reparameterize(self, mu, logstd, noise_factor=0.1):
        
        std = torch.exp(logstd)  # Compute standard deviation from logstd
        #std = F.softplus(logstd)
        eps = torch.randn_like(std)  # Generate noise
        z = eps * std + mu  # Reparameterized latent variable z
        noise = torch.randn_like(z) * noise_factor
        z_noisy = z + noise  # 
        #z_transformed, _ = self.flow(z)  #
        #return z_transformed

        return z


    def forward(self, x, edge_index, spatial_coord=None):
        
        # Encoding step to obtain the mean, log variance, and mixture coefficients
        mu, logstd = self.encode(x, edge_index, spatial_coord)

        # Sample the latent variable using the reparameterization trick
        z = self.reparameterize(mu, logstd)

        return mu, logstd, z

class VGAEEncoder_atac(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, spatial_coord_dim=2, dropout_rate=0, K=3):
        super(VGAEEncoder_atac, self).__init__()

        # Define the linear encoder for initial feature transformation
        self.linear_encoder = nn.Sequential(
            nn.Linear(in_channels,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #nn.init.xavier_uniform_(self.linear_encoder[0].weight)  
        #nn.init.xavier_uniform_(self.linear_encoder[3].weight)  
        # Define the graph convolution layers
        self.conv1 = GATv2Conv(in_channels,256)
        self.norm1 = torch.nn.LayerNorm(128)
        self.conv2 = GATv2Conv(256, hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.conv3 = GATv2Conv(256, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.act1 = nn.ReLU()
        #self.conv2 = GCN(512, 256)
        #self.conv3 = GCN(256, hidden_channels)
        
        # Define the layers for the GMM parameters
        
        self.conv_mu = GATv2Conv(hidden_channels, latent_channels)
        self.norm3 = torch.nn.LayerNorm(latent_channels)
        self.conv_logstd = GATv2Conv(hidden_channels, latent_channels)
        self.norm4 = torch.nn.LayerNorm(latent_channels)

        self.conv_pi = SGConv(hidden_channels, latent_channels )  # K Gaussians for mixing coefficients (pi)

        self.K = K  # Number of Gaussians in the GMM

    def encode(self, x, edge_index, spatial_coord=None):
          
        #if spatial_coord is not None:
            # Concatenate spatial coordinates with the input features
        #    x = torch.cat([x, spatial_coord], dim=1)
        
        #x = self.linear_encoder(x)
        # Apply the graph convolution layers
        x = self.conv1(x, edge_index).relu()
        #x1 = self.norm1(x1)
        #x1 = F.relu(x1)
        
        x = self.conv2(x, edge_index).relu()
        #x2 = self.norm2(x2)
        #x2 = F.relu(x2)
        #x = x2
        #x = self.conv3(x, edge_index).relu()
        #x = self.conv3(x, edge_index).relu()

        # Compute the parameters for the GMM: K Gaussians
        mu = self.conv_mu(x, edge_index)  # mu for all K Gaussians
        #mu = self.norm3(mu)
        logstd = self.conv_logstd(x, edge_index)  # logstd for all K Gaussians
        #logstd = self.norm4(logstd)
        
        return mu, logstd


    def reparameterize(self, mu, logstd, noise_factor=0.1):
        
        std = torch.exp(logstd)  # Compute standard deviation from logstd
        #std = F.softplus(logstd)
        eps = torch.randn_like(std)  # Generate noise
        z = eps * std + mu  # Reparameterized latent variable z
        noise = torch.randn_like(z) * noise_factor
        z_noisy = z + noise  # 
        #z_transformed, _ = self.flow(z)  #
        #return z_transformed

        return z


    def forward(self, x, edge_index, spatial_coord=None):
        
        # Encoding step to obtain the mean, log variance, and mixture coefficients
        mu, logstd = self.encode(x, edge_index, spatial_coord)

        # Sample the latent variable using the reparameterization trick
        z = self.reparameterize(mu, logstd)

        return mu, logstd, z


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2 , nhid1),
            torch.nn.BatchNorm1d(nhid1),
            nn.LeakyReLU(0.2, inplace=True),
            #torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb, spatial_coords, size_factors=None):
        if spatial_coords is not None:
            x = torch.cat([emb, spatial_coords], dim=1)
            
        x = self.decoder(emb)
        adj_pred = torch.sigmoid(torch.matmul(x, x.t()))
        retval3 = torch.sigmoid(self.pi(x))
        retval2 = self.DispAct(self.disp(x))
        retval1 = self.MeanAct(self.mean(x))
        return retval1, retval2, retval3, adj_pred


# Decoder
class VGAEDecoder(nn.Module):
    def __init__(self):
        super(VGAEDecoder, self).__init__()

    def forward(self, z):
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_pred


# Reparameterize
def reparameterize(mu, logstd):
    std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


# VGAE Model for ATAC (with RNADecoder)
class VGAEModel_atac(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_outputs_rna, spatial_coord_dim = 2):
        super(VGAEModel_atac, self).__init__()
        self.encoder = VGAEEncoder_atac(in_channels, hidden_channels, latent_channels)
        self.decoder = RNADecoder(
            num_outputs=num_outputs_rna,
            num_units=latent_channels,
            final_activation=[activations.Exp(), activations.ClippedSoftplus(),nn.Sigmoid()],  
        )
        #self.decoder = Decoder(
        #    latent_channels,  
        #    num_outputs_rna,
        #)
        

    def forward(self, x, edge_index, spatial_coord, size_factors=None):
        mu, logstd,z = self.encoder(x,edge_index,spatial_coord)
        adj = to_dense_adj(edge_index)[0].to(z.device)  
        #z = reparameterize(mu, logstd)
        
        retval1, retval2, retval3, adj_pred = self.decoder(z, spatial_coord, size_factors=None)
        return z, retval1, retval2, retval3, mu, logstd, adj_pred

    def recon_loss(self, preds, truth):
        retval1 = preds
        loss1 = F.mse_loss(retval1, truth) 
        return loss1

    
    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(1 + 2 * logstd - mu**2 - torch.exp(2 * logstd))
    def compute_elbo(self, mu, logstd, recon, beta=0.01):
      
        kl = self.kl_loss(mu, logstd)  
        elbo_loss = recon + beta * kl 
        return elbo_loss.mean() , recon, kl.mean() 

class VGAEModel_atac_2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_outputs_rna, spatial_coord_dim = 2):
        super(VGAEModel_atac_2, self).__init__()
        self.encoder = VGAEEncoder_atac(in_channels, hidden_channels, latent_channels)
        self.decoder = ATACDecoder(
            num_outputs=num_outputs_rna,
            num_units=latent_channels,
            final_activation=nn.Sigmoid(),  
        )
        

    def forward(self, x, edge_index, spatial_coord = None,size_factors=None):
        
        mu, logstd, z = self.encoder(x, edge_index, spatial_coord)
        #z = reparameterize(mu, logstd)

        # 
        y, adj_pred = self.decoder(z,spatial_coord)
        
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return z, y, mu, logstd, adj_pred

    def recon_loss(self, adj_pred, adj_orig):
        # 
        loss = F.binary_cross_entropy(adj_pred, adj_orig)
        return loss

    def kl_loss(self, mu, logstd):
        
        kl = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
        return kl.mean()
    def compute_elbo(self, mu, logstd, recon, beta=0.01):
       
        kl = self.kl_loss(mu, logstd)  #
        elbo_loss = recon + beta * kl  # 
        return elbo_loss.mean() , recon, kl.mean()

  

class VGAEModel_rna(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_outputs_atac, spatial_coord_dim = 2):
        super(VGAEModel_rna, self).__init__()
        self.encoder = VGAEEncoder(in_channels, hidden_channels, latent_channels)
        self.decoder = ATACDecoder(
            num_outputs=num_outputs_atac,
            num_units=latent_channels,
            final_activation=nn.Sigmoid(),  
        )

    def forward(self, x, edge_index, spatial_coord = None,size_factors=None):
        
        mu, logstd, z = self.encoder(x, edge_index, spatial_coord)
        #z = reparameterize(mu, logstd)

        # 
        y, adj_pred = self.decoder(z,spatial_coord)
        
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        #print("ddd")
        #print(adj_pred.shape)
        #print(np.max(y.detach().cpu().numpy()))
        return z, y, mu, logstd, adj_pred

    def recon_loss(self, adj_pred, adj_orig):
        # 
        loss = F.binary_cross_entropy(adj_pred, adj_orig)
        return loss

    def kl_loss(self, mu, logstd):
        
        kl = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
        return kl.mean()
    def compute_elbo(self, mu, logstd, recon, beta=0.01):
       
        kl = self.kl_loss(mu, logstd)  #
        elbo_loss = recon + beta * kl  # 
        return elbo_loss.mean() , recon, kl.mean() 

class VGAEModel_rna_2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_outputs_atac, spatial_coord_dim = 2):
        super(VGAEModel_rna_2, self).__init__()
        self.encoder = VGAEEncoder(in_channels, hidden_channels, latent_channels)
        self.decoder = RNADecoder(
            num_outputs=num_outputs_atac,
            num_units=latent_channels,
            final_activation=[activations.Exp(), activations.ClippedSoftplus(),nn.Sigmoid()],  
        )

    def forward(self, x, edge_index, spatial_coord, size_factors=None):
        mu, logstd,z = self.encoder(x,edge_index,spatial_coord)
        adj = to_dense_adj(edge_index)[0].to(z.device)  
        #z = reparameterize(mu, logstd)
        
        retval1, retval2, retval3, adj_pred = self.decoder(z, spatial_coord, size_factors=None)
        return z, retval1, retval2, retval3, mu, logstd, adj_pred

    def recon_loss(self, preds, truth):
        retval1 = preds
        loss1 = F.mse_loss(retval1, truth) 
        return loss1

    
    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(1 + 2 * logstd - mu**2 - torch.exp(2 * logstd))
    def compute_elbo(self, mu, logstd, recon, beta=0.01):
      
        kl = self.kl_loss(mu, logstd)  
        elbo_loss = recon + beta * kl 
        return elbo_loss.mean() , recon, kl.mean()


class VGAEModel_rna_protein(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_outputs_protein, spatial_coord_dim = 2):
        super(VGAEModel_rna_protein, self).__init__()
        self.encoder = VGAEEncoder_rna_protein(in_channels, hidden_channels, latent_channels)
        self.decoder = ProteinDecoder(num_outputs=num_outputs_protein, num_units=latent_channels,final_activation=nn.Identity())
        #self.decoder = Decoder(
        #    latent_channels,  
        #    num_outputs_rna,
        #)
        

    def forward(self, x, edge_index, spatial_coord, size_factors=None):
    
        mu, logstd, z = self.encoder(x, edge_index, spatial_coord)
        #z = reparameterize(mu, logstd)

        # 
        y,adj_pred = self.decoder(z,spatial_coord)
        
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        #print("ddd")
        #print(adj_pred.shape)
        #print(np.max(y.detach().cpu().numpy()))
        return z, y, mu, logstd,adj_pred

    def recon_loss(self, adj_pred, adj_orig):
        # 
        loss = F.binary_cross_entropy(adj_pred, adj_orig)
        return loss

    def kl_loss(self, mu, logstd):
        
        kl = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
        return kl.mean()
    def compute_elbo(self, mu, logstd, recon, beta=0.01):
       
        kl = self.kl_loss(mu, logstd)  #
        elbo_loss = recon + beta * kl  # 
        return elbo_loss.mean() , recon, kl.mean()  
        

class VGAEModel_protein_rna(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_outputs_rna, spatial_coord_dim = 2):
        super(VGAEModel_protein_rna, self).__init__()
        self.encoder = VGAEEncoder_protein(in_channels, hidden_channels, latent_channels)
        self.decoder = RNADecoder(
             num_outputs=num_outputs_rna,
             num_units=latent_channels,
             final_activation=[activations.Exp(), activations.ClippedSoftplus(),nn.Sigmoid()],  
        )
        #self.decoder = Decoder(
        #    latent_channels,  
        #    num_outputs_rna,
        #)
        

    def forward(self, x, edge_index, spatial_coord, size_factors=None):
        mu, logstd,z = self.encoder(x,edge_index,spatial_coord)
        adj = to_dense_adj(edge_index)[0].to(z.device)  
        #z = reparameterize(mu, logstd)
        
        retval1, retval2, retval3, adj_pred = self.decoder(z, spatial_coord, size_factors=None)
        return z, retval1, retval2, retval3, mu, logstd, adj_pred

    def recon_loss(self, preds, truth):
        retval1 = preds
        loss1 = F.mse_loss(retval1, truth) 
        return loss1

    
    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(1 + 2 * logstd - mu**2 - torch.exp(2 * logstd))
    def compute_elbo(self, mu, logstd, recon, beta=0.1):
      
        kl = self.kl_loss(mu, logstd)  
        elbo_loss = recon + beta * kl 
        return elbo_loss.mean() , recon, kl.mean() 




class ATACDecoder(nn.Module):
    def __init__(
        self,
            num_outputs: int,
            num_units=16,
            #final_activation=nn.PReLU,
            final_activation=nn.Sigmoid,
            seed: int = 182822,
    ):
        super(ATACDecoder, self).__init__()
        torch.manual_seed(seed)  ##Seed the CPU to generate random numbers so that the results are deterministic
        self.num_outputs = num_outputs
        self.latent_dim = num_units

        self.decode1 = nn.Linear(self.latent_dim, 128)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.decode2 = nn.Linear(128, 512)
        nn.init.xavier_uniform_(self.decode2.weight)
        self.bn2 = nn.BatchNorm1d(512)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)


        self.decode3 = nn.Linear(512, self.num_outputs)
        nn.init.xavier_uniform_(self.decode3.weight)
        #nn.init.constant_(self.decode3.bias,-2.0)
        self.final_activations = final_activation

    def forward(self, x, spatial_coords):
        #if spatial_coords is not None:
        #    x = torch.cat([x, spatial_coords], dim=1)
        adj_pred = torch.sigmoid(torch.matmul(x, x.t()))
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.act2(self.bn2(self.decode2(x)))
        x = self.final_activations(self.decode3(x))

        return x, adj_pred


class ProteinDecoder(nn.Module):
    def __init__(
        self,
            num_outputs: int,
            num_units=16,
            # activation=nn.PReLU,
            final_activation=nn.Identity(),
            seed: int = 182822,
    ):
        super(ProteinDecoder, self).__init__()
        torch.manual_seed(seed)
        self.num_outputs = num_outputs
        self.latent_dim = num_units

        self.decode1 = nn.Linear(self.latent_dim, 256)
        #nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.decode2 = nn.Linear(256,self.num_outputs)
        #nn.init.xavier_uniform_(self.decode2.weight)
        self.final_activations = final_activation

    def forward(self, x, spatial_coords):
        adj_pred = torch.sigmoid(torch.matmul(x, x.t()))
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.final_activations(self.decode2(x))
        return x, adj_pred


class RNADecoder_orign(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_units: int = 16,
        intermediate_dim: int = 64,
        activation=nn.ReLU,
        final_activation: list = [activations.Exp(), activations.ClippedSoftplus(),nn.Sigmoid()],
        seed: int = 182822,

    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_outputs = num_outputs
        self.num_units = num_units
        assert len(final_activation) == 3
        self.final_activations = final_activation
        print("rrrr")
        print(final_activation)

        #self.decode1 = nn.Linear(self.num_units + 2, 64)
        self.decoder_1 = nn.Linear(self.num_units + 2,128)
        nn.init.xavier_uniform_(self.decoder_1.weight)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.ReLU()
        self.decoder_2 = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(256, self.num_outputs)
        self.disp = torch.nn.Linear(256, self.num_outputs)
        self.mean = torch.nn.Linear(256, self.num_outputs)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
        
        
        
        self.decode1 = nn.Linear(self.num_units , 64)
        #nn.init.kaiming_uniform_(self.decode1.weight, nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        #self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.act1 = nn.ReLU()
 
        self.decode2 = nn.Linear(64, 256)
        #nn.init.kaiming_uniform_(self.decode2.weight, nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.decode2.weight)
        self.bn2 = nn.BatchNorm1d(256)
        #self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.act2 = nn.ReLU()

        self.decode21 = nn.Linear(256, self.num_outputs)
        #nn.init.kaiming_uniform_(self.decode21.weight)
        nn.init.xavier_uniform_(self.decode21.weight)
        self.decode22 = nn.Linear(256, self.num_outputs)
        #nn.init.kaiming_uniform_(self.decode22.weight)
        nn.init.xavier_uniform_(self.decode22.weight)
        self.decode23 = nn.Linear(256, self.num_outputs)
        #nn.init.kaiming_uniform_(self.decode23.weight)
        nn.init.xavier_uniform_(self.decode23.weight)

        self.fusion_layer = nn.Linear(self.num_units + 2, self.num_units)
        self.prelu = nn.ELU()

    def forward(self, x, spatial_coords, size_factors=None):
      
        #x = torch.cat([x, spatial_coords], dim=1)
        #if spatial_coords is not None:
        #    x = torch.cat([x, spatial_coords], dim=1)
            #x = self.fusion_layer(x)
        print("rrrrrrrr")
        print(x.shape)
        print(spatial_coords.shape)
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.act2(self.bn2(self.decode2(x)))
        #x = self.act3(self.bn3(self.decoder_1(x)))
        #x = self.decoder_2(x)
        adj_pred = torch.sigmoid(torch.matmul(x, x.t()))
        retval1 = self.decode21(x)  # This is invariably the counts
        retval2 = self.decode22(x)
        retval3 = self.decode23(x)
        
        
        #retval1 = self.final_activations(retval1)
        #if "act1" in self.final_activations.keys():
        #    retval1 = self.final_activations["act1"](retval1)
        #if size_factors is not None:
        #    sf_scaled = size_factors.view(-1, 1).repeat(1, retval1.shape[1])
        #    retval1 = retval1 * sf_scaled  # Elementwise multiplication

        #retval2 = self.decode22(x)
        #if "act2" in self.final_activations.keys():
        #    retval2 = self.final_activations["act2"](retval2)

        #retval3 = self.decode23(x)
        #if "act3" in self.final_activations.keys():
        #    retval3 = self.final_activations["act3"](retval3)
        
        retval1 = self.final_activations[0](retval1)
        retval2 = self.final_activations[1](retval2)
        retval3 = self.final_activations[2](retval3)
        
        #retval3 = torch.sigmoid(self.pi(x))
        #retval2 = self.DispAct(self.disp(x))
        #retval1 = self.MeanAct(self.mean(x))
        
        return retval1,retval2,retval3, adj_pred


class RNADecoder(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_units: int = 16,
        intermediate_dim: int = 64,
        activation=nn.LeakyReLU,
        final_activation=None,
        seed: int = 182822,

    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_outputs = num_outputs
        self.num_units = num_units

        self.decode1 = nn.Linear(self.num_units, 64)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.decode2 = nn.Linear(64, 256)
        nn.init.xavier_uniform_(self.decode2.weight)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.decode21 = nn.Linear(256, self.num_outputs)
        nn.init.xavier_uniform_(self.decode21.weight)
        self.decode22 = nn.Linear(256, self.num_outputs)
        nn.init.xavier_uniform_(self.decode22.weight)
        self.decode23 = nn.Linear(256, self.num_outputs)
        nn.init.xavier_uniform_(self.decode23.weight)
        
        self.final_activations = nn.ModuleDict()
        if final_activation is not None:
            if isinstance(final_activation, list) or isinstance(
                final_activation, tuple
            ):
                assert len(final_activation) <= 3
                for i, act in enumerate(final_activation):
                    if act is None:
                        continue
                    self.final_activations[f"act{i+1}"] = act
            elif isinstance(final_activation, nn.Module):
                self.final_activations["act1"] = final_activation
            else:
                raise ValueError(
                    f"Unrecognized type for final_activation: {type(final_activation)}"
                )

    def forward(self, x, spatial_coords, size_factors=None):
        #if spatial_coords is not None:
        #    x = torch.cat([x, spatial_coords], dim=1)
        x = self.act1(self.bn1(self.decode1(x)))
        x = self.act2(self.bn2(self.decode2(x)))
        adj_pred = torch.sigmoid(torch.matmul(x, x.t()))
        retval1 = self.decode21(x)  # This is invariably the counts
        #retval1 = self.final_activations(retval1)
        if "act1" in self.final_activations.keys():
            retval1 = self.final_activations["act1"](retval1)
        if size_factors is not None:
            sf_scaled = size_factors.view(-1, 1).repeat(1, retval1.shape[1])
            retval1 = retval1 * sf_scaled  # Elementwise multiplication

        retval2 = self.decode22(x)
        if "act2" in self.final_activations.keys():
            retval2 = self.final_activations["act2"](retval2)

        retval3 = self.decode23(x)
        if "act3" in self.final_activations.keys():
            retval3 = self.final_activations["act3"](retval3)

        return retval1,retval2,retval3,adj_pred


class RNADecoder_DDDD(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_units: int = 16,
        intermediate_dim: int = 64,
        activation=nn.PReLU,
        final_activation: list = [activations.Exp(), activations.ClippedSoftplus(),nn.Sigmoid()],
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_units = num_units + 2

        self.decode1 = nn.Linear(self.num_units, intermediate_dim)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(intermediate_dim)
        self.act1 = activation()

        self.decode21 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode21.weight)
        self.decode22 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode22.weight)
        self.decode23 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode23.weight)

        self.final_activations = nn.ModuleDict()
        if final_activation is not None:
            if isinstance(final_activation, list) or isinstance(final_activation, tuple):
                assert len(final_activation) <= 3
                for i, act in enumerate(final_activation):
                    if act is None:
                        continue
                    self.final_activations[f"act{i+1}"] = act
            elif isinstance(final_activation, nn.Module):
                self.final_activations["act1"] = final_activation
            else:
                raise ValueError(f"Unrecognized type for final_activation: {type(final_activation)}")

    def forward(self, x, spatial_coords, size_factors=None):
        x = torch.cat([x, spatial_coords], dim=1)
        adj_pred = torch.sigmoid(torch.matmul(x, x.t()))
        x = self.act1(self.bn1(self.decode1(x)))

        retval1 = self.decode21(x)
        if "act1" in self.final_activations.keys():
            retval1 = self.final_activations["act1"](retval1)
        if size_factors is not None:
            sf_scaled = size_factors.view(-1, 1).repeat(1, retval1.shape[1])
            retval1 = retval1 * sf_scaled  

        retval2 = self.decode22(x)
        if "act2" in self.final_activations.keys():
            retval2 = self.final_activations["act2"](retval2)

        retval3 = self.decode23(x)
        if "act3" in self.final_activations.keys():
            retval3 = self.final_activations["act3"](retval3)

        return retval1, retval2, retval3, adj_pred



class Inference(nn.Module):
    def __init__(self, num_inputs: int,final_activation=nn.Sigmoid,):
        super().__init__()
        self.num_inputs = num_inputs
        #self.final_activation=final_activation

        self.encode0 = nn.Linear(self.num_inputs, 128)
        nn.init.xavier_uniform_(self.encode0.weight)
        self.bn0 = nn.BatchNorm1d(128)
        self.act0 = nn.LeakyReLU(0.2, inplace=True)

        self.encode1 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.encode2 = nn.Linear(64,1)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.act2 = final_activation

    def forward(self, x):
        x = self.act0(self.bn0(self.encode0(x)))
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.encode2(x))
        return x



class GeneratorATAC(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations2=nn.Sigmoid(),

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,
        self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )   #num_inputs=input_dim1+2
        #self.RNAencoder = RNAEncoderWithVariational(num_inputs=input_dim1, num_units=hidden_dim, hidden_channels=hidden_dim)
        #self.RNAdecoder = RNADecoder(num_outputs=out_dim1, num_units=hidden_dim,final_activation=final_activations1)
        #self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
        self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)
        self.spatial_fc = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Assuming spatial_coords has shape [batch_size, 2]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024)
        ) 

    #def forward(self, x, spatial_coords):
    def forward(self, x):
        #spatial_encoded = self.spatial_fc(spatial_coords)
        
        #x = torch.cat([x, spatial_coords], dim=-1)
        #print("2222")
        #print(x.shape)
        
        encoded = self.RNAencoder(x)
        decoded=self.ATACdecoder(encoded)
        return encoded,decoded


class GeneratorATAC_old(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations2=nn.Sigmoid(),

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,
        self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        #self.RNAdecoder = RNADecoder(num_outputs=out_dim1, num_units=hidden_dim,final_activation=final_activations1)
        #self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
        self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x):
        encoded = self.RNAencoder(x)
        decoded=self.ATACdecoder(encoded)
        return encoded, decoded
        

class GeneratorATAC_ii(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations2=nn.Sigmoid(),

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,
        self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        #self.RNAencoder = RNAEncoderWithVariational(num_inputs=input_dim1, num_units=hidden_dim, hidden_channels=hidden_dim)
        #self.RNAdecoder = RNADecoder(num_outputs=out_dim1, num_units=hidden_dim,final_activation=final_activations1)
        #self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
        self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x, edge_index):
        encoded = self.RNAencoder(x, edge_index)
        decoded=self.ATACdecoder(encoded)
        return encoded,decoded


class GeneratorProtein(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations2=nn.Identity(),

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,
        self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        self.ProteinDecoder = ProteinDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x):
        encoded = self.RNAencoder(x)
        decoded=self.ProteinDecoder(encoded)
        return encoded, decoded


class GeneratorRNA(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: list,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations1: list = [activations.Exp(), activations.ClippedSoftplus(),nn.Sigmoid()],

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,

        #self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        self.RNAdecoder = RNADecoder(num_outputs=input_dim1, num_units=hidden_dim, final_activation=final_activations1)
        #self.ATACencoder = SplitEnc(num_inputs=input_dim2, num_units=hidden_dim)  ####  num_inputs=input_dim2+2
        self.ATACencoder = SplitEnc(input_dim=input_dim2, z_dim=hidden_dim) 
        #self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)
        self.spatial_fc = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Assuming spatial_coords has shape [batch_size, 2]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1024)
        )


    #def forward(self, x, spatial_coords):
    def forward(self, x):
        #print(x.shape)
        #spatial_encoded = self.spatial_fc(spatial_coords)
        #print(spatial_encoded.shape)
        #rrr
        #spatial_features = self.spatial_encoder(spatial_coords)
        #x = torch.cat([x, spatial_coords], dim=-1)
        
        #print("dddd")
        #print(x.shape)
        encoded = self.ATACencoder(x)
        decoded=self.RNAdecoder(encoded)
        return encoded, decoded


class GeneratorRNA_old(nn.Module):
    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            #out_dim: List[int],
            hidden_dim: int = 16,
            final_activations1: list = [activations.Exp(), activations.ClippedSoftplus()],

            flat_mode: bool = True,  # Controls if we have to re-split inputs
            seed: int = 182822,
    ):
        # https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.input_dim1 = input_dim1,
        self.input_dim2 = input_dim2,

        #self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
        self.RNAdecoder = RNADecoder(num_outputs=input_dim1, num_units=hidden_dim,final_activation=final_activations1)
        self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
        #self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)


    def forward(self, x):
        encoded = self.ATACencoder(x)
        decoded=self.RNAdecoder(encoded)
        return encoded,decoded


class Discriminator(nn.Module):
    def __init__(self,input_dim: int,seed: int = 182822,):
        super(Discriminator, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16,1)
            #nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)

        return y




class Discriminator_old(nn.Module):
    def __init__(self,input_dim: int,seed: int = 182822,):
        super(Discriminator1, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.Linear(input_dim + 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16,1)
            #nn.Sigmoid(),
        )
    def forward(self, data, edge_index, spatial_coords=None):
        x = data  
       
        
        if spatial_coords is not None:
            x = torch.cat([x, spatial_coords], dim=1)
        y = self.model(x)
        return y
    
    #def forward(self, x):
    #    y = self.model(x)

    #    return y


class Discriminator1_OLD(nn.Module):
    def __init__(self, input_dim: int, spatial_dim: int = 2, hidden_dim: int = 256, seed: int = 182822):
        super(Discriminator1, self).__init__()
        torch.manual_seed(seed)
        
        self.spatial_dim = spatial_dim

        self.data_encoder = nn.Sequential(
            nn.Linear(input_dim + self.spatial_dim , hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.prelu = nn.PReLU()
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)  

        self.conv1 = GCNConv(hidden_dim, 32)
        self.conv2 = GCNConv(32, 16)

        self.output_layer = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
            # nn.Sigmoid() 
        )

    def forward(self, data, edge_index, spatial_coords=None):
        x = data  
        
        if spatial_coords is not None:
            x = torch.cat([x, spatial_coords], dim=1)

        data_features = self.data_encoder(x)

        x = self.conv1(data_features, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu(x)

        output = self.output_layer(x)

        return output
        

class Discriminator1(nn.Module):
    def __init__(self, input_dim: int, spatial_dim: int = 2, edge_dim: int = 256, seed: int = 182822):
        super(Discriminator1, self).__init__()
        torch.manual_seed(seed)
        
        self.spatial_dim = spatial_dim  
        self.edge_dim = edge_dim  # 

        # 
        self.graph_conv1 = GCNConv(input_dim+2, 512)  # 
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.graph_conv2 = GCNConv(input_dim + 2, 256)  # 

        self.spatial_fc = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1)
            # nn.Sigmoid() 
        )

    def forward(self, data, edge_index, spatial_coords=None):
        
        x = data  
        if spatial_coords is not None:
        
            x = torch.cat([x, spatial_coords], dim=1)
        
        x = self.graph_conv1(x, edge_index).relu()  #
        #x = torch.relu(x)  
     
        #x = self.graph_conv2(x, edge_index)
        #x = torch.relu(x)


        output = self.fc(x)

        return output



class DiscriminatorProtein(nn.Module):
    def __init__(self,input_dim: int,seed: int = 182822,):
        super(DiscriminatorProtein, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            # nn.Linear(input_dim, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,1)
            #nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)

        return y

# class GeneratorATAC(nn.Module):
#     def __init__(
#             self,
#             input_dim1: int,
#             input_dim2: int,
#             #out_dim: List[int],
#             hidden_dim: int = 16,
#             final_activations2=nn.Sigmoid(),
#
#             flat_mode: bool = True,  # Controls if we have to re-split inputs
#             seed: int = 182822,
#     ):
#         nn.Module.__init__(self)
#         torch.manual_seed(seed)  ##为CPU设置种子用于生成随机数，以使得结果是确定的
#
#         self.flat_mode = flat_mode
#         self.input_dim1 = input_dim1,
#         self.input_dim2 = input_dim2,
#         self.final_activations = final_activations2
#         self.RNAencoder = RNAEncoder(num_inputs=input_dim1, num_units=hidden_dim,  )
#         #self.RNAdecoder = RNADecoder(num_outputs=out_dim1, num_units=hidden_dim,final_activation=final_activations1)
#         #self.ATACencoder = ATACEncoder(num_inputs=input_dim2, num_units=hidden_dim)
#         self.ATACdecoder = ATACDecoder(num_outputs=input_dim2, num_units=hidden_dim,final_activation=final_activations2)
#         self.inference = Inference(num_inputs=input_dim1, final_activation=final_activations2)
#         self.region_factors = torch.nn.Parameter(torch.zeros(self.input_dim2))
#         nn.init.uniform_(self.region_factors)
#
#
#     def forward(self, x):
#         encoded = self.RNAencoder(x)
#         decoded=self.ATACdecoder(encoded)
#         final=decoded*self.inference(x)
#         final=final*self.final_activations(self.region_factors)
#         return final