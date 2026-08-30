from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
from torch_geometric_temporal import DCRNN
from torch_geometric_temporal.nn.recurrent import DCRNN, A3TGCN2, BatchedDCRNN
from torch_geometric.nn import GCNConv, EdgeConv
from torch_geometric.nn.pool.glob import global_mean_pool
from hydra.utils import get_class
import torch.nn.functional as F
from collections import deque
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np


class NodeEmbeder(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim: dimension of the embedding space
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.project = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        # learnable missing token
        self.missing_token = torch.nn.Parameter(
            torch.zeros(embed_dim), requires_grad=True
        )
        # initialize linear layer for skip connection
        self.linear_skip = torch.nn.Linear(
            1, embed_dim
        )

        # initialize the missing token
        torch.nn.init.normal_(
            self.missing_token, mean=0.0, std=0.2
        )

    def forward(self, x_raw, mask_hist):
        """
        Lift raw node features into a higher-dimensional representation. Replace missing
        values with a learnable mussing token. Concatenate observation status as an
        additional node feature.
        Args:
            x_raw: [N, L] raw node features
            mask_hist: [N, L]  binary mask (1=observed, 0=missing) for each node at each timestep
        Returns:
          x_in: [N, L, D+1] (embedded + mask channel) input for the GNN
        """
        # project per lag [N,L,1] -> [N,L,D]
        x_proj = self.project(x_raw.unsqueeze(-1)) + self.linear_skip(
            x_raw.unsqueeze(-1)
        )
        # [1,1,D], reshape missing token for broadcasting
        e = self.missing_token.view(
            1, 1, self.embed_dim
        )
        # compute input features by using raw values for observed timesteps and the missing token for unobserved timesteps
        x_tilde = mask_hist.unsqueeze(-1) * x_proj + \
            (1 - mask_hist).unsqueeze(-1) * e

        # add a mask channel to indicate which timesteps are observed and which are missing
        m_ch = mask_hist.unsqueeze(-1).float() # [N,L,1]

        return torch.cat([x_tilde, m_ch], dim=-1) # [N,L,D+1]
    


class ManageMasking(nn.Module):
    def __init__(self, L, embed_dim, probability_of_selecting_nodes, num_nodes, device):
        """
        Args:
            L: lookback window size 
            embed_dim: dimension of the embedding space
            probability_of_selecting_nodes: probability of selecting a node to be observed at each timestep
            num_nodes: number of nodes in the graph
        """
        super().__init__()
        self.embedder = NodeEmbeder(embed_dim)
        self.probability_of_selecting_nodes = probability_of_selecting_nodes
        self.L = L
        self.num_nodes = num_nodes
        self.device = device
        initial_temporal_masks = [
            (torch.rand(num_nodes) < probability_of_selecting_nodes).to(device).float()
            for _ in range(L)
        ]
        self.temporal_train_masks = deque(
            initial_temporal_masks,
            maxlen=L
        )
        self.temporal_test_masks = deque(
            initial_temporal_masks,
            maxlen=L
        )

    def reset_train_temporal_masks(self):
        """
        Reset the temporal masks to initial state.
        This should be done every time that we finish
        processing an individual SIR(S) simulation.
        """
        initial_temporal_masks = [
            (torch.rand(self.temporal_train_masks[0].shape) < self.probability_of_selecting_nodes).to(
                self.device).float()
            for _ in range(self.L)
        ]

        self.temporal_train_masks = deque(
            initial_temporal_masks, maxlen=self.L)

    def reset_test_temporal_masks(self):
        """
        Reset the test temporal masks to initial state.
        This should be done every time that we finish
        processing an individual SIR(S) simulation.
        """
        initial_temporal_masks = [
            (torch.rand(self.temporal_test_masks[0].shape) < self.probability_of_selecting_nodes).to(
                 self.device).float()
            for _ in range(self.L)
        ]

        self.temporal_test_masks = deque(
            initial_temporal_masks, maxlen=self.L)
        
    def make_test_mask_equal_to_train_mask(self):
        """
        Make the test temporal masks equal to the train temporal masks.
        This is useful for evaluating the model on the same nodes that were observed during training.
        """
        self.temporal_test_masks = deque(
            [mask.clone() for mask in self.temporal_train_masks], maxlen=self.L)

    def forward(self, x_raw, return_temporal_mask, external_mask=None):
        """
        Args:
            x_raw: [N, L] raw node features
            return_temporal_mask: whether to return the temporal mask used for this timestep
            external_mask: [N, L] external mask (1=observed, 0=missing) for each node at each timestep
        Returns:
            x_in: [N, L, D+1] (embedded + mask channel) input for the GNN
            activation_mask: [N] binary mask (1=observed, 0=missing) for each node at this timestep
        """
        # Determine activation mask for the newest timestep
        if external_mask is not None:
            activation_mask = external_mask.to(self.device).float()
        else:
            # Generate a new activation mask
            activation_mask = (
                    torch.rand(
                        x_raw.shape[0], device=self.device) < self.probability_of_selecting_nodes
                ).float()
        # ensure at least one node is active to avoid empty supervision
        if activation_mask.sum() == 0:
            rand_idx = torch.randint(
                0, x_raw.shape[0], (1,), device= self.device)
            activation_mask[rand_idx] = 1.0
        if self.training:
            # Append the activation mask to the training masks
            self.temporal_train_masks.append(activation_mask)
            mask_hist = torch.stack(list(self.temporal_train_masks),
                                    dim=1).to(self.device).float()
        else:
            # Append the activation mask to the test masks
            self.temporal_test_masks.append(activation_mask)
            mask_hist = torch.stack(list(self.temporal_test_masks),
                                    dim=1).to(self.device).float()
        if return_temporal_mask:
            return self.embedder(x_raw, mask_hist), activation_mask.bool()
        return self.embedder(x_raw, mask_hist)



