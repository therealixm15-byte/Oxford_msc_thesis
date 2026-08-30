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
from src.temporal_masking import ManageMasking
    

class GRU_GCN_Layer(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, gcn_hidden_dim, dropout):
        """
        Implement a layer of GRU + GCN       
        """
        # For GCN_GRU layer
        # super().__init__()
        # self.gru_cell = nn.GRUCell(
        #     input_size=gcn_hidden_dim,
        #     hidden_size=gru_hidden_dim
        # )
        # self.gcn = GCNConv(
        #     in_channels=input_dim,
        #     out_channels=gcn_hidden_dim
        # )
        # self.project_h = nn.Linear(gru_hidden_dim, gcn_hidden_dim)
        # self.act = nn.ReLU()
        # self.dropout_before_gcn = nn.Dropout(dropout)
        # self.dropout_after_gcn = nn.Dropout(dropout)

        super().__init__()
        self.gru_cell = nn.GRUCell(
            input_size=input_dim,
            hidden_size=gru_hidden_dim
        )
        self.gcn = GCNConv(
            in_channels=gru_hidden_dim,
            out_channels=gcn_hidden_dim
        )
        self.project_h = nn.Linear(gcn_hidden_dim, gru_hidden_dim)
        self.act = nn.ReLU()
        self.dropout_before_gcn = nn.Dropout(dropout)
        self.dropout_after_gcn = nn.Dropout(dropout)

    # For GCN_GRU layer
    # def forward(self, x_t, h, edge_index, edge_weight=None):
    #     """
    #     Args:
    #         x_t: [num_nodes, input_dim]
    #         h: [num_nodes, gru_hidden_dim]
    #         edge_index: [2, num_edges]
    #         edge_weight: [num_edges]
    #     Returns:
    #         h_next: [num_nodes, gru_hidden_dim], the updated hidden state to be used as the previous hidden state at the next timestep
    #     """
    #     # 1. Spatial processing with the GCN
    #     gcn_input = self.dropout_before_gcn(x_t)
    #     h_gcn = self.gcn(
    #         gcn_input,
    #         edge_index,
    #         edge_weight
    #     )  # [num_nodes, gcn_hidden_dim]
    #     h_gcn = self.dropout_after_gcn(h_gcn)
    #     # 2. Temporal processing with the GRU
    #     h_gru = self.gru_cell(
    #         h_gcn,
    #         h
    #     )  # [num_nodes, gru_hidden_dim]
    #     h_next = self.project_h(h_gru)  # [num_nodes, gcn_hidden_dim]
    #     return h_next

    def forward(self, x_t, h, edge_index, edge_weight=None):
        """
        Args:
            x_t: [num_nodes, input_dim]
            h: [num_nodes, gru_hidden_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            h_next: [num_nodes, gru_hidden_dim], the updated hidden state to be used as the previous hidden state at the next timestep
        """
        h_gru = self.gru_cell(x_t, h) # [num_nodes, gru_hidden_dim]
        gcn_input = self.dropout_before_gcn(h_gru)
        h_gcn = self.gcn(gcn_input, edge_index, edge_weight) # [num_nodes, gcn_hidden_dim]
        h_gcn = self.act(h_gcn)
        h_gcn = self.dropout_after_gcn(h_gcn)
        h_next = self.project_h(h_gcn) # [num_nodes, gru_hidden_dim]
        return h_next


class GRU_GCN_Model(nn.Module):
    def __init__(self, input_dim, embed_dim, gru_hidden_dim, gcn_hidden_dim, 
                 output_dim, probability_of_selecting_nodes, num_nodes,
                 return_temporal_mask, device, dropout, num_layers=1): 
        """
        Stack multiple layers of GRU_GCN_Layer followed by dropout and a final prediction head 
        """      
        super().__init__()
        # the input_dim of the raw input is L, the lookback window size
        self.embed = ManageMasking(input_dim, embed_dim, probability_of_selecting_nodes, num_nodes, device)
        self.gru_hidden_dim = gru_hidden_dim
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.device = device
        self.return_temporal_mask = return_temporal_mask
        self.layer_list = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layer_list.append(GRU_GCN_Layer(input_dim * (embed_dim + 1), gru_hidden_dim, gcn_hidden_dim, dropout))
            else:
                self.layer_list.append(GRU_GCN_Layer(gru_hidden_dim, gru_hidden_dim, gcn_hidden_dim, dropout))
        self.dropout = nn.Dropout(dropout)
        self.prediction_head = nn.Linear(gru_hidden_dim, output_dim)
    
    def forward(self, x_t, edge_index, edge_weight, hidden_states, external_mask=None): 
        """
        Args:
            x_t: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            hidden_states: [num_layers, num_nodes, gru_hidden_dim]
            external_mask: [num_nodes, input_dim] (optional)
        Returns:
            y_pred: [num_nodes, output_dim]
            hidden_states: [num_nodes, gru_hidden_dim], a list of hidden states, one for each layer
        """
        if hidden_states is None:
            hidden_states = [torch.zeros(self.num_nodes, self.gru_hidden_dim, device=self.device) for _ in range(self.num_layers)]
        if self.return_temporal_mask:
            previous_layer, temporal_mask = self.embed(x_t, self.return_temporal_mask, external_mask)
        else:
            previous_layer = self.embed(x_t, self.return_temporal_mask, external_mask)
        # reshape x_t from [num_nodes, L, D+1] to [num_nodes, L*(D+1)] to feed into the GRU_GCN_Layer
        if previous_layer.dim() == 3:
            previous_layer = previous_layer.reshape(previous_layer.size(0), -1)
        for layer_idx, layer in enumerate(self.layer_list):
            hidden_states[layer_idx] = layer(
                previous_layer, hidden_states[layer_idx], edge_index, edge_weight
                )
            previous_layer = self.dropout(hidden_states[layer_idx])
        final_hidden = self.dropout(hidden_states[-1])
        y_pred = self.prediction_head(final_hidden)
        y_pred_mean = y_pred[:, 0]
        y_pred_std = F.softplus(y_pred[:, 1]) + 1e-6
        if self.return_temporal_mask:
            return y_pred_mean, y_pred_std, hidden_states, temporal_mask
        #h = hidden_states[-1]
        return y_pred_mean, y_pred_std, hidden_states


