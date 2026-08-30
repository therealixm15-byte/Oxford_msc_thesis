from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.temporal_masking import ManageMasking


class GCN_Layer(nn.Module):
    def __init__(
        self,
        input_dim,
        gcn_hidden_dim,
        dropout,
    ):
        """
        Implement a GCN layer with dropout and ReLU activation.
        """
        super().__init__()
        self.gcn = GCNConv(
            in_channels=input_dim,
            out_channels=gcn_hidden_dim,
        )
        self.act = nn.ReLU()
        self.dropout_before_gcn = nn.Dropout(
            dropout
        )
        self.dropout_after_gcn = nn.Dropout(
            dropout
        )
    def forward(
        self,
        x_t,
        edge_index,
        edge_weight=None,
    ):
        gcn_input = self.dropout_before_gcn(
            x_t
        )
        h_gcn = self.gcn(
            gcn_input,
            edge_index,
            edge_weight,
        )
        h_gcn = self.act(
            h_gcn
        )
        h_gcn = self.dropout_after_gcn(
            h_gcn
        )
        return h_gcn


class GCN_Model(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        gru_hidden_dim,
        gcn_hidden_dim,
        output_dim,
        probability_of_selecting_nodes,
        num_nodes,
        return_temporal_mask,
        device,
        dropout,
        num_layers=1,
    ):
        """
        GCN-only ablation of the GRU-GCN model.
        """
        super().__init__()
        self.embed = ManageMasking(
            input_dim,
            embed_dim,
            probability_of_selecting_nodes,
            num_nodes,
            device,
        )

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.device = device
        self.return_temporal_mask = (
            return_temporal_mask
        )
        self.gru_hidden_dim = gru_hidden_dim
        self.layer_list = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = (
                    input_dim
                    * (embed_dim + 1)
                )
            else:

                layer_input_dim = (
                    gcn_hidden_dim
                )
            self.layer_list.append(
                GCN_Layer(
                    input_dim=layer_input_dim,
                    gcn_hidden_dim=gcn_hidden_dim,
                    dropout=dropout,
                )
            )
        self.dropout = nn.Dropout(
            dropout
        )

        self.prediction_head = nn.Linear(
            gcn_hidden_dim,
            output_dim,
        )

    def forward(
        self,
        x_t,
        edge_index,
        edge_weight,
        hidden_states=None,
        external_mask=None,
    ):
        if self.return_temporal_mask:

            (
                previous_layer,
                temporal_mask,
            ) = self.embed(
                x_t,
                self.return_temporal_mask,
                external_mask,
            )
        else:

            previous_layer = self.embed(
                x_t,
                self.return_temporal_mask,
                external_mask,
            )
        if previous_layer.dim() == 3:

            previous_layer = (
                previous_layer.reshape(
                    previous_layer.size(0),
                    -1,
                )
            )
        layer_states = []

        for layer in self.layer_list:

            previous_layer = layer(
                previous_layer,
                edge_index,
                edge_weight,
            )

            previous_layer = self.dropout(
                previous_layer
            )
            layer_states.append(
                previous_layer
            )
        final_hidden = self.dropout(
            previous_layer
        )

        y_pred = self.prediction_head(
            final_hidden
        )

        y_pred_mean = y_pred[:, 0]

        y_pred_std = (
            F.softplus(
                y_pred[:, 1]
            )
            + 1e-6
        )
        if self.return_temporal_mask:

            return (
                y_pred_mean,
                y_pred_std,
                layer_states,
                temporal_mask,
            )

        return (
            y_pred_mean,
            y_pred_std,
            layer_states,
        )