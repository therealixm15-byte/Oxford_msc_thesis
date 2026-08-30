import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import DCRNN

from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import random

#enable MC Dropout
def enable_mc_dropout(model):
  for m in model.modules():
    if isinstance(m, nn.Dropout):
      m.train()


def clone_hidden_states(hidden_states):
    """
    Copy recurrent hidden states so that MC-dropout passes do not modify
    the temporal state used by the main model.
    """
    if hidden_states is None:
        return None
    if torch.is_tensor(hidden_states):
        return hidden_states.detach().clone()
    return [
        hidden_state.detach().clone()
        for hidden_state in hidden_states
    ]


def sample_posterior_predictive(means, standard_deviations):
    """
    Draw one predictive sample from each approximate posterior predictive Gaussian produced by each
    MC-dropout forward pass.

    Args:
        means:
            Tensor of shape [num_mc_samples, num_nodes], representing the mean of the approximate posterior predictive Gaussian for each MC-dropout forward pass.
        standard_deviations:
            Tensor of shape [num_mc_samples, num_nodes], representing the standard deviation of the approximate posterior predictive Gaussian for each MC-dropout forward pass.

    Returns:
        predictive_samples:
            Tensor of shape [num_mc_samples, num_nodes].
    """
    standard_deviations = standard_deviations.clamp_min(1e-4)
    predictive_samples = (
        means
        + standard_deviations * torch.randn_like(means)
    )
    return predictive_samples


def Random(model, snapshot, hidden_states, AL_mask, num_nodes_to_select, cfg, device):
    """
    Randomly select a number of nodes, determined by a fixed budget, to be labeled and added to the training set.
    """
    model.eval()
    selected_nodes = []
    num_nodes = cfg.model.model.num_nodes
    for _ in range(cfg.training.active_learning.num_nodes_to_test):
        node_id = random.randint(0, num_nodes - 1)
        selected_nodes.append(node_id)
    model.train()
    return selected_nodes


def Random_Variable(model, snapshot, hidden_states, AL_mask, num_nodes_to_select, cfg, device):
    """
    Randomly select a number of nodes, determined by Variable batch allocation, to be labeled and added to the training set.
    """
    model.eval()
    selected_nodes = []
    num_nodes = cfg.model.model.num_nodes
    for _ in range(num_nodes_to_select):
        node_id = random.randint(0, num_nodes - 1)
        selected_nodes.append(node_id)
    model.train()
    return selected_nodes



# def MC_Dropout_Variance(model, snapshot, hidden_states, AL_mask, num_nodes_to_select, cfg, device):
# #def MC_Dropout_Variance(model, snapshot, hidden_states, AL_mask, cfg, device):
#     """
#     Select nodes whose point predictions disagree most across MC-dropout
#     forward passes.

#     Returns:
#         selected_nodes: a list of node indices that have been selected by the policy
#         mc_predictions: a tensor containing the MC-dropout predictions for each node (used for computing metrics of quality of quantified uncertainty)
#     """
#     model.eval()
#     enable_mc_dropout(model)
#     mc_predictions = []
#     standard_deviations = []
#     with torch.no_grad():
#         for _ in range(cfg.training.active_learning.mc_dropout_samples):
#             hidden_states_copy = clone_hidden_states(hidden_states)

#             prediction, std, _, _ = model(
#                 snapshot.x,
#                 snapshot.edge_index,
#                 snapshot.edge_attr,
#                 hidden_states_copy,
#                 external_mask=AL_mask,
#             )

#             mc_predictions.append(prediction.reshape(-1))
#             standard_deviations.append(std.reshape(-1))

#         mc_predictions = torch.stack(mc_predictions, dim=0) # [num_mc_samples, num_nodes]
#         standard_deviations = torch.stack(standard_deviations, dim=0)

#         scores = mc_predictions.var(
#             dim=0,
#             unbiased=False,
#         )
#         selected_nodes = torch.topk(scores, k=cfg.training.active_learning.num_nodes_to_test).indices
#     predictive_samples = sample_posterior_predictive(means=mc_predictions,standard_deviations=standard_deviations,) #[num_mc_samples, num_nodes]
#     return selected_nodes, predictive_samples


def Degree_Centrality(model, snapshot, hidden_states, AL_mask, num_nodes_to_select, cfg, device):
    """
    Select nodes with the highest total degree centrality.
    Select a fixed budget of nodes.
    For an undirected graph, this computes the number of incident edges.
    For a directed graph, this computes:

        total degree = in-degree + out-degree
    """
    model.eval()
    num_nodes = snapshot.x.size(0)
    edge_index = snapshot.edge_index.to(device)
    out_degree = torch.bincount(
        edge_index[0],
        minlength=num_nodes,
    )
    in_degree = torch.bincount(
        edge_index[1],
        minlength=num_nodes,
    )
    total_degree = (in_degree + out_degree).float()
    selected_nodes = torch.topk(
        total_degree,
        k=cfg.training.active_learning.num_nodes_to_test,
    ).indices
    model.train()
    return selected_nodes


def Degree_Centrality_Variable(model, snapshot, hidden_states, AL_mask, num_nodes_to_select, cfg, device):
    """
    Select nodes with the highest total degree centrality.
    Select a number of nodes determined by Variable batch allocation.
    For an undirected graph, this computes the number of incident edges.
    For a directed graph, this computes:

        total degree = in-degree + out-degree
    """
    model.eval()
    num_nodes = snapshot.x.size(0)
    edge_index = snapshot.edge_index.to(device)
    out_degree = torch.bincount(
        edge_index[0],
        minlength=num_nodes,
    )
    in_degree = torch.bincount(
        edge_index[1],
        minlength=num_nodes,
    )
    total_degree = (in_degree + out_degree).float()
    selected_nodes = torch.topk(
        total_degree,
        k=num_nodes_to_select,
    ).indices
    model.train()
    return selected_nodes



def BALSA_KL_Pairs(model, snapshot, hidden_states, AL_mask, num_nodes_to_select, cfg, device):
    """
    BALSA-KL-Pairs using exact KL divergence between Gaussian predictive
    distributions.
    Each MC-dropout forward pass produces one Gaussian predictive
    distribution per node:
        p_t(y_i | x_i) = Normal(mean_i_t, std_i_t)
    The acquisition score is the sum of KL disagreements between consecutive
    MC-dropout trials.
    Select a fixed budget of nodes.

    Returns:
        selected_nodes: a list of node indices that have been selected by the policy
        means: a tensor containing the MC-dropout predictions for each node (used for computing metrics of quality of quantified uncertainty)
    """
    model.eval()
    enable_mc_dropout(model)

    num_mc_samples = (
        cfg.training.active_learning.mc_dropout_samples
    )
    num_nodes = snapshot.x.size(0)
    means = torch.empty(
        num_mc_samples,
        num_nodes,
        device=device,
    )
    standard_deviations = torch.empty(
        num_mc_samples,
        num_nodes,
        device=device,
    )
    with torch.no_grad():
        for trial in range(num_mc_samples):
            hidden_states_copy = clone_hidden_states(hidden_states)
            mean, std, _, _ = model(
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr,
                hidden_states_copy,
                external_mask=AL_mask,
            )
            means[trial] = mean.view(-1)
            standard_deviations[trial] = std.view(-1)
        balsa_scores = torch.zeros(
            num_nodes,
            device=device,
        )
        for trial in range(num_mc_samples - 1):
            mean_p = means[trial]
            mean_q = means[trial + 1]
            std_p = standard_deviations[trial]
            std_q = standard_deviations[trial + 1]
            pairwise_kl = (
                torch.log(std_q / std_p)
                + (
                    std_p.square()
                    + (mean_p - mean_q).square()
                ) / (2.0 * std_q.square())
                - 0.5
            )
            balsa_scores += pairwise_kl
        selected_nodes = torch.topk(
            balsa_scores,
            #k=num_nodes_to_select
            k = cfg.training.active_learning.num_nodes_to_test,
        ).indices
    predictive_samples = sample_posterior_predictive(means,standard_deviations=standard_deviations,)
    model.train()
    return selected_nodes, predictive_samples


def BALSA_KL_Pairs_Variable(model, snapshot, hidden_states, AL_mask, num_nodes_to_select, cfg, device):
    """
    BALSA-KL-Pairs using exact KL divergence between Gaussian predictive
    distributions.
    Each MC-dropout forward pass produces one Gaussian predictive
    distribution per node:
        p_t(y_i | x_i) = Normal(mean_i_t, std_i_t)
    The acquisition score is the sum of KL disagreements between consecutive
    MC-dropout trials.
    Select a number of nodes determined by Variable batch allocation.

    Returns:
        selected_nodes: a list of node indices that have been selected by the policy
        means: a tensor containing the MC-dropout predictions for each node (used for computing metrics of quality of quantified uncertainty)
    """
    model.eval()
    enable_mc_dropout(model)

    num_mc_samples = (
        cfg.training.active_learning.mc_dropout_samples
    )
    num_nodes = snapshot.x.size(0)
    means = torch.empty(
        num_mc_samples,
        num_nodes,
        device=device,
    )
    standard_deviations = torch.empty(
        num_mc_samples,
        num_nodes,
        device=device,
    )
    with torch.no_grad():
        for trial in range(num_mc_samples):
            hidden_states_copy = clone_hidden_states(hidden_states)
            mean, std, _, _ = model(
                snapshot.x,
                snapshot.edge_index,
                snapshot.edge_attr,
                hidden_states_copy,
                external_mask=AL_mask,
            )
            means[trial] = mean.view(-1)
            standard_deviations[trial] = std.view(-1)
        balsa_scores = torch.zeros(
            num_nodes,
            device=device,
        )
        for trial in range(num_mc_samples - 1):
            mean_p = means[trial]
            mean_q = means[trial + 1]
            std_p = standard_deviations[trial]
            std_q = standard_deviations[trial + 1]
            pairwise_kl = (
                torch.log(std_q / std_p)
                + (
                    std_p.square()
                    + (mean_p - mean_q).square()
                ) / (2.0 * std_q.square())
                - 0.5
            )
            balsa_scores += pairwise_kl
        selected_nodes = torch.topk(
            balsa_scores,
            #k=num_nodes_to_select
            k = num_nodes_to_select,
        ).indices
    predictive_samples = sample_posterior_predictive(means,standard_deviations=standard_deviations,)
    model.train()
    return selected_nodes, predictive_samples


def True_MSE_Oracle(
    model,
    snapshot,
    hidden_states,
    AL_mask,
    num_nodes_to_select,
    cfg,
    device,
):
    """
    Oracle acquisition policy that selects the nodes with the largest
    true prediction squared error.
    Select a fixed budget of nodes.

    This oracle deliberately uses the ground-truth labels and is
    intended only as a diagnostic upper bound.

    Returns:
        selected_nodes:
            Tensor of selected node indices.

        diagnostics:
            Dictionary containing per-node squared errors and predictions.
    """

    num_nodes = snapshot.y.numel()
    model.eval()

    with torch.no_grad():
        hidden_states_copy = clone_hidden_states(hidden_states)
        predicted_mean, predicted_std, _, _ = model(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_attr,
            hidden_states_copy,
            external_mask=AL_mask,
        )

        predicted_mean = predicted_mean.view(-1)
        predicted_std = predicted_std.view(-1)
        y_true = snapshot.y.view(-1)

        # Per-node squared error
        node_mse = (
            predicted_mean - y_true
        ).square()

        node_mse = torch.nan_to_num(
            node_mse,
            nan=-torch.inf,
            posinf=torch.finfo(node_mse.dtype).max,
            neginf=-torch.inf,
        )

        selected_nodes = torch.topk(
            node_mse,
            k=cfg.training.active_learning.num_nodes_to_test,
            largest=True,
        ).indices

    model.train()
    return selected_nodes


def True_MSE_Oracle_Variable(
    model,
    snapshot,
    hidden_states,
    AL_mask,
    num_nodes_to_select,
    cfg,
    device,
):
    """
    Oracle acquisition policy that selects the nodes with the largest
    true prediction squared error.
    Select a number of nodes determined by Variable batch allocation.

    This oracle deliberately uses the ground-truth labels and is
    intended only as a diagnostic upper bound.

    Returns:
        selected_nodes:
            Tensor of selected node indices.

        diagnostics:
            Dictionary containing per-node squared errors and predictions.
    """

    num_nodes = snapshot.y.numel()
    model.eval()

    with torch.no_grad():
        hidden_states_copy = clone_hidden_states(hidden_states)
        predicted_mean, predicted_std, _, _ = model(
            snapshot.x,
            snapshot.edge_index,
            snapshot.edge_attr,
            hidden_states_copy,
            external_mask=AL_mask,
        )

        predicted_mean = predicted_mean.view(-1)
        predicted_std = predicted_std.view(-1)
        y_true = snapshot.y.view(-1)

        # Per-node squared error
        node_mse = (
            predicted_mean - y_true
        ).square()

        node_mse = torch.nan_to_num(
            node_mse,
            nan=-torch.inf,
            posinf=torch.finfo(node_mse.dtype).max,
            neginf=-torch.inf,
        )

        selected_nodes = torch.topk(
            node_mse,
            k=num_nodes_to_select,
            largest=True,
        ).indices

    model.train()
    return selected_nodes


def Predicted_Incidence(
    model,
    snapshot,
    hidden_states,
    AL_mask,
    num_nodes_to_select,
    cfg,
    device,
    predicted_mean,
    gamma=0.7,
    delta=1.0,
):
    """
    Choose nodes using:
        Predicted incidence magnitude
        Mobility-weighted mean neighbour incidence magnitude
    Select a fixed budget of nodes.
    Score:
        score_i =
            gamma * normalized_predicted_incidence
            + delta * normalized_weighted_neighbour_incidence
    """
    num_nodes = snapshot.x.shape[0]
    predicted_mean = predicted_mean.detach()
    model.eval()

    # # ---------------------------------------------------------
    # # Graph information
    # # ---------------------------------------------------------
    edge_index = snapshot.edge_index
    edge_weight = snapshot.edge_attr.view(-1)

    source = edge_index[0]
    destination = edge_index[1]

    # Remove self-loops for ALL neighbour-based features
    non_self_edges = source != destination

    edge_source = source[non_self_edges]
    edge_destination = destination[non_self_edges]
    edge_weight_neigh = edge_weight[non_self_edges]

    # ---------------------------------------------------------
    # node predicted incidence magnitude
    # ---------------------------------------------------------

    predicted_incidence_magnitude = (
        predicted_mean.abs()
    )

    # ---------------------------------------------------------
    # mobility-weighted neighbour incidence magnitude
    # ---------------------------------------------------------
    node_incidence = predicted_incidence_magnitude

    # ---------- Incoming incidence ----------
    incoming_incidence_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    incoming_incidence_weight_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    incoming_incidence_sum.index_add_(
        0,
        edge_destination,
        edge_weight_neigh
        * node_incidence[edge_source],
    )

    incoming_incidence_weight_sum.index_add_(
        0,
        edge_destination,
        edge_weight_neigh,
    )

    incoming_mean_incidence = (
        incoming_incidence_sum
        / incoming_incidence_weight_sum.clamp_min(1e-8)
    )

    # ---------- Outgoing incidence ----------
    outgoing_incidence_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    outgoing_incidence_weight_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    outgoing_incidence_sum.index_add_(
        0,
        edge_source,
        edge_weight_neigh
        * node_incidence[edge_destination],
    )

    outgoing_incidence_weight_sum.index_add_(
        0,
        edge_source,
        edge_weight_neigh,
    )

    outgoing_mean_incidence = (
        outgoing_incidence_sum
        / outgoing_incidence_weight_sum.clamp_min(1e-8)
    )

    # Equal contribution from incoming and outgoing neighbourhoods
    weighted_neighbour_incidence = (
        0.5 * incoming_mean_incidence
        + 0.5 * outgoing_mean_incidence
        #0 * incoming_mean_incidence
        #+ 1.0 * outgoing_mean_incidence
    )

    # ---------------------------------------------------------
    # Normalize features across nodes
    # ---------------------------------------------------------
    def min_max_normalize(x):
        return (
            (x - x.min())
            / (x.max() - x.min() + 1e-8)
        )

    normalized_predicted_incidence = (
        min_max_normalize(
            predicted_incidence_magnitude
        )
    )

    normalized_weighted_neighbour_incidence = (
        min_max_normalize(
            weighted_neighbour_incidence
        )
    )

    # ---------------------------------------------------------
    # Final acquisition score
    # ---------------------------------------------------------

    acquisition_score = (
        + gamma
        * normalized_predicted_incidence

        + delta
        * normalized_weighted_neighbour_incidence
    )

    # ---------------------------------------------------------
    # Select highest-scoring nodes
    # ---------------------------------------------------------
    selected_nodes = torch.topk(
        acquisition_score,
        k=cfg.training.active_learning.num_nodes_to_test,
        largest=True,
    ).indices

    model.train()
    return selected_nodes




def Predicted_Incidence_Variable(
    model,
    snapshot,
    hidden_states,
    AL_mask,
    num_nodes_to_select,
    cfg,
    device,
    predicted_mean,
    gamma=0.7,
    delta=1.0,
):
    """
        Choose nodes using:
            Predicted incidence magnitude
            Mobility-weighted mean neighbour incidence magnitude
        Select a number of nodes determined by Variable batch allocation.
        Score:
            score_i =
                gamma * normalized_predicted_incidence
                + delta * normalized_weighted_neighbour_incidence
        """
    num_nodes = snapshot.x.shape[0]
    predicted_mean = predicted_mean.detach()
    model.eval()
    # # ---------------------------------------------------------
    # # Graph information
    # # ---------------------------------------------------------
    edge_index = snapshot.edge_index
    edge_weight = snapshot.edge_attr.view(-1)

    source = edge_index[0]
    destination = edge_index[1]

    # Remove self-loops for ALL neighbour-based features
    non_self_edges = source != destination

    edge_source = source[non_self_edges]
    edge_destination = destination[non_self_edges]
    edge_weight_neigh = edge_weight[non_self_edges]

    # ---------------------------------------------------------
    # node predicted incidence magnitude
    # ---------------------------------------------------------
    predicted_incidence_magnitude = (
        predicted_mean.abs()
    )

    # ---------------------------------------------------------
    # mobility-weighted neighbour incidence magnitude
    # ---------------------------------------------------------
    node_incidence = predicted_incidence_magnitude

    # ---------- Incoming incidence ----------
    incoming_incidence_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    incoming_incidence_weight_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    incoming_incidence_sum.index_add_(
        0,
        edge_destination,
        edge_weight_neigh
        * node_incidence[edge_source],
    )

    incoming_incidence_weight_sum.index_add_(
        0,
        edge_destination,
        edge_weight_neigh,
    )

    incoming_mean_incidence = (
        incoming_incidence_sum
        / incoming_incidence_weight_sum.clamp_min(1e-8)
    )

    # ---------- Outgoing incidence ----------
    outgoing_incidence_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    outgoing_incidence_weight_sum = torch.zeros(
        num_nodes,
        device=device,
    )

    outgoing_incidence_sum.index_add_(
        0,
        edge_source,
        edge_weight_neigh
        * node_incidence[edge_destination],
    )

    outgoing_incidence_weight_sum.index_add_(
        0,
        edge_source,
        edge_weight_neigh,
    )

    outgoing_mean_incidence = (
        outgoing_incidence_sum
        / outgoing_incidence_weight_sum.clamp_min(1e-8)
    )

    # Equal contribution from incoming and outgoing neighbourhoods
    weighted_neighbour_incidence = (
        0.5 * incoming_mean_incidence
        + 0.5 * outgoing_mean_incidence
    )

    # ---------------------------------------------------------
    # Normalize features across nodes
    # ---------------------------------------------------------
    def min_max_normalize(x):
        return (
            (x - x.min())
            / (x.max() - x.min() + 1e-8)
        )

    normalized_predicted_incidence = (
        min_max_normalize(
            predicted_incidence_magnitude
        )
    )

    normalized_weighted_neighbour_incidence = (
        min_max_normalize(
            weighted_neighbour_incidence
        )
    )

    # ---------------------------------------------------------
    # Final acquisition score
    # ---------------------------------------------------------
    acquisition_score = (
        + gamma
        * normalized_predicted_incidence

        + delta
        * normalized_weighted_neighbour_incidence
    )

    # ---------------------------------------------------------
    # Select highest-scoring nodes
    # ---------------------------------------------------------
    selected_nodes = torch.topk(
        acquisition_score,
        k=num_nodes_to_select,
        largest=True,
    ).indices

    model.train()
    return selected_nodes


