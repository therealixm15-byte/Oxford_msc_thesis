# Code from dl4bi/benchmarks/meta_regression/

import random
from typing import List, Optional
import numpy as np
import pandas as pd
import jax.numpy as jnp
from collections import deque
import os
import requests
import tarfile
import matplotlib.pyplot as plt
import networkx as nx
import json
import imageio
import jax
import jraph
from flax import linen as nn
from node2vec import Node2Vec
import torch
from tqdm import tqdm
from dataclasses import dataclass
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


@dataclass
class SIRSimulation:
    sim_ids: List[int]
    betas: List[float]
    gammas: List[float]
    X_per_100k: np.array
    X: np.array
    y: np.array
    sigmas: List[Optional[float]] = None


def concat_simulations_temporal_dim(simulations):
    """
    Concatenate simulations along the temporal dimension.

    Args:
        simulations: List of SIR simulations.
            Each simulation is a dictionary with the following keys:
            - 'X': Array of shape (num_steps, L, n_regions).
            - 'X_per_100k': Array of shape (num_steps, n_regions).
            - 'y': Array of shape (num_steps, n_regions).
            - 'sim_id': Simulation ID.
            - 'beta': Beta parameter for the simulation.
            - 'gamma': Gamma parameter for the simulation.
            - 'sigma': Sigma parameter for the simulation (only if do_sirs is True).

    Returns:
        SIRSimulation object.
        - 'sim_ids': Array of shape (num_simulations * num_steps,).
        - 'betas': Array of shape (num_simulations,).
        - 'gammas': Array of shape (num_simulations,).
        - 'sigmas': Array of shape (num_simulations,).
        - 'X': Array of shape (num_simulations * num_steps, L, n_regions).
        - 'X_per_100k': Array of shape (num_simulations * num_steps, n_regions).
        - 'y': Array of shape (num_simulations * num_steps, n_regions).
    """

    Xs, Xs_100k, ys = [], [], []
    sim_ids, betas, gammas, sigmas = [], [], [], []

    for sim in simulations:
        Xs.append(sim['X'])
        Xs_100k.append(sim['X_per_100k'])
        ys.append(sim['y'])
        sim_ids.append([sim['sim_id']] * sim['y'].shape[0])
        betas.append(sim['beta'])
        gammas.append(sim['gamma'])
        sigmas.append(sim.get('sigma', None))

    # sanity check
    base_shape = Xs[0].shape[1:]
    assert all(x.shape[1:] == base_shape for x in Xs), "X shapes mismatch"

    # Y should always be 2-D with the same N counties
    assert all(y.ndim == 2 and y.shape[1] == base_shape[1] for y in ys), \
        "Y shapes mismatch"

    X = np.concatenate(Xs, axis=0) # Shape: (num_simulations * num_steps, L, n_regions)
    X = X.transpose(0, 2, 1)  # [timestep], [counties], [past_feats]
    X_100k = np.concatenate(Xs_100k, axis=0)
    y = np.concatenate(ys, axis=0) # Shape: (num_simulations * num_steps, n_regions)
    sim_ids = np.concatenate(sim_ids, axis=0) # Shape: (num_simulations * num_steps,)

    return SIRSimulation(
        sim_ids=sim_ids,
        betas=betas,
        gammas=gammas,
        sigmas=sigmas,
        X=X,
        X_per_100k=X_100k,
        y=y
    )



def make_static_undirected_unweighted_graph(
    edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a static, undirected, and unweighted version of the mobility graph, to run 
    experiment that compares gains of using more complicated input to the GNN surrogate
    """
    reverse_edges = edge_index[[1, 0], :]

    undirected_edge_index = np.concatenate(
        [edge_index, reverse_edges],
        axis=1,
    )

    undirected_edge_index = np.unique(
        undirected_edge_index,
        axis=1,
    )

    undirected_edge_weights = np.ones(
        undirected_edge_index.shape[1],
        dtype=np.float32,
    )

    return undirected_edge_index, undirected_edge_weights


def convert_simulations_to_dataset(
    sir_simulations, populations,
    edge_index, edge_attr, cfg
):
    """
    Convert SIR simulations to a dataset.

    Args:
        sir_simulations: List of SIR simulations.
        populations: Population array.
        edge_index: Edge index array.
        edge_attr: Edge attribute array.
        cfg: Configuration object.

    Returns:
        dataset_train_geom: Training dataset.
        sim_ids_train: Training simulation IDs.
        dataset_test_geom: Test dataset.
        sim_ids_test: Test simulation IDs.
        simulation_params: Simulation parameters.
    """
    if not sir_simulations:
        return

    simulation_params = []

    for sim_id, simulation in enumerate(sir_simulations):
        # simulation data to process
        values = np.stack(simulation['values']) # Shape: (T_eff, n_regions)
        shifted = np.vstack([np.zeros((1, values.shape[1])), values[:-1, :]])
        daily_diff = values - shifted  # DEBUG: values[1:] - values[:-1]
        incidence = daily_diff / populations[None, :]
        if cfg.dataset.output_binary:
            # Convert incidence to binary labels (1 if incidence > 0, i.e. the number of infected people increased across timesteps, else 0)
            incidence_per_100k = (incidence > 0).astype(float)
        else:
            incidence_per_100k = incidence * 1e5

        X_list = []
        y_list = []

        # Use actual simulated length
        T_eff = int(values.shape[0])
        L = int(cfg.dataset.num_lookback_steps)

        print(T_eff, L)

        # number of sliding windows: (x[0:L] -> y[L]), (x[1:L+1] -> y[L+1]), ...
        num_steps = max(0, T_eff - L)
        if num_steps == 0:
            continue
        
        for step in range(num_steps):
            X_list.append(incidence_per_100k[step:step+L, :])
            y_list.append(incidence_per_100k[step+L, :])

        simulation['sim_id'] = sim_id
        simulation['X'] = np.stack(X_list, axis=0) # Shape: (num_steps, L, n_regions)
        simulation['y'] = np.stack(y_list, axis=0) # Shape: (num_steps, n_regions)
        simulation['X_per_100k'] = incidence_per_100k[:-1]
        simulation_params.append({
            'sim_id': simulation['sim_id'],
            'beta': torch.tensor(simulation['beta']).to(device),
            'gamma': torch.tensor(simulation['gamma']).to(device),
            'sigma': torch.tensor(simulation['sigma']).to(device) if 'sigma' in simulation else None,
        })

    sim_indices = list(range(cfg.dataset.num_simulations))

    # shuffle indices (inplace!) in a reproducible way
    random.Random(cfg.data_seed).shuffle(sim_indices)

    num_sims_for_AL = cfg.dataset.num_sims_for_AL
    num_train_sims = int(
        cfg.dataset.train_test_ratio * 
        (cfg.dataset.num_simulations - num_sims_for_AL)
    )
    num_test_sims = int(
        cfg.dataset.num_simulations - num_train_sims - num_sims_for_AL
    )
    train_simulations = [
        sir_simulations[idx]
        for idx in sim_indices[:num_train_sims]
    ]
    test_simulations = [
        sir_simulations[idx]
        for idx in sim_indices[num_train_sims:num_train_sims + num_test_sims]
    ]
    AL_simulations = [
        sir_simulations[idx]
        for idx in sim_indices[num_train_sims + num_test_sims:]
    ]

    # concat along temporal dimension
    dataset_train = concat_simulations_temporal_dim(train_simulations)
    dataset_test = concat_simulations_temporal_dim(test_simulations)
    dataset_AL = concat_simulations_temporal_dim(AL_simulations)

    # normalize data only if the output is not binary
    if not cfg.dataset.output_binary:
        # std and mean over temporal dimension
        mean_x = np.mean(dataset_train.X, axis=0)
        std_x = np.std(dataset_train.X, axis=0) + 1e-6  # add eps to avoid /0
        mean_y = np.mean(dataset_train.y, axis=0)
        std_y = np.std(dataset_train.y, axis=0) + 1e-6  # add eps to avoid /0
        # input and target normalization
        dataset_train.X = (dataset_train.X - mean_x) / std_x
        dataset_test.X = (dataset_test.X - mean_x) / std_x
        dataset_AL.X = (dataset_AL.X - mean_x) / std_x
        dataset_train.y = (dataset_train.y - mean_y) / std_y
        dataset_test.y = (dataset_test.y - mean_y) / std_y
        dataset_AL.y = (dataset_AL.y - mean_y) / std_y

    # extract sim_ids (both training and test)
    sim_ids_train = dataset_train.sim_ids
    sim_ids_test = dataset_test.sim_ids
    sim_ids_AL = dataset_AL.sim_ids

    dataset_train_full = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_attr,
        features=dataset_train.X,
        targets=dataset_train.y
    )

    dataset_test_full = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_attr,
        features=dataset_test.X,
        targets=dataset_test.y
    )

    dataset_AL_full = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_attr,
        features=dataset_AL.X,
        targets=dataset_AL.y
    )

    # convert simulation_params to a dict
    simulation_params = {
        param['sim_id']: param
        for param in simulation_params
    }

    # Construct static graph lists matching each split of the full dataset
    (ablated_edge_index, ablated_edge_weights) = make_static_undirected_unweighted_graph(edge_index)

    dataset_train_ablated = StaticGraphTemporalSignal(
        edge_index=ablated_edge_index,
        edge_weight=ablated_edge_weights,
        features=dataset_train.X,
        targets=dataset_train.y,
    )

    dataset_test_ablated = StaticGraphTemporalSignal(
        edge_index=ablated_edge_index,
        edge_weight=ablated_edge_weights,
        features=dataset_test.X,
        targets=dataset_test.y,
    )

    dataset_AL_ablated = StaticGraphTemporalSignal(
        edge_index=ablated_edge_index,
        edge_weight=ablated_edge_weights,
        features=dataset_AL.X,
        targets=dataset_AL.y,
    )


    return (
        dataset_train_full,
        dataset_train_ablated,
        sim_ids_train,

        dataset_test_full,
        dataset_test_ablated,
        sim_ids_test,

        dataset_AL_full,
        dataset_AL_ablated,
        sim_ids_AL,
        
        simulation_params,
    )