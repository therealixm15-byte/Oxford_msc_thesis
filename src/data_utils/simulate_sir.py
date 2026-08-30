# Code from dl4bi/benchmarks/meta_regression/cache/US_outbreaks/sir_simulation.py
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import numpy as np
import pandas as pd


def get_multiple_sir_sims(cfg):
    """
    Simulate SIR model for multiple simulations.

    Args:
        cfg: Configuration object.

    Returns:
        simulation_values: List of dictionaries with length num_simulations, each containing the following keys:
            - 'values': List of infection counts for each region at each time step.
                Length: T_eff, where T_eff is the effective number of time steps.
                Each element is an array of infection counts for each region at that time step, with shape (n_regions,).
            - 'beta': Beta parameter for the simulation.
            - 'gamma': Gamma parameter for the simulation.
            - 'sigma': Sigma parameter for the simulation (only if do_sirs is True).
        population: Population array.
        edge_index: Edge index array.
        edge_weight: Edge weight array.
    """
    np.random.seed(cfg.data_seed)
    simulation_values = []

    counties_data = pd.read_csv(
        cfg.dataset.dataset.population_csv_path,
        dtype={'FIPS': str}
    )
    population = counties_data['population'].values
    n_regions = len(population)
    print(f'Number of counties in US: {n_regions}')

    # load mobility network 
    theta_df = pd.read_csv(
        cfg.dataset.dataset.mobility_matrix_path,
        header=None,
    )

    theta = theta_df.values

    source_nodes, dest_nodes = np.nonzero(theta)

    edge_index = np.array(
        [source_nodes, dest_nodes]
    )

    edge_weight = theta[
        source_nodes,
        dest_nodes
    ]

    print(f"Graph created with {n_regions} nodes and {edge_index.shape[1]} sparse edges.")

        
    in_neighbours = {}
    out_neighbours = {}

    for i in range(n_regions):
        in_neighbours[i] = set(
            np.nonzero(theta[:, i])[0].tolist()
        )

        out_neighbours[i] = set(
            np.nonzero(theta[i, :])[0].tolist()
        )

    # Time-control
    dynamic_T = bool(getattr(cfg.dataset.SIR, "dynamic_T", False))
    max_T = int(getattr(cfg.dataset.SIR, "max_T", 500)) 
    min_T = int(getattr(cfg.dataset.SIR, "min_T", 50))
    L = int(cfg.dataset.num_lookback_steps)
    assert max_T >= min_T
    assert min_T > L + cfg.dataset.min_prediction_steps

    # Parameter range
    beta_range = (cfg.dataset.SIR.beta_lower, cfg.dataset.SIR.beta_upper)
    gamma_range = (cfg.dataset.SIR.gamma_lower, cfg.dataset.SIR.gamma_upper)
    do_sirs = bool(getattr(cfg.dataset, "do_sirs", False))
    if do_sirs:
        sigma_range = (cfg.dataset.SIR.sigma_lower, cfg.dataset.SIR.sigma_upper)

    # Simulate
    while len(simulation_values) < cfg.dataset.num_simulations:
        values = []

        beta = np.random.uniform(*beta_range)
        gamma = np.random.uniform(*gamma_range)
        sigma = np.random.uniform(*sigma_range) if do_sirs else None

        # Initial state at t=0
        S_t = population.astype(np.int64).copy()
        I_t = np.zeros(n_regions, dtype=np.int64)
        R_t = np.zeros(n_regions, dtype=np.int64)

        source = cfg.dataset.SIR.source
        if source == -1:
            source = np.random.randint(n_regions)
        I_t[source] = 1
        S_t[source] -= 1

        # Store t=0
        values.append(I_t.astype(np.int32))
        steps = 0
        while steps < max_T:
            if dynamic_T: 
                # run until no infected nodes remain or max_T
                if np.sum(I_t) == 0:
                    break
            # compute effective populations
            S_eff = np.zeros(n_regions)
            I_eff = np.zeros(n_regions)
            R_eff = np.zeros(n_regions)
            N_eff = np.zeros(n_regions)

            for i in range(n_regions):
                S_eff[i] = 0
                I_eff[i] = 0
                R_eff[i] = 0
                for l in in_neighbours[i]:
                    S_eff[i] += theta[l, i] * S_t[l]
                    I_eff[i] += theta[l, i] * I_t[l]
                    R_eff[i] += theta[l, i] * R_t[l]

                N_eff[i] = S_eff[i] + I_eff[i] + R_eff[i]

            # Calculate beta_eff for each region
            beta_eff = np.zeros(n_regions)
            for l in range(n_regions):
                if N_eff[l] > 0:
                    beta_eff[l] = (1 - np.exp(-1 * beta)) * (I_eff[l] / N_eff[l])
                else:
                    beta_eff[l] = 0

            # Update S, I, and R for each region
            S_t_next = S_t.copy()
            I_t_next = I_t.copy()
            R_t_next = R_t.copy()
            for i in range(n_regions):
                # infection probability from outgoing neighbours
                prob_infection = 0
                for l in out_neighbours[i]:
                    prob_infection += theta[i, l] * beta_eff[l]
                prob_infection = 1 - np.exp(-prob_infection)
                prob_infection = np.clip(prob_infection, 0.0, 1.0)

                new_infected = np.random.binomial(S_t[i], prob_infection) if prob_infection > 0 else 0
                new_recovered = np.random.binomial(I_t[i], gamma)
                new_susceptible = np.random.binomial(R_t[i], sigma) if do_sirs else 0

                S_t_next[i] = S_t[i] - new_infected + new_susceptible
                I_t_next[i] = I_t[i] + new_infected - new_recovered
                R_t_next[i] = R_t[i] + new_recovered - new_susceptible

                #if len(simulation_values) == 0:
                #    print(f"Simulation 0 - Infected people at timestep {steps} on node {i}: {I_t_next[i]}")

                # Clamp and conserve
                S_t_next[i] = max(0, int(S_t_next[i]))
                I_t_next[i] = max(0, int(I_t_next[i]))
                R_t_next[i] = max(0, int(R_t_next[i]))
                total = S_t_next[i] + I_t_next[i] + R_t_next[i]
                if total > population[i]:
                    diff = total - population[i]
                    S_t_next[i] -= diff
                    S_t_next[i] = max(0, S_t_next[i])

            # Advance
            S_t = S_t_next
            I_t = I_t_next
            R_t = R_t_next

            values.append(I_t.astype(np.int32))
            steps += 1

        # Enforce that steps must be at least min_T
        if steps < min_T:
            continue

        # Stash simulation results
        if do_sirs:
            simulation_values.append(
                {
                    'values': values,
                    'beta': beta,
                    'gamma': gamma,
                    'sigma': sigma
                }
            )
        else:
            simulation_values.append(
                {
                    'values': values,
                    'beta': beta,
                    'gamma': gamma
                }
            )

    return simulation_values, population, edge_index, edge_weight