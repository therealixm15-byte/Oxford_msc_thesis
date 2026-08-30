import os
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from tqdm import tqdm
import networkx as nx
# from src.surrogate_model import TemporalGNN
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
from typing import Dict, List, Optional, Tuple, Any

device = 'cuda' if torch.cuda.is_available() else \
    'mps' if torch.mps.is_available() else 'cpu'


# def visualise_individual_node(model, test_dataset, cfg, node_id):
#     model.eval()

#     y_preds = []
#     y_trues = []

#     for snapshot in test_dataset:
#         with torch.no_grad():
#             snapshot = snapshot.to(device)
#             if isinstance(model, TemporalGNN):
#                 x = snapshot.x.unsqueeze(0).unsqueeze(2)
#             else:
#                 x = snapshot.x
#             y_hat = model(x, snapshot.edge_index, snapshot.edge_attr)
#             y_preds.append(y_hat.squeeze())
#             y_trues.append(snapshot.y)

#     y_preds = torch.stack(y_preds).cpu()  # shape: [T, N]
#     y_trues = torch.stack(y_trues).cpu()  # shape: [T, N]

#     plt.figure(figsize=(12, 5))

#     plt.plot(y_trues[:, node_id].numpy(), label='Ground Truth', linewidth=2)
#     plt.plot(y_preds[:, node_id].numpy(), label='Prediction', linestyle='--')
#     plt.title(f"Forecast for Node {node_id}")
#     plt.xlabel("Time Step")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def visualise_sir_dataset(dataset):
    # graph = nx.read_adjlist(
    #     'data/mobility_matrix.csv',
    #     delimiter=','
    # )

    adjacency_matrix = np.genfromtxt(
        'data/mobility_matrix.csv',
        delimiter=','
    )

    vmax = max([
        max(data.y)
        for data in dataset
    ])

    vmin = min([
        min(data.y)
        for data in dataset
    ])

    np.fill_diagonal(adjacency_matrix, 0)
    rows, cols = np.where(adjacency_matrix != 0)
    edges = zip(
        rows.tolist(), cols.tolist(), [
            adjacency_matrix[r, c] for r, c in zip(rows, cols)
        ]
    )

    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)

    # generate node positions and visualise
    pos = nx.spring_layout(graph, seed=33)
    for step, data in tqdm(enumerate(dataset), total=dataset.targets.shape[0]):
        fig, ax = plt.subplots(figsize=(15, 15))
        edges, weights = zip(
            *nx.get_edge_attributes(graph, 'weight').items())
        nx.draw(graph, pos, node_color=data.y, edgelist=edges,
                edge_color="#A0CBE2", width=1.0, vmin=vmin, vmax=vmax)
        plt.title('Graph with Outbreaks at Step ' + str(step))
        plt.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax)), ax=ax)  # cmap=cmap,
        plt.savefig('sir_visualisations/' + 'graph_' + str(step) + '.png')
        mpl.pyplot.close()
    images = []
    for step in tqdm(range(dataset.targets.shape[0])):
        image_path = f'sir_visualisations/graph_{step}.png'
        if os.path.exists(image_path):
            images.append(imageio.imread(image_path))

    gif_path = 'sir_visualisations/' + 'graph.gif'
    for step in tqdm(range(dataset.targets.shape[0])):
        image_path = f'sir_visualisations/graph_{step}.png'
        if os.path.exists(image_path):
            os.remove(image_path)
    imageio.mimsave(gif_path, images, fps=1.3)
    print(f"GIF saved at {gif_path}")

    # print(gr)
    # nx.draw(gr, node_size=5)

    # print(graph.nodes())
    # print(nx.path_graph(4))
    # return
    # map node names to integers
    # mapping = {node: int(node) for node in graph.nodes()}
    # graph = nx.relabel_nodes(graph, mapping)
    # print(graph)
    # pass
    
def build_mobility_graph(theta: np.ndarray, threshold: float = 0.0, directed: bool = True) -> Any:
    """Build a graph from theta matrix with optional thresholding.

    Args:
        theta: Mobility matrix [N, N]. Entry i->j is weight from i to j.
        threshold: Drop edges with absolute weight <= threshold.
        directed: Whether to create a directed graph.

    Returns:
        A NetworkX graph with edge attribute 'weight'.
    """
    num_nodes = theta.shape[0]
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    rows, cols = np.where(np.abs(theta) > threshold)
    for i, j in zip(rows, cols):
        w = float(theta[i, j])
        if w == 0.0:
            continue
        G.add_edge(int(i), int(j), weight=w)
    return G


def _compute_layout(G: Any, layout: str = 'spring', seed: int = 42) -> Dict[int, Tuple[float, float]]:
    if layout == 'spring':
        return nx.spring_layout(G, seed=seed)
    if layout == 'kamada_kawai':
        return nx.kamada_kawai_layout(G)
    if layout == 'circular':
        return nx.circular_layout(G)
    return nx.spring_layout(G, seed=seed)


def _render_graph_frame(
    G: Any,
    node_values: np.ndarray,
    pos: Optional[Dict[int, Tuple[float, float]]] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'Reds',
    edge_alpha: float = 0.15
) -> np.ndarray:
    """Render a single frame and return as RGB array."""
    if pos is None:
        pos = _compute_layout(G)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    # Draw edges with alpha scaled by weight percentiles
    weights = np.array([d.get('weight', 0.0) for _, _, d in G.edges(data=True)])
    if len(weights) > 0:
        wmin = np.percentile(weights, 5)
        wmax = np.percentile(weights, 95)
        wspan = (wmax - wmin) if (wmax - wmin) > 1e-12 else 1.0
        edge_colors = [0.2 + 0.8 * ((d.get('weight', 0.0) - wmin) / wspan) for _, _, d in G.edges(data=True)]
    else:
        edge_colors = []

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors if edge_colors else 'gray',
        alpha=edge_alpha,
        arrows=G.is_directed(),
        ax=ax
    )

    # Draw nodes colored by current values
    vmin_eff = float(np.min(node_values)) if vmin is None else vmin
    vmax_eff = float(np.max(node_values)) if vmax is None else vmax
    nx.draw(
        G,
        pos=pos,
        node_color=node_values,
        edge_color='gray',
        cmap=cmap,
        vmin=vmin_eff,
        vmax=vmax_eff,
        node_size=500,
        with_labels=True,
        ax=ax
    )
    # Add a colorbar to indicate node value scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_eff, vmax=vmax_eff))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Node Value', rotation=270, labelpad=15)
    if title:
        ax.set_title(title)

    fig.canvas.draw()
    # Use backend-agnostic buffer access and drop alpha if present
    buf = np.asarray(fig.canvas.buffer_rgba())
    if buf.shape[-1] == 4:
        frame = buf[:, :, :3].copy()
    else:
        frame = buf.copy()
    plt.close(fig)
    return frame


def generate_outbreak_gif(
    theta_csv_path: str,
    values_time_series: np.ndarray,
    out_gif_path: str,
    layout: str = 'spring',
    threshold: float = 0.0,
    cmap: str = 'Reds'
) -> str:
    """Generate a GIF visualizing outbreaks over time on the mobility graph.

    Args:
        theta_csv_path: Path to theta CSV (N x N).
        values_time_series: [T, N] array of outbreak values (normalized or real).
        out_gif_path: Output path for GIF.
        layout: Graph layout: 'spring', 'kamada_kawai', or 'circular'.
        threshold: Drop edges with abs(weight) <= threshold.
        cmap: Matplotlib colormap name for node coloring.

    Returns:
        Path to the written GIF.
    """
    theta = np.loadtxt(theta_csv_path, delimiter=',')
    G = build_mobility_graph(theta, threshold=threshold, directed=True)
    if 'lattice' in theta_csv_path:
        print("Lattice graph detected")
        number_of_nodes = theta.shape[0]
        lattice_size = int(np.sqrt(number_of_nodes))
        pos = {i: (i % lattice_size, i // lattice_size) for i in range(number_of_nodes)}
        nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        plt.title("2D Lattice Graph Visualization")
        plt.show()
    else:
        pos = _compute_layout(G, layout=layout)

    values = np.asarray(values_time_series)
    # Normalize color scale across time for consistent legend
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    frames: List[np.ndarray] = []
    for t in range(values.shape[0]):
        title = f"Outbreaks at t={t}"
        frame = _render_graph_frame(
            G,
            node_values=values[t],
            pos=pos,
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )
        frames.append(frame)

    os.makedirs(os.path.dirname(out_gif_path), exist_ok=True)
    imageio.mimsave(out_gif_path, frames, duration=0.01)
    return out_gif_path


def generate_outbreak_gif_from_simulation(
    simulation_values: List[np.ndarray],
    theta_csv_path: str,
    out_gif_path: str,
    layout: str = 'spring',
    threshold: float = 0.0,
    cmap: str = 'Reds'
) -> str:
    """Generate a GIF from raw SIR simulation 'values' (infected counts per region).

    Args:
        simulation_values: List length T; each item is [N] infected counts (ints).
        theta_csv_path: Path to theta CSV (N x N).
        out_gif_path: Output GIF path.
        layout: Graph layout used for node positions.
        threshold: Drop edges with abs(weight) <= threshold.
        cmap: Matplotlib colormap name for node coloring.

    Returns:
        Path to the written GIF.
    """
    values_time_series = np.stack(simulation_values, axis=0)  # [T, N]

    return generate_outbreak_gif(
        theta_csv_path=theta_csv_path,
        values_time_series=values_time_series,
        out_gif_path=out_gif_path,
        layout=layout,
        threshold=threshold,
        cmap=cmap
    )


def generate_outbreak_gifs_for_multiple_simulations(
    simulations: List[dict],
    theta_csv_path: str,
    out_dir: str,
    layout: str = 'spring',
    threshold: float = 0.0,
    cmap: str = 'Reds'
) -> List[str]:
    """Batch-generate GIFs for a list of SIR simulations.

    Each item in 'simulations' is a dict with key 'values' as produced by the simulator.
    Returns the list of GIF file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    gif_paths: List[str] = []
    for idx, sim in enumerate(simulations):
        values = sim['values']
        gif_path = os.path.join(out_dir, f'simulation_{idx:03d}.gif')
        generate_outbreak_gif_from_simulation(
            simulation_values=values,
            theta_csv_path=theta_csv_path,
            out_gif_path=gif_path,
            layout=layout,
            threshold=threshold,
            cmap=cmap
        )
        gif_paths.append(gif_path)
    return gif_paths

