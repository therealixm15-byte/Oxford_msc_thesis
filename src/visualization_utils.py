import torch
import wandb
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from plotly.express import colors as px_colors


POLICY_COLORS = {
    "No_Fine_Tuning": "#636EFA",
    "BALSA_KL_Pairs": "#EF553B",
    "Predicted_Incidence": "#00CC96",
    "Degree_Centrality": "#AB63FA",
    "Random": "#FFA15A",
    "True_MSE_Oracle": "#19D3F3",
}



def get_policy_color(policy_name):
    """
    Return a consistent color for a policy.

    Variable policies use the same color as their corresponding
    fixed-budget policy.
    """

    base_policy_name = policy_name.replace(
        "_Variable",
        "",
    )

    return POLICY_COLORS.get(
        base_policy_name,
        None,
    )


def create_plotly_time_series_plot(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temporal_mask: torch.Tensor,
    sim_num_steps: list[int],   # e.g., [3, 2, 4, 2]
    node_id: int,
    epoch: int,
    log_key: str | None = None,
    log_to_wandb: bool = True,
):
    """
    Plot y_true / y_pred / mask and draw vertical separators at the end of each
    simulation segment defined by sim_num_steps (run-lengths).

    Args:
        y_true: Ground truth values [T]
        y_pred: Predicted values [T]
        temporal_mask: Binary mask indicating observed timesteps [T]
        sim_num_steps: List of integers indicating the number of timesteps in each simulation segment
        node_id: Node identifier
        epoch: Current epoch
        log_key: Optional key for logging to wandb
        log_to_wandb: Whether to log the plot to wandb
    """
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().flatten().numpy()
        return np.asarray(x).flatten()
    y_true_np = to_np(y_true)
    y_pred_np = to_np(y_pred)
    mask_np = to_np(temporal_mask)
    T = len(y_true_np)
    xs = np.arange(T)
    tickvals = []
    ticktext = []

    boundaries = np.cumsum(np.asarray(sim_num_steps))

    for boundary, sim_length in zip(boundaries[:-1], sim_num_steps[:-1]):
        tickvals.append(boundary - 0.5)
        ticktext.append(sim_length + 9)  # Add 9 to account for the lookback window of 10 timesteps
    # Single subplot
    fig = make_subplots()
    # ------------------------------------------------------------------
    # Shade OBSERVED regions (mask == 1)
    # ------------------------------------------------------------------
    observed = mask_np == 1

    y_min = min(y_true_np.min(), y_pred_np.min())
    y_max = max(y_true_np.max(), y_pred_np.max())

    start = None
    first_region = True

    for i, hidden in enumerate(observed):
        if hidden and start is None:
            start = i

        elif not hidden and start is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        start - 0.5,
                        i - 0.5,
                        i - 0.5,
                        start - 0.5,
                        start - 0.5,
                    ],
                    y=[y_min, y_min, y_max, y_max, y_min],
                    fill="toself",
                    fillcolor="rgba(34, 139, 34, 0.25)",
                    line=dict(width=0),
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=False,
                    name="Observed",
                )
            )
            first_region = False
            start = None

    if start is not None:
        fig.add_trace(
            go.Scatter(
                x=[
                    start - 0.5,
                    T - 0.5,
                    T - 0.5,
                    start - 0.5,
                    start - 0.5,
                ],
                y=[y_min, y_min, y_max, y_max, y_min],
                fill="toself",
                fillcolor="rgba(34, 139, 34, 0.25)",
                line=dict(width=0),
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                name="Observed",
            )
        )
    # ------------------------------------------------------------------
    # Plot ground truth and predictions
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_true_np,
            mode="lines",
            name="Ground truth",
            line=dict(color="blue", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_pred_np,
            mode="lines",
            name="Predictions",
            line=dict(color="red", width=2.5),
        )
    )
    # ------------------------------------------------------------------
    # Draw simulation boundaries
    # ------------------------------------------------------------------
    if sim_num_steps:
        boundaries = np.cumsum(np.asarray(sim_num_steps))
        boundaries = boundaries[(boundaries > 0) & (boundaries < T)]
        for x in boundaries:
            fig.add_vline(
                x=float(x),
                line_dash="dash",
                line_width=2.5,
                line_color="black",
            )
    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f"Node {node_id} (Epoch {epoch})",
            x=0.5,
            xanchor="center",
            font=dict(size=28),
        ),
        xaxis=dict(
            title=dict(
                text="End timestep of each concatenated simulation",
                font=dict(size=24),
            ),
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            title=dict(
                text="Value",
                font=dict(size=24),
            ),
            tickfont=dict(size=20),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.95,      # below the title
            xanchor="left",
            x=0,
            font=dict(size=20),
        ),
        margin=dict(l=60, r=30, t=100, b=50),
    )
    if log_to_wandb:
        key = log_key or f"node{node_id}_epoch{epoch}"
        wandb.log({key: fig})
    return fig


def create_plotly_uncertainty_time_series_plot(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_std: torch.Tensor,
    temporal_mask: torch.Tensor,
    sim_num_steps: list[int],
    node_id: int,
    policy_name: str,
    interval_multiplier: float = 1.96,
):
    """
    Plot ground truth, predictive mean, predictive uncertainty, and observed
    timesteps for one node across concatenated disease simulations.

    Green background regions indicate timesteps at which the node was observed.
    """

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().flatten().numpy()
        return np.asarray(x).flatten()

    y_true_np = to_np(y_true)
    y_pred_np = to_np(y_pred)
    y_std_np = to_np(y_std)
    mask_np = to_np(temporal_mask)

    if not (
        len(y_true_np)
        == len(y_pred_np)
        == len(y_std_np)
        == len(mask_np)
    ):
        raise ValueError(
            "y_true, y_pred, y_std, and temporal_mask must have equal lengths."
        )

    T = len(y_true_np)
    xs = np.arange(T)
    tickvals = []
    ticktext = []

    boundaries = np.cumsum(np.asarray(sim_num_steps))

    for boundary, sim_length in zip(boundaries[:-1], sim_num_steps[:-1]):
        tickvals.append(boundary - 0.5)
        ticktext.append(sim_length + 9)  # Add 9 to account for the lookback window of 10 timesteps

    lower = y_pred_np - interval_multiplier * y_std_np
    upper = y_pred_np + interval_multiplier * y_std_np

    observed = mask_np == 1

    y_min = min(
        y_true_np.min(),
        y_pred_np.min(),
        lower.min(),
    )
    y_max = max(
        y_true_np.max(),
        y_pred_np.max(),
        upper.max(),
    )

    if np.isclose(y_min, y_max):
        padding = 1e-6 if y_min == 0 else abs(y_min) * 0.01
        y_min -= padding
        y_max += padding

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Shade OBSERVED regions
    # ------------------------------------------------------------------
    start = None
    first_region = True

    for i, is_observed in enumerate(observed):
        if is_observed and start is None:
            start = i

        elif not is_observed and start is not None:
            fig.add_trace(
                go.Scatter(
                    x=[
                        start - 0.5,
                        i - 0.5,
                        i - 0.5,
                        start - 0.5,
                        start - 0.5,
                    ],
                    y=[
                        y_min,
                        y_min,
                        y_max,
                        y_max,
                        y_min,
                    ],
                    fill="toself",
                    fillcolor="rgba(34, 139, 34, 0.25)",
                    line=dict(width=0),
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=first_region,
                    name="Observed",
                )
            )

            first_region = False
            start = None

    # Handle an observed region extending to the final timestep
    if start is not None:
        fig.add_trace(
            go.Scatter(
                x=[
                    start - 0.5,
                    T - 0.5,
                    T - 0.5,
                    start - 0.5,
                    start - 0.5,
                ],
                y=[
                    y_min,
                    y_min,
                    y_max,
                    y_max,
                    y_min,
                ],
                fill="toself",
                fillcolor="rgba(34, 139, 34, 0.25)",
                line=dict(width=0),
                mode="lines",
                hoverinfo="skip",
                showlegend=first_region,
                name="Observed",
            )
        )

    # ------------------------------------------------------------------
    # Predictive uncertainty band
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.15)",
            line=dict(color="rgba(0, 0, 0, 0)"),
            name="95% prediction interval",
            hoverinfo="skip",
        )
    )

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_true_np,
            mode="lines",
            name="Ground truth",
            line=dict(color="blue"),
        )
    )

    # ------------------------------------------------------------------
    # Predictive mean
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_pred_np,
            mode="lines",
            name="Predictive mean",
            line=dict(color="red"),
        )
    )

    # ------------------------------------------------------------------
    # Simulation boundaries
    # ------------------------------------------------------------------
    if sim_num_steps:
        boundaries = np.cumsum(
            np.asarray(sim_num_steps, dtype=int)
        )

        boundaries = boundaries[
            (boundaries > 0) & (boundaries < T)
        ]

        for boundary in boundaries:
            fig.add_vline(
                x=float(boundary) - 0.5,
                line_dash="dash",
                line_width=2,
                line_color="black",
            )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                f"{policy_name}: Node {node_id} "
                f"prediction and uncertainty"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="End timestep of each concatenated simulation",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        yaxis_title="Normalized incidence",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0,
        ),
        margin=dict(
            l=60,
            r=30,
            t=100,
            b=50,
        ),
    )

    return fig



def create_prediction_scatter_plot(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    epoch: int,
    sample_size: int = 1000,
    title_prefix: str = "Prediction Quality"
) -> Any:
    """
    Create a prediction-vs-ground-truth scatter plot.
    Each point represents one prediction:
        x-axis = ground truth
        y-axis = prediction
    Points are colored by absolute prediction error.
    A dashed y=x line represents perfect predictions.

    Args:
        y_true: Ground truth values [N]
        y_pred: Predicted values [N] 
        epoch: Current epoch
        sample_size: Number of points to sample for the plot
        title_prefix: Prefix for the plot title

    Returns:
        A wandb.Image containing the Matplotlib figure.
    """
    # Detach from the computation graph and move to CPU
    y_true_flat = y_true.detach().cpu().flatten()
    y_pred_flat = y_pred.detach().cpu().flatten()
    # Convert to NumPy
    y_true_np = y_true_flat.numpy()
    y_pred_np = y_pred_flat.numpy()
    # Absolute prediction error for each point
    absolute_error = np.abs(y_pred_np - y_true_np)
    # Use fixed limits if provided; otherwise calculate common limits
    lower = min(y_true_np.min(), y_pred_np.min())
    upper = max(y_true_np.max(), y_pred_np.max())
    # Add a little padding around the data
    axis_range = upper - lower
    padding = 0.05 * axis_range if axis_range > 0 else 0.1
    lower -= padding
    upper += padding
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        y_true_np,
        y_pred_np,
        c=absolute_error,
        cmap="viridis",
        s=18,
        alpha=0.7,
        edgecolors="none",
    )
    # Perfect-prediction reference line
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="Perfect prediction",
    )
    # Same limits for both axes
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    # One unit on the x-axis has the same visual size as one unit on the y-axis
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{title_prefix} (epoch {epoch})")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Absolute prediction error")
    ax.legend()
    fig.tight_layout()
    wandb_image = wandb.Image(fig)
    # Prevent figures from accumulating in memory during training
    plt.close(fig)
    return wandb_image


def calculate_prediction_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    temporal_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive prediction metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        temporal_mask: Optional mask for observed values

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    # Apply mask if provided
    if temporal_mask is not None:
        mask_flat = temporal_mask.flatten().bool()
        # mask_flat = ~mask_flat # Invert mask: keep only where mask == 0
        y_true_masked = y_true.flatten()[mask_flat]
        y_pred_masked = y_pred.flatten()[mask_flat]
    else:
        y_true_masked = y_true.flatten()
        y_pred_masked = y_pred.flatten()

    # Basic metrics
    mse = torch.mean((y_pred_masked - y_true_masked) ** 2)
    mae = torch.mean(torch.abs(y_pred_masked - y_true_masked))
    rmse = torch.sqrt(mse)

    # Correlation coefficient
    if len(y_true_masked) > 1:
        try:
            correlation = torch.corrcoef(torch.stack(
                [y_pred_masked, y_true_masked]))[0, 1]
            if torch.isnan(correlation):
                correlation = torch.tensor(0.0)
        except:
            correlation = torch.tensor(0.0)
    else:
        correlation = torch.tensor(0.0)

    # Relative metrics
    mean_true = torch.mean(y_true_masked)
    if mean_true != 0:
        mape = torch.mean(
            torch.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
        normalized_rmse = rmse / torch.abs(mean_true)
    else:
        mape = torch.tensor(float('inf'))
        normalized_rmse = torch.tensor(float('inf'))

    # R² score
    ss_res = torch.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = torch.sum((y_true_masked - mean_true) ** 2)
    if ss_tot != 0:
        r2_score = 1 - (ss_res / ss_tot)
    else:
        r2_score = torch.tensor(0.0)

    metrics.update({
        #'mse': mse.item(),
        'mae': mae.item(),
        #'rmse': rmse.item(),
        'correlation': correlation.item(),
        'mape': mape.item() if not torch.isinf(mape) else float('inf'),
        #'normalized_rmse': normalized_rmse.item() if not torch.isinf(normalized_rmse) else float('inf'),
        'r2_score': r2_score.item()
    })

    #if temporal_mask is not None:
    #    metrics.update({
    #        'total_observed_timesteps': temporal_mask.sum().item(),
    #        'avg_observed_ratio': temporal_mask.mean().item(),
    #        'num_fully_observed_nodes': (temporal_mask.sum(dim=0) == temporal_mask.shape[0]).sum().item()
    #    })

    return metrics


def log_model_health(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate and return model health metrics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary of health metrics
    """
    health_metrics = {}

    # Gradient norms
    total_grad_norm = 0
    max_grad_norm = 0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
            max_grad_norm = max(max_grad_norm, param_norm.item())
            param_count += 1

    if param_count > 0:
        total_grad_norm = total_grad_norm ** (1. / 2)
        avg_grad_norm = total_grad_norm / param_count
    else:
        total_grad_norm = 0
        avg_grad_norm = 0

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    # Weight norms
    weight_norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weight_norms.append(param.data.norm(2).item())

    health_metrics.update({
        'total_grad_norm': total_grad_norm,
        'max_grad_norm': max_grad_norm,
        'avg_grad_norm': avg_grad_norm,
        'param_count': param_count,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'avg_weight_norm': np.mean(weight_norms) if weight_norms else 0,
        'max_weight_norm': np.max(weight_norms) if weight_norms else 0,
    })



def hex_to_rgba(hex_color, opacity):
    """Convert a hexadecimal color to an RGBA string."""
    hex_color = hex_color.lstrip("#")

    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    return f"rgba({red}, {green}, {blue}, {opacity})"

def create_budget_allocation_plot(
    mean_num_nodes_to_test_list,
    std_num_nodes_to_test_list,
    lookback_length,
):
    """
    Plot mean variable testing budget allocation across seeds,
    with +/- 1 standard deviation across seeds.

    The first testing decision corresponds to disease timestep L,
    since the first model input contains a lookback window of
    L historical incidence values.
    """

    mean_budget = np.asarray(
        mean_num_nodes_to_test_list,
        dtype=float,
    )

    std_budget = np.asarray(
        std_num_nodes_to_test_list,
        dtype=float,
    )

    timesteps = np.arange(
        lookback_length,
        lookback_length + len(mean_budget),
    )

    lower = mean_budget - std_budget
    upper = mean_budget + std_budget

    # Number of selected nodes cannot be negative
    lower = np.maximum(lower, 0)

    line_color = "#636EFA"
    shading_color = "rgba(99, 110, 250, 0.30)"

    fig = go.Figure()

    # ---------------------------------------------------------
    # +/- 1 standard deviation shaded region
    # ---------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [
                    timesteps,
                    timesteps[::-1],
                ]
            ),
            y=np.concatenate(
                [
                    upper,
                    lower[::-1],
                ]
            ),
            fill="toself",
            fillcolor=shading_color,
            line=dict(
                color="rgba(0,0,0,0)"
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ---------------------------------------------------------
    # Mean testing budget
    # ---------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=mean_budget,
            mode="lines",
            line=dict(
                color=line_color,
                width=2.5,
            ),
            showlegend=False,
            hovertemplate=(
                "Timestep: %{x}<br>"
                "Mean testing budget: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Variable Testing Budget Allocation",
            x=0.5,
            xanchor="center",
            font=dict(size=28),
        ),
        xaxis=dict(
            title=dict(
                text="Timestep of disease spread",
                font=dict(size=24),
            ),
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            title=dict(
                text="Nodes selected",
                font=dict(size=24),
            ),
            tickfont=dict(size=20),
        ),
        showlegend=False,
        margin=dict(
            l=60,
            r=30,
            t=100,
            b=60,
        ),
    )

    return fig

def create_budget_allocation_plot(
    num_nodes_to_test_list,
    lookback_length,
):
    """
    Plot Variable testing budget allocation across timesteps.

    The first testing decision corresponds to disease timestep L,
    since the first model input contains a lookback window of
    L historical incidence values.
    """

    timesteps = list(
        range(
            lookback_length,
            lookback_length
            + len(num_nodes_to_test_list),
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=num_nodes_to_test_list,
            mode="lines",
            name="Testing budget",
            line=dict(width=2.5),
        )
    )

    fig.update_layout(
        title=dict(
            text="Variable Testing Budget Allocation",
            font=dict(size=28),
        ),
        xaxis=dict(
            title=dict(
                text="Timestep of disease spread",
                font=dict(size=24),
            ),
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            title=dict(
                text="Nodes selected",
                font=dict(size=24),
            ),
            tickfont=dict(size=20),
        ),
        legend=dict(
            font=dict(size=20),
        ),
        template="plotly_white",
    )

    return fig


def create_policy_results_table_plot(
    policy_eval_histories,
    max_timestep,
    lookback_length,
    timestep_spacing=50,
):
    """
    Create a Plotly table of median evaluation MSE values averaged across multiple seeds.

    Rows:
        Policies.

    Columns:
        First available prediction timestep L,
        then 50, 100, 150, ...

    The lowest MSE in each column is shown in bold.
    """

    # ---------------------------------------------------------
    # Timesteps to display
    # ---------------------------------------------------------
    sampled_timesteps = [lookback_length]

    sampled_timesteps.extend(
        timestep
        for timestep in range(
            timestep_spacing,
            max_timestep,
            timestep_spacing,
        )
        if timestep > lookback_length
    )

    policy_names = list(
        policy_eval_histories.keys()
    )

    # ---------------------------------------------------------
    # Wrap long policy names onto multiple lines
    # ---------------------------------------------------------
    def wrap_policy_name(
        name,
        max_chars=20,
    ):
        parts = name.split("_")

        if len(parts) == 1:
            return name

        lines = []
        current_line = parts[0]

        for part in parts[1:]:

            candidate = (
                current_line
                + "_"
                + part
            )

            if len(candidate) > max_chars:
                lines.append(
                    current_line
                )
                current_line = part
            else:
                current_line = candidate

        lines.append(
            current_line
        )

        return "<br>".join(lines)

    displayed_policy_names = [
        wrap_policy_name(name)
        for name in policy_names
    ]

    # ---------------------------------------------------------
    # Extract MSE values
    # ---------------------------------------------------------
    values = []

    for policy in policy_names:

        history = policy_eval_histories[
            policy
        ]

        mse = np.asarray(
            history["median"],
            dtype=float,
        )

        row = []

        for timestep in sampled_timesteps:

            # history index 0 corresponds to simulation timestep L
            index = (
                timestep
                - lookback_length
            )

            if (
                0 <= index < len(mse)
            ):
                row.append(
                    mse[index]
                )
            else:
                row.append(
                    np.nan
                )

        values.append(
            row
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    # ---------------------------------------------------------
    # Find minimum value in each timestep column
    # ---------------------------------------------------------
    best_rows = []

    for col_idx in range(
        values.shape[1]
    ):

        column = values[
            :,
            col_idx,
        ]

        if np.all(
            np.isnan(column)
        ):
            best_rows.append(
                None
            )
        else:
            best_rows.append(
                int(
                    np.nanargmin(
                        column
                    )
                )
            )

    # ---------------------------------------------------------
    # Construct table values
    # ---------------------------------------------------------
    table_columns = [
        displayed_policy_names
    ]

    for col_idx in range(
        len(sampled_timesteps)
    ):

        column_text = []

        for row_idx in range(
            len(policy_names)
        ):

            value = values[
                row_idx,
                col_idx,
            ]

            if np.isnan(value):
                text = "--"

            else:
                text = (
                    f"{value:.4f}"
                )

                if (
                    best_rows[col_idx]
                    is not None
                    and row_idx
                    == best_rows[col_idx]
                ):
                    text = (
                        f"<b>{text}</b>"
                    )

            column_text.append(
                text
            )

        table_columns.append(
            column_text
        )

    # ---------------------------------------------------------
    # Column widths
    #
    # Give the policy column extra width.
    # ---------------------------------------------------------
    policy_column_width = 2.2

    timestep_column_width = 1.0

    column_widths = (
        [policy_column_width]
        + [
            timestep_column_width
            for _ in sampled_timesteps
        ]
    )

    total_column_width = sum(
        column_widths
    )

    # x-coordinate where the timestep columns begin
    timestep_x_start = (
        policy_column_width
        / total_column_width
    )

    # ---------------------------------------------------------
    # Plotly table
    # ---------------------------------------------------------
    table_top = 0.84

    fig = go.Figure(
        data=[
            go.Table(
                domain=dict(
                    x=[0, 1],
                    y=[0, table_top],
                ),
                columnwidth=column_widths,
                header=dict(
                    values=[
                        "<b>Policy</b>"
                    ]
                    + [
                        f"<b>{t}</b>"
                        for t
                        in sampled_timesteps
                    ],
                    align="center",
                    height=46,
                    font=dict(size=22),
                ),
                cells=dict(
                    values=table_columns,
                    align=[
                        "left"
                    ]
                    + [
                        "center"
                        for _
                        in sampled_timesteps
                    ],
                    height=64,
                    font=dict(size=20),
                ),
            )
        ]
    )

    # ---------------------------------------------------------
    # Add merged "Timestep" header above timestep columns
    # ---------------------------------------------------------
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=timestep_x_start,
        x1=1.0,
        y0=table_top,
        y1=0.91,
        fillcolor="#C6D4E3",
        line=dict(
            color="white",
            width=1,
        ),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=(
            timestep_x_start
            + (
                1.0
                - timestep_x_start
            ) / 2
        ),
        y=0.875,
        text="<b>Timestep</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(size=22),
    )

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                "Median evaluation MSE "
                "across acquisition policies"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=28),
        ),
        height=(
            220
            + 64
            * len(policy_names)
        ),
        margin=dict(
            l=20,
            r=20,
            t=80,
            b=20,
        ),
    )

    return fig


def aggregate_policy_histories_across_seeds(
    seed_policy_histories,
):
    """
    Aggregate policy evaluation histories across random seeds.

    For each policy and timestep:
        - central line = mean of the seed-level median MSE values
        - lower bound = mean - standard deviation across seeds
        - upper bound = mean + standard deviation across seeds

    Args:
        seed_policy_histories:
            List of policy_eval_histories dictionaries,
            one dictionary per random seed.

    Returns:
        aggregated_histories:
            Dictionary in the same format expected by
            create_policy_comparison_plot.
    """
    
    if not seed_policy_histories:
        raise ValueError(
            "seed_policy_histories is empty."
        )

    policy_names = list(
        seed_policy_histories[0].keys()
    )

    aggregated_histories = {}

    for policy_name in policy_names:

        # Histories for this policy from every seed
        seed_histories = [
            seed_history[policy_name]
            for seed_history in seed_policy_histories
            if policy_name in seed_history
        ]

        max_length = max(
            len(history["median"])
            for history in seed_histories
        )

        mean_across_seeds = []
        lower_std_across_seeds = []
        upper_std_across_seeds = []

        for timestep_idx in range(max_length):

            values = [
                history["median"][timestep_idx]
                for history in seed_histories
                if timestep_idx < len(history["median"])
            ]

            mean_value = float(
                np.mean(values)
            )

            if len(values) > 1:
                std_value = float(np.std(values, ddof=1))
            else:
                std_value = 0.0

            mean_across_seeds.append(
                mean_value
            )

            lower_std_across_seeds.append(
                mean_value - std_value
            )

            upper_std_across_seeds.append(
                mean_value + std_value
            )

        # Keep the unshifted indexing convention.
        # create_policy_comparison_plot will shift by L.
        timesteps = list(
            range(
                1,
                max_length + 1,
            )
        )

        aggregated_histories[
            policy_name
        ] = {
            "timesteps": timesteps,

            # These names are retained because
            # create_policy_comparison_plot expects them.
            "median": mean_across_seeds,
            "p25": lower_std_across_seeds,
            "p75": upper_std_across_seeds,
        }

    return aggregated_histories




def aggregate_uncertainty_histories_across_seeds(
    seed_uncertainty_histories,
):
    """
    Aggregate uncertainty metrics across random seeds.

    Within each seed, the supplied history already contains
    the median metric across deployment simulations.

    Across seeds:
        - central line = mean
        - lower bound = mean - standard deviation
        - upper bound = mean + standard deviation
    """

    if not seed_uncertainty_histories:
        raise ValueError(
            "seed_uncertainty_histories is empty."
        )

    metric_keys = set()

    for seed_history in seed_uncertainty_histories:
        metric_keys.update(
            seed_history.keys()
        )

    aggregated_histories = {}

    for metric_key in sorted(metric_keys):

        histories = [
            seed_history[metric_key]
            for seed_history in seed_uncertainty_histories
            if metric_key in seed_history
        ]

        if not histories:
            continue

        max_length = max(
            len(history["median"])
            for history in histories
        )

        mean_across_seeds = []
        lower_std_across_seeds = []
        upper_std_across_seeds = []

        for timestep_idx in range(max_length):

            values = [
                history["median"][timestep_idx]
                for history in histories
                if timestep_idx < len(
                    history["median"]
                )
            ]

            mean_value = float(
                np.mean(values)
            )

            # Works for one seed as well
            if len(values) > 1:
                std_value = float(np.std(values, ddof=1))
            else:
                std_value = 0.0

            mean_across_seeds.append(
                mean_value
            )

            lower_std_across_seeds.append(
                mean_value - std_value
            )

            upper_std_across_seeds.append(
                mean_value + std_value
            )

        reference_history = histories[0]

        aggregated_histories[
            metric_key
        ] = {
            "policy_name":
                reference_history[
                    "policy_name"
                ],

            "metric_name":
                reference_history[
                    "metric_name"
                ],

            "timesteps": list(
                range(
                    1,
                    max_length + 1,
                )
            ),

            # Keep these names because
            # create_policy_comparison_plot expects them.
            "median":
                mean_across_seeds,

            "p25":
                lower_std_across_seeds,

            "p75":
                upper_std_across_seeds,
        }

    return aggregated_histories




def create_pretrained_ablation_results_table(
    full_model_mse,
    ablated_model_mse,
    gcn_only_model_mse,
):
    """
    Create a Plotly table comparing the mean test MSE across seeds
    for the three pretrained model architectures.
    """

    model_names = [
        "Full GRU-GCN model",
        "Graph-ablated model",
        "Temporal-ablated GCN model",
    ]

    mse_values = [
        full_model_mse,
        ablated_model_mse,
        gcn_only_model_mse,
    ]

    # Find the model with the lowest MSE
    best_idx = int(
        np.argmin(mse_values)
    )

    mse_text = []

    for idx, value in enumerate(mse_values):

        text = f"{value:.4f}"

        if idx == best_idx:
            text = f"<b>{text}</b>"

        mse_text.append(text)

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[
                    2.5,
                    1.5,
                ],
                header=dict(
                    values=[
                        "<b>Model</b>",
                        "<b>Mean test MSE across seeds</b>",
                    ],
                    align="center",
                    height=46,
                    font=dict(size=22),
                ),
                cells=dict(
                    values=[
                        model_names,
                        mse_text,
                    ],
                    align=[
                        "left",
                        "center",
                    ],
                    height=54,
                    font=dict(size=20),
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="No_Fine_Tuning ablation results",
            x=0.5,
            xanchor="center",
            font=dict(size=28),
        ),
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig



def create_policy_comparison_plot(
    policy_histories,
    metric_name,
    lookback_length,
    show_quantiles=True,
    color_overrides=None,
    xaxis_title="Timestep of disease spread",
):
    """
    Create a Plotly figure comparing multiple active-learning policies.

    The first model prediction corresponds to disease timestep
    lookback_length, since the first input contains the previous L
    incidence values.

    Args:
        policy_histories:
            Dictionary mapping policy names to their performance histories.

        metric_name:
            Name of the metric being compared.

        lookback_length:
            Number of historical timesteps L used by the model.
            The first plotted prediction therefore corresponds to
            disease timestep L.

        show_quantiles:
            Whether to show the interquartile range (IQR) shaded region.

        color_overrides:
            Optional dictionary mapping policy names to specific colors.

    Returns:
        Plotly figure object.
    """

    fig = go.Figure()

    for policy_name, history in policy_histories.items():

        # ---------------------------------------------------------
        # Choose consistent policy color
        # ---------------------------------------------------------
        if (
            color_overrides is not None
            and policy_name in color_overrides
        ):
            policy_color = color_overrides[
                policy_name
            ]
        else:
            policy_color = get_policy_color(
                policy_name
            )

        # Fallback for names that are not acquisition policies
        if policy_color is None:
            policy_color = px_colors.qualitative.Plotly[
                list(policy_histories.keys()).index(policy_name)
                % len(px_colors.qualitative.Plotly)
            ]

        shading_color = hex_to_rgba(
            policy_color,
            0.18,
        )

        median = np.asarray(
            history["median"],
            dtype=float,
        )

        p25 = np.asarray(
            history["p25"],
            dtype=float,
        )

        p75 = np.asarray(
            history["p75"],
            dtype=float,
        )

        # Original histories start at 1.
        # Shift them so the first model prediction is plotted at timestep L.
        timesteps = (
            np.asarray(
                history["timesteps"],
                dtype=int,
            )
            + lookback_length
            - 1
        )

        # ---------------------------------------------------------
        # IQR shaded region
        # ---------------------------------------------------------
        if show_quantiles:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate(
                        [
                            timesteps,
                            timesteps[::-1],
                        ]
                    ),
                    y=np.concatenate(
                        [
                            p75,
                            p25[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=shading_color,
                    line=dict(
                        color="rgba(0,0,0,0)"
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=policy_name,
                )
            )

        # ---------------------------------------------------------
        # Median line
        # ---------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=median,
                mode="lines",
                name=policy_name,
                line=dict(
                    color=policy_color,
                    width=2.5,
                ),
                marker=dict(
                    color=policy_color
                ),
                legendgroup=policy_name,
                hovertemplate=(
                    "Timestep: %{x}<br>"
                    "Median: %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                f"Active-learning policy comparison: "
                f"{metric_name}"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=28),
        ),
        xaxis=dict(
            title=dict(
                text=xaxis_title,
                font=dict(size=24),
            ),
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            title=dict(
                text=metric_name,
                font=dict(size=24),
            ),
            tickfont=dict(size=20),
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(size=20),
        ),
        margin=dict(
            l=60,
            r=30,
            t=100,
            b=60,
        ),
    )

    return fig