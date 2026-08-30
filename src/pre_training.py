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
from src.visualization_utils import create_budget_allocation_plot, create_prediction_scatter_plot, create_plotly_time_series_plot, calculate_prediction_metrics
from src.adaptive_batch_allocation import dhondt_allocation
import wandb
from hydra.utils import instantiate
from itertools import groupby


def train_surrogate_model(
    train_dataset, sim_ids_train, test_dataset, sim_ids_test,
    model, optimizer,
    lr_scheduler, simulation_params, cfg, device, gcn_only=False,
):
    model = model.to(device)
    if cfg.dataset.output_binary:
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    else:
        loss_fn = F.gaussian_nll_loss
    mse_fn = nn.MSELoss()

    # per-step temporal smoothing weight (optional in config)
    smooth_w = getattr(cfg.training.pretraining, 'smooth_loss_weight', 0.1)
    # if we are running the ablation experiment with no temporal component, temporal smoothing is not needed
    if gcn_only:
        smooth_w = 0.0
    truncated_BPTT_length = getattr(cfg.training.pretraining, 'truncated_BPTT_length', 1024)
    
    prev_y_hat = None
    hidden_states = None
    num_nodes_to_test_list = []
    train_loss_history = []
    eval_mse_history = []
    for epoch in range(cfg.training.pretraining.max_epoch):
        model.train()
        optimizer.zero_grad()
        prev_sim_id = -1
        loss = torch.tensor(0.0, device=device)
        loss_sum = torch.tensor(0.0, device=device)
        log_mse = torch.tensor(0.0, device=device)
        for time, snapshot in enumerate(train_dataset):
            snapshot = snapshot.to(device)
            y_true = snapshot.y.view(-1)
            #if time == 0:
            #    print(y_true)

            #check if we are in a new simulation
            # NOTE: this is a hack to ensure that we reset the temporal masks
            # every time we start a new simulation in the train dataset
            # this is important because the train simulations are just
            # a concatenation of multiple simulations
            if sim_ids_train[time] != prev_sim_id:
                prev_sim_id = sim_ids_train[time]
                prev_y_hat = None
                hidden_states = None
                model.embed.reset_train_temporal_masks()
            x_t = snapshot.x
            y_hat, y_hat_std, hidden_states, temporal_mask = model(x_t, snapshot.edge_index, snapshot.edge_attr, hidden_states)
            y_hat = y_hat.view(-1) # [num_nodes]
            y_hat_std = y_hat_std.view(-1) # [num_nodes]
            y_true = snapshot.y.view(-1)
            #if (time % 100) == 0:
            #    print(f"y_hat of some node {y_hat[0].item()} and y_true of the same node {y_true[0].item()} at timestep {time}")
            y_hat_masked = y_hat[temporal_mask]
            y_hat_std_masked = y_hat_std[temporal_mask]
            y_true_masked = y_true[temporal_mask]
            loss += loss_fn(y_hat_masked, y_true_masked, y_hat_std_masked.square(), full=True)
            log_mse += mse_fn(y_hat.squeeze().float(), y_true.float())
            #print("bce:",loss_fn(y_hat_masked, y_true_masked).item())

            # Temporal smoothing regularizer across consecutive predictions
            if smooth_w > 0 and prev_y_hat is not None:
                # Apply smoothing loss to encourage temporal consistency
                temporal_smooth_loss = mse_fn(
                    y_hat.squeeze(), prev_y_hat.to(y_hat.device)
                )
                #print("smooth:", temporal_smooth_loss.item(), "smooth_w:", smooth_w)
                loss += smooth_w * temporal_smooth_loss

            # Backpropagation every truncated_BPTT_length steps
            # use time + 1 to not backprop after the very first step
            if (time + 1) % truncated_BPTT_length == 0 and hidden_states is not None:
                average_loss = loss / truncated_BPTT_length
                loss_sum += loss.item()
                average_loss.backward()
                # # for when we are running the ablation experiment with no temporal component
                # if gcn_only:
                #     torch.nn.utils.clip_grad_norm_(
                #         model.parameters(),
                #         max_norm=cfg.model.clip_grad_max_norm
                #     )
                #     optimizer.step()
                loss = torch.tensor(0.0, device=device)
                for h in hidden_states:
                    h.detach_()  # Detach hidden state to break gradient flow      
            prev_y_hat = y_hat.squeeze().detach()
  
        # Final backpropagation for any remaining steps if epoch ended in the middle of a truncated BPTT chunk
        if (time + 1) % truncated_BPTT_length != 0:
            average_loss = loss / ((time + 1) % truncated_BPTT_length)
            loss_sum += loss.item()
            average_loss.backward()

        # clipping gradients for training stability
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.model.clip_grad_max_norm
        )

        # # Normal GRU-GCN accumulates gradients across the whole epoch.
        # # GCN-only has already updated after each complete chunk.
        # if not gcn_only or (time + 1) % truncated_BPTT_length != 0:

        # accumulate gradients of multiple temporal chunks before optimizer step for stability
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loss_sum = loss_sum / (time+1)
        log_mse = log_mse / (time+1)

        if epoch == cfg.training.pretraining.max_epoch - 1:
            eval_mse, y_preds, y_trues, temporal_masks, median_timestep_mse = evaluate_surrogate_model(
                    test_dataset, sim_ids_test,
                    model, simulation_params, device, return_timestep_median_mse=True
                )
            
            # compute number of tests allocated to each timestep using adaptive test allocation mechanism
            num_nodes_to_test_dict, num_nodes_to_test_list = dhondt_allocation(
                total_budget=cfg.training.active_learning.num_nodes_to_test * (cfg.dataset.SIR.max_T - cfg.dataset.num_lookback_steps),
                parties = list(range(1, len(median_timestep_mse) + 1)), 
                votes=median_timestep_mse.tolist(),
                max_seats_per_party=cfg.model.model.num_nodes,
                )
            # create plot of budget allocation across timesteps
            budget_allocation_plot = create_budget_allocation_plot(num_nodes_to_test_list, lookback_length=cfg.dataset.num_lookback_steps)

            print(f"Adaptive test allocation:")
            print()
            print(num_nodes_to_test_dict)
        else:
            eval_mse, y_preds, y_trues, temporal_masks = evaluate_surrogate_model(
                    test_dataset, sim_ids_test,
                    model, simulation_params, device
                )
            
        train_loss_history.append(
                        torch.sqrt(loss_sum).item()
                    )
        eval_mse_history.append(
            eval_mse.item()
        )
        print(f"Pre-training phase - epoch {epoch}, training loss: {loss_sum.item():.4f}, train RMSE: {torch.sqrt(log_mse).item():.4f}, eval RMSE: {torch.sqrt(eval_mse).item():.4f}")
        print()

        # log in wandb every 10 epochs
        if epoch % 10 == 0 and cfg.seed == cfg.plot_seed:
            # Log metrics to wandb
            log_dict = {
                # "pretrain/epoch": epoch,
                # "pretrain/train_loss": loss_sum.item(),
                # "pretrain/train_rmse": torch.sqrt(log_mse).item(),
                # "pretrain/eval_rmse": torch.sqrt(eval_mse).item(),
                # "pretrain/lr": optimizer.param_groups[0]["lr"]
            }

            # # Calculate additional metrics using utility function
            # prediction_metrics = calculate_prediction_metrics(
            #     y_trues, y_preds, temporal_masks)
            
            # # Add prediction metrics to log dict
            # for key, value in prediction_metrics.items():
            #     log_dict[f'pretrain/eval_{key}'] = value

            # Create visualizations for a subset of nodes to reduce memory usage
            max_nodes_to_plot = min(2, y_preds.shape[1])  # Plot max 2 nodes
            node_indices = torch.linspace(
                0, y_preds.shape[1]-1, max_nodes_to_plot, dtype=torch.long)

            # collapse sim_ids_test to a list of ints
            sim_num_steps = [len(list(g)) for _, g in groupby(sim_ids_test)]
            for i, node_id in enumerate(node_indices):
                log_dict[f"pretrain/plotly_epoch_{epoch}_node_{node_id}"] = create_plotly_time_series_plot(
                    y_trues[:, node_id], y_preds[:, node_id], temporal_masks[:, node_id], sim_num_steps,
                    int(node_id.item()), epoch
                )

            # Add scatter plot for overall prediction quality
            log_dict[f"pretrain/epoch_{epoch}_scatter_plot"] = create_prediction_scatter_plot(
                    y_trues, y_preds, epoch
                )

            # add plot of budget allocation across timesteps to wandb
            # MUST have final epoch be divisible by ten, otherwise the budget allocation plot will never be logged to wandb
            if epoch == cfg.training.pretraining.max_epoch - 1:
                log_dict["pretrain/adaptive_budget_allocation"] = budget_allocation_plot
            
            wandb.log(log_dict)
        elif cfg.seed == cfg.plot_seed:
            wandb.log(
                {
                    # "pretrain/epoch": epoch,
                    # 'pretrain/train_loss': loss_sum.item(),
                    # 'pretrain/train_rmse': torch.sqrt(log_mse).item(),
                    # "pretrain/eval_rmse": torch.sqrt(eval_mse).item(),
                    # 'pretrain/lr': optimizer.param_groups[0]["lr"]
                }
            )


    return model, num_nodes_to_test_list, train_loss_history, eval_mse_history

            

def evaluate_surrogate_model(
    test_dataset,
    sim_ids_test,
    model,
    simulation_params,
    device,
    return_timestep_median_mse=False,
):
    """
    Evaluate the surrogate model on the test dataset. If return_timestep_median_mse is True, 
    also compute the median next timestep MSE across all simulations in the testing dataset.
    """
    model = model.to(device)
    model.eval()
    mse_fn = nn.MSELoss()
    eval_mse = torch.tensor(0.0, device=device)
    hidden_states = None
    prev_sim_id = -1
    y_preds = []
    y_trues = []
    temporal_masks = []
    if return_timestep_median_mse:
        all_sim_timestep_mse = []
        current_sim_timestep_mse = []
    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset):
            snapshot = snapshot.to(device)
            # Check whether a new simulation has started
            if sim_ids_test[time] != prev_sim_id:
                if return_timestep_median_mse:
                    # Save previous simulation
                    if current_sim_timestep_mse:
                        all_sim_timestep_mse.append(
                            torch.tensor(
                                current_sim_timestep_mse,
                                dtype=torch.float32,
                            )
                        )
                    current_sim_timestep_mse = []
                prev_sim_id = sim_ids_test[time]
                hidden_states = None
                model.embed.reset_test_temporal_masks()
            x_t = snapshot.x
            (
                y_hat,
                _,
                hidden_states,
                temporal_mask,
            ) = model(
                x_t,
                snapshot.edge_index,
                snapshot.edge_attr,
                hidden_states,
            )
            y_hat = y_hat.view(-1)
            y_true = snapshot.y.view(-1)
            y_preds.append(
                y_hat.detach().cpu()
            )
            y_trues.append(
                y_true.detach().cpu()
            )
            temporal_masks.append(
                temporal_mask.float().detach().cpu()
            )
            # MSE across all nodes at this timestep
            timestep_mse = mse_fn(
                y_hat.float(),
                y_true.float(),
            )
            eval_mse += timestep_mse
            if return_timestep_median_mse:
                current_sim_timestep_mse.append(
                    timestep_mse.item()
                )
        # Save final simulation
        if (
            return_timestep_median_mse
            and current_sim_timestep_mse
        ):
            all_sim_timestep_mse.append(
                torch.tensor(
                    current_sim_timestep_mse,
                    dtype=torch.float32,
                )
            )
        eval_mse = eval_mse / (time + 1)
    y_preds = torch.stack(y_preds)
    y_trues = torch.stack(y_trues)
    temporal_masks = torch.stack(temporal_masks)
    # median next timestep MSE across all simulations in the testing dataset
    if return_timestep_median_mse:
        max_sim_length = max(
            len(sim_mse)
            for sim_mse in all_sim_timestep_mse
        )
        # Pad shorter simulations with NaN
        sim_timestep_mse_tensor = torch.full(
            (
                len(all_sim_timestep_mse),
                max_sim_length,
            ),
            float("nan"),
            dtype=torch.float32,
        )
        for sim_idx, sim_mse in enumerate(
            all_sim_timestep_mse
        ):
            sim_timestep_mse_tensor[
                sim_idx,
                :len(sim_mse),
            ] = sim_mse
        # Shape: [max_sim_length]
        median_timestep_mse = torch.nanmedian(
            sim_timestep_mse_tensor,
            dim=0,
        ).values
        return (
            eval_mse,
            y_preds,
            y_trues,
            temporal_masks,
            median_timestep_mse,
        )
    return (
        eval_mse,
        y_preds,
        y_trues,
        temporal_masks,
    )