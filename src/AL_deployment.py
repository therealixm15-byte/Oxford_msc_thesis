from email import policy

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
from hydra.utils import instantiate
import copy
import numpy as np
import wandb
import plotly.graph_objects as go
from src.visualization_utils import create_policy_comparison_plot, create_plotly_uncertainty_time_series_plot, create_policy_results_table_plot
from src.quality_of_quantified_uncertainty import interval_metrics, crps


def AL_deployment(
    AL_dataset,
    AL_dataset_ablated,
    sim_ids_AL,
    model,
    ablated_model,
    optimizer,
    ablated_optimizer,
    lr_scheduler,
    ablated_lr_scheduler,
    simulation_params,
    num_nodes_to_test_list,
    cfg,
    device,
    policy_fn=None,
):
    # Run experiment will different acquisition functions
    # save the pretrained model state to restore after each acquisition function has been run 
    pretrained_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())
    policy_train_histories = {}
    policy_eval_histories = {}
    incidence_history = {}
    simulation_update_timesteps = []
    plot_uncertainty_metrics = {}
    all_uncertainty_metrics = {}
    # Evaluate the pretrained model without any active learning acquisition or fine-tuning, for comparison with the active learning deployment phase
    pretrained_eval_history = evaluate_pretrained_baseline(
        AL_dataset=AL_dataset,
        sim_ids_AL=sim_ids_AL,
        model=model,
        pretrained_model_state=pretrained_model_state,
        cfg=cfg,
        device=device,
    )
    policy_eval_histories["No_Fine_Tuning"] = pretrained_eval_history
    model.load_state_dict(copy.deepcopy(pretrained_model_state))
    for policy_name, policy_cfg in cfg.training.active_learning.policies.items():
        print(f"Running AL experiment with policy: {policy_name}")
        simulation_update_timesteps = []
        # Restore identical starting conditions
        model.load_state_dict(copy.deepcopy(pretrained_model_state))
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer_state))
        lr_scheduler.load_state_dict(copy.deepcopy(initial_scheduler_state))
        # Instantiate this policy
        if policy_name == "Ablated_Model":
            model = ablated_model
            optimizer = ablated_optimizer
            lr_scheduler = ablated_lr_scheduler
            AL_dataset = AL_dataset_ablated
            policy_fn = instantiate(cfg.training.active_learning.policies["BALSA_KL_Pairs"])
        else:
            model = model
            optimizer = optimizer
            lr_scheduler = lr_scheduler
            AL_dataset = AL_dataset
            policy_fn = instantiate(policy_cfg)
        if cfg.dataset.output_binary:
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            loss_fn = F.gaussian_nll_loss
        mse_fn = nn.MSELoss()

        # per-step temporal smoothing weight (optional in config)
        smooth_w = getattr(cfg.training.active_learning, 'smooth_loss_weight', 0.1)
        truncated_BPTT_length = getattr(cfg.training.active_learning, 'truncated_BPTT_length', 1024)

        # Save original pretrained state to restore after each simulation in the AL deployment phase
        initial_model_state = copy.deepcopy(model.state_dict())
        initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
        initial_scheduler_state = copy.deepcopy(lr_scheduler.state_dict())
        unique_sim_ids = sorted(set(sim_ids_AL))
        acc_train_loss = []
        acc_eval_loss = []
        all_true_incidence = []
        # create plot for one node's predictions and true incidence values across all timesteps of the simulation with uncertainty intervals for the predictions
        policy_plot_y_trues = []
        policy_plot_y_preds = []
        policy_plot_y_stds = []
        policy_plot_masks = []
        policy_sim_num_steps = []
        for sim_id in unique_sim_ids:
            print(f"Starting AL deployment for simulation {sim_id}")
            # Restore original pretrained model before each AL experiment
            model.load_state_dict(initial_model_state)
            optimizer.load_state_dict(initial_optimizer_state)
            lr_scheduler.load_state_dict(initial_scheduler_state)
            # Select only snapshots belonging to this simulation
            sim_indices = [
                i for i, sid in enumerate(sim_ids_AL)
                if sid == sim_id
            ]
            sim_train_losses = []
            sim_eval_losses = []
            acc_uncertainty_metrics = {}
            optimizer.zero_grad()
            prev_y_hat = None
            hidden_states = None
            true_incidence = []
            model.embed.reset_train_temporal_masks()
            loss = torch.tensor(0.0, device=device)
            #instantiate AL_mask for the observed nodes by AL policy
            AL_mask = torch.zeros(cfg.model.model.num_nodes, dtype=torch.bool, device=device)
            # save snapshots of windows within truncated_BPTT_length to train for multiple epochs by replaying the windows
            window_snapshots = []
            evaluate_loss = -1
            sim_plot_y_trues = []
            sim_plot_mc_means = []
            sim_plot_mc_stds = []
            temporal_masks = [ torch.zeros(cfg.model.model.num_nodes,dtype=torch.float32)]
            for local_time_of_sim, dataset_idx in enumerate(sim_indices[:-1]):
                model.train()
                #print(f"this is used to index the AL_dataset: {dataset_idx}")
                snapshot = AL_dataset[dataset_idx].to(device)
                snapshot = snapshot.to(device)
                y_true = snapshot.y.view(-1)
                x_t = snapshot.x
                if len(window_snapshots) == 0:
                    saved_initial_hidden_states = (
                        None if hidden_states is None else [h.detach().clone() for h in hidden_states]
                    )
                hidden_states_copy = (None if hidden_states is None else [h.detach().clone() for h in hidden_states])

                # now we use an external mask as an argument to the model's forward pass for the first time
                # this mask keeps track of the nodes that have been selected by the AL policy on each timestep
                y_hat, y_hat_std, hidden_states, _ = model(x_t, snapshot.edge_index, snapshot.edge_attr, hidden_states, external_mask=AL_mask)
                y_hat = y_hat.view(-1) # [num_nodes]
                y_hat_std = y_hat_std.view(-1) # [num_nodes]
                y_true = snapshot.y.view(-1)

                # evaluate the model on the next timestep after the current timestep
                evaluate_loss = evaluate_surrogate_model(AL_dataset, dataset_idx, hidden_states_copy, AL_mask, model, device, cfg)
                sim_eval_losses.append((evaluate_loss))


                num_nodes_to_select = num_nodes_to_test_list[local_time_of_sim]
                if policy_name == "MC_Dropout_Variance" or policy_name == "BALSA_KL_Pairs" or policy_name == "Ablated_Model" or policy_name == "BALSA_KL_Pairs_Variable":
                    AL_selected_nodes, mc_predictions = policy_fn(model, snapshot, hidden_states_copy, AL_mask, num_nodes_to_select, cfg, device)
                    mc_mean = mc_predictions.mean(dim=0)
                    mc_std = mc_predictions.std(
                        dim=0,
                        unbiased=False,
                    )
                    sim_plot_y_trues.append(
                        y_true.detach().cpu()
                    )
                    sim_plot_mc_means.append(
                        mc_mean.detach().cpu()
                    )
                    sim_plot_mc_stds.append(
                        mc_std.detach().cpu()
                    )
                    coverage, width = interval_metrics(mc_predictions, y_true)
                    crps_value = crps(mc_predictions, y_true)
                    acc_uncertainty_metrics.setdefault(f"{policy_name}_coverage", []).append(coverage)
                    acc_uncertainty_metrics.setdefault(f"{policy_name}_width", []).append(width)
                    acc_uncertainty_metrics.setdefault(f"{policy_name}_crps", []).append(crps_value)
                elif policy_name == "Predicted_Incidence_Variable" or policy_name == "Predicted_Incidence":
                    AL_selected_nodes = policy_fn(model, snapshot, hidden_states_copy, AL_mask, num_nodes_to_select, cfg, device, y_hat.detach())
                else:
                    AL_selected_nodes = policy_fn(model, snapshot, hidden_states_copy, AL_mask, num_nodes_to_select, cfg, device)
                window_snapshots.append({"snapshot": snapshot,"AL_mask": AL_mask.clone()})
                # update AL_mask for the next timestep using selected datapoints from the active learning policy
                # as the node states change over time, the information of previously selected nodes does not carry over to the next timestep
                # thus, we first need to reset the AL_mask to all False before updating it with the newly selected nodes
                AL_mask = torch.zeros(cfg.model.model.num_nodes, dtype=torch.bool, device=device)
                for node_id in AL_selected_nodes:
                    AL_mask[node_id] = True
                y_hat_masked = y_hat[AL_mask]
                y_hat_std_masked = y_hat_std[AL_mask]
                y_true_masked = y_true[AL_mask]
                loss += loss_fn(y_hat_masked, y_true_masked, y_hat_std_masked.square(), full=True)
                temporal_masks.append(AL_mask.float().detach().cpu())

                # Temporal smoothing regularizer across consecutive predictions
                if smooth_w > 0 and prev_y_hat is not None:
                    # Apply smoothing loss to encourage temporal consistency
                    temporal_smooth_loss = mse_fn(
                        y_hat.squeeze(), prev_y_hat.to(y_hat.device)
                    )
                    loss += smooth_w * temporal_smooth_loss

                # assess if it is time to update the model parameters based on the truncated BPTT length
                is_evaluation_step = ((local_time_of_sim + 1) % truncated_BPTT_length == 0) #and local_time_of_sim > 151)
                is_final_step = (local_time_of_sim == len(sim_indices[:-1]) - 1)

                # collect true incidence values for this timestep to be used for plotting
                true_incidence.append(y_true.detach().cpu().numpy())
                final_label_mask = AL_mask.clone()
                # parameter update
                if is_evaluation_step or is_final_step:
                    simulation_update_timesteps.append(local_time_of_sim + 1)
                    if is_evaluation_step:
                        average_loss = loss / truncated_BPTT_length
                    else:
                        average_loss = loss / ((local_time_of_sim + 1) % truncated_BPTT_length)
                    for _ in range(cfg.training.active_learning.max_epoch):
                        optimizer.zero_grad()
                        epoch_loss = torch.tensor(0.0, device=device)
                        epoch_hidden = (
                            None if saved_initial_hidden_states is None else [h.detach().clone()for h in saved_initial_hidden_states]
                        )
                        prev_epoch_y_hat = None
                        for item_idx, item in enumerate(window_snapshots):
                            snapshot = item["snapshot"]
                            epoch_input_mask = item["AL_mask"]
                            # Mask selected from this snapshot for its supervised loss.
                            if item_idx + 1 < len(window_snapshots):
                                epoch_label_mask = window_snapshots[item_idx + 1]["AL_mask"]
                            else:
                                epoch_label_mask = final_label_mask
                            y_hat, y_hat_std, epoch_hidden, _ = model(
                                snapshot.x,
                                snapshot.edge_index,
                                snapshot.edge_attr,
                                epoch_hidden,
                                external_mask=epoch_input_mask,
                            )
                            y_hat = y_hat.view(-1)
                            y_hat_std = y_hat_std.view(-1)
                            y_true = snapshot.y.view(-1)
                            epoch_loss += loss_fn(
                                y_hat[epoch_label_mask],
                                y_true[epoch_label_mask],
                                y_hat_std[epoch_label_mask].square(),
                                full=True,
                            )
                            if smooth_w > 0 and prev_epoch_y_hat is not None:
                                epoch_loss += smooth_w * mse_fn(
                                    y_hat.squeeze(),
                                    prev_epoch_y_hat,
                                )
                            prev_epoch_y_hat = y_hat.detach()
                        epoch_loss = epoch_loss / len(window_snapshots)
                        epoch_loss.backward()
                        # clipping gradients for training stability
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=cfg.model.clip_grad_max_norm,
                        )
                        optimizer.step()
                        lr_scheduler.step()
                    sim_train_losses.append((average_loss.item()))
                    print(f"AL deployment phase - timestep {local_time_of_sim}, train loss: {average_loss:.4f}, evaluation error: {evaluate_loss:.4f}")
                    print()
                    # recompute hidden states using the updated model weights after all the epochs, to continue AL deployment from the last timestep of the truncated BPTT window
                    model.eval()
                    with torch.no_grad():
                        hidden_states = (
                            None if saved_initial_hidden_states is None else [h.detach().clone()for h in saved_initial_hidden_states]
                        )
                        for item in window_snapshots:
                            snapshot = item["snapshot"]
                            epoch_input_mask = item["AL_mask"]
                            _, _, hidden_states, _ = model(
                                snapshot.x,
                                snapshot.edge_index,
                                snapshot.edge_attr,
                                hidden_states,
                                external_mask=epoch_input_mask,
                            )
                        hidden_states = [h.detach() for h in hidden_states] # Detach hidden state to break gradient flow 
                    window_snapshots = []
                    loss = torch.tensor(0.0, device=device)           
                prev_y_hat = y_hat.squeeze().detach()
            
            acc_train_loss.append(sim_train_losses)
            acc_eval_loss.append(sim_eval_losses)
            # collect uncertainty metrics for this simulation to be used for plotting
            if policy_name == "MC_Dropout_Variance" or policy_name == "BALSA_KL_Pairs" or policy_name == "BALSA_KL_Pairs_Variable":
                all_uncertainty_metrics.setdefault(f"{policy_name}_coverage", []).append(acc_uncertainty_metrics[f"{policy_name}_coverage"])
                all_uncertainty_metrics.setdefault(f"{policy_name}_width", []).append(acc_uncertainty_metrics[f"{policy_name}_width"])
                all_uncertainty_metrics.setdefault(f"{policy_name}_crps", []).append(acc_uncertainty_metrics[f"{policy_name}_crps"])
            # collect true incidence values to use for plotting 
            all_true_incidence.append(np.stack(true_incidence, axis=0))
            # save the predictions and true incidence values for this simulation to be used for the per-node plots
            if (
                policy_name in {
                    #"MC_Dropout_Variance",
                    "BALSA_KL_Pairs",
                    "BALSA_KL_Pairs_Variable",
                }
                and sim_plot_y_trues
            ):
                sim_y_trues = torch.stack(
                    sim_plot_y_trues,
                    dim=0,
                )
                sim_mc_means = torch.stack(
                    sim_plot_mc_means,
                    dim=0,
                )
                sim_mc_stds = torch.stack(
                    sim_plot_mc_stds,
                    dim=0,
                )
                temporal_masks = temporal_masks[:-1]
                sir_temporal_masks = torch.stack(temporal_masks, dim=0)
                policy_plot_y_trues.append(sim_y_trues)
                policy_plot_y_preds.append(sim_mc_means)
                policy_plot_y_stds.append(sim_mc_stds)
                policy_sim_num_steps.append(sim_y_trues.shape[0])
                policy_plot_masks.append(sir_temporal_masks)

        # Find and store median and interquartile (25th and 75th percentiles) range of training and evaluation losses across all simulations for this policy
        max_train_len = max(len(sim) for sim in acc_train_loss)

        median_train_loss = []
        p25_train_loss = []
        p75_train_loss = []

        for i in range(max_train_len):
            losses = [
                sim[i]
                for sim in acc_train_loss
                if i < len(sim)
            ]
            median_train_loss.append(float(np.median(losses)))
            p25_train_loss.append(float(np.percentile(losses, 25)))
            p75_train_loss.append(float(np.percentile(losses, 75)))

        max_eval_len = max(len(sim) for sim in acc_eval_loss)

        median_eval_loss = []
        p25_eval_loss = []
        p75_eval_loss = []

        for i in range(max_eval_len):
            losses = [
                sim[i]
                for sim in acc_eval_loss
                if i < len(sim)
            ]

            median_eval_loss.append(float(np.median(losses)))
            p25_eval_loss.append(float(np.percentile(losses, 25)))
            p75_eval_loss.append(float(np.percentile(losses, 75)))

        policy_train_histories[policy_name] = {
            "timesteps": simulation_update_timesteps,
            "median": median_train_loss,
            "p25": p25_train_loss,
            "p75": p75_train_loss,
        }

        eval_timesteps = list(range(1, max_eval_len + 1))
        policy_eval_histories[policy_name] = {
            "timesteps": eval_timesteps,
            "median": median_eval_loss,
            "p25": p25_eval_loss,
            "p75": p75_eval_loss,
        }

        # concatenate all simulations for a policy for per-node plots
        if (
            policy_name in {
                #"MC_Dropout_Variance",
                "BALSA_KL_Pairs",
            }
            and policy_plot_y_trues
        ):
            concatenated_y_trues = torch.cat(
                policy_plot_y_trues,
                dim=0,
            )
            concatenated_y_preds = torch.cat(
                policy_plot_y_preds,
                dim=0,
            )
            concatenated_y_stds = torch.cat(
                policy_plot_y_stds,
                dim=0,
            )
            concatenated_temporal_masks = torch.cat(
                policy_plot_masks,
                dim=0,
            )

            # Create visualizations for a subset of nodes to reduce memory usage
            max_nodes_to_plot = min(
                5,
                concatenated_y_preds.shape[1],
            )

            node_indices = torch.linspace(
                0,
                concatenated_y_preds.shape[1] - 1,
                max_nodes_to_plot,
                dtype=torch.long,
            )

            for node_id in node_indices:
                uncertainty_time_series_fig = (
                    create_plotly_uncertainty_time_series_plot(
                        y_true=concatenated_y_trues[:, node_id],
                        y_pred=concatenated_y_preds[:, node_id],
                        y_std=concatenated_y_stds[:, node_id],
                        temporal_mask=concatenated_temporal_masks[:, node_id],
                        sim_num_steps=policy_sim_num_steps,
                        node_id=int(node_id.item()),
                        policy_name=policy_name,
                    )
                )
                if cfg.seed == cfg.plot_seed and False:  # Disable logging to wandb to avoid clutter
                    wandb.log({
                        (
                            f"deployment/uncertainty_time_series/"
                            f"{policy_name}/node_{int(node_id.item())}"
                        ): uncertainty_time_series_fig,
                    })

    # mean_true_incidence_per_sim = []

    # for sim_incidence in all_true_incidence:
    #     masked_sim_incidence = np.where(
    #         np.abs(sim_incidence) > 0.01,
    #         sim_incidence,
    #         np.nan,
    #     )

    #     valid_mask = (
    #     np.abs(sim_incidence) > 0.01
    #     )

    #     valid_counts = valid_mask.sum(axis=1)

    #     incidence_sum = np.where(
    #         valid_mask,
    #         sim_incidence,
    #         0.0,
    #     ).sum(axis=1)

    #     mean_incidence = np.divide(
    #         incidence_sum,
    #         valid_counts,
    #         out=np.zeros_like(
    #             incidence_sum,
    #             dtype=float,
    #         ),
    #         where=valid_counts > 0,
    #     )

    #     mean_true_incidence_per_sim.append(
    #         mean_incidence
    #     )

    # max_time = max(
    #     len(sim_incidence)
    #     for sim_incidence in mean_true_incidence_per_sim
    # )

    # median_true_incidence = []
    # p25_true_incidence = []
    # p75_true_incidence = []

    # for i in range(max_time):
    #     incidences = [
    #         sim_incidence[i]
    #         for sim_incidence in mean_true_incidence_per_sim
    #         if i < len(sim_incidence)
    #         and not np.isnan(sim_incidence[i])
    #     ]

    #     median_true_incidence.append(float(np.median(incidences)))
    #     p25_true_incidence.append(float(np.percentile(incidences, 25)))
    #     p75_true_incidence.append(float(np.percentile(incidences, 75)))


    # incidence_history = {
    #     "True incidence": {
    #         "timesteps": eval_timesteps,
    #         "median": median_true_incidence,
    #         "p25": p25_true_incidence,
    #         "p75": p75_true_incidence,
    #     }
    # }


    # For each simulation, compute the median, 25th percentile,
    # and 75th percentile across nodes at every timestep.
    sim_node_medians = []
    sim_node_p25 = []
    sim_node_p75 = []

    for sim_incidence in all_true_incidence:
        # sim_incidence shape: [timesteps, num_nodes]

        sim_node_medians.append(
            np.median(sim_incidence, axis=1)
        )

        sim_node_p25.append(
            np.percentile(sim_incidence, 25, axis=1)
        )

        sim_node_p75.append(
            np.percentile(sim_incidence, 75, axis=1)
        )


    # Aggregate each statistic across simulations
    max_time = max(
        len(sim)
        for sim in sim_node_medians
    )

    median_true_incidence = []
    p25_true_incidence = []
    p75_true_incidence = []

    for i in range(max_time):

        # Median across simulations of the within-simulation node median
        median_true_incidence.append(
            float(np.median([
                sim[i]
                for sim in sim_node_medians
                if i < len(sim)
            ]))
        )

        # Median across simulations of the within-simulation node 25th percentile
        p25_true_incidence.append(
            float(np.median([
                sim[i]
                for sim in sim_node_p25
                if i < len(sim)
            ]))
        )

        # Median across simulations of the within-simulation node 75th percentile
        p75_true_incidence.append(
            float(np.median([
                sim[i]
                for sim in sim_node_p75
                if i < len(sim)
            ]))
        )


    incidence_history = {
        "True incidence": {
            "timesteps": list(range(1, max_time + 1)),
            "median": median_true_incidence,
            "p25": p25_true_incidence,
            "p75": p75_true_incidence,
        }
    }





    uncertainty_policy_names = [
        #"MC_Dropout_Variance",
        "BALSA_KL_Pairs",
        "BALSA_KL_Pairs_Variable",
    ]
    uncertainty_metric_names = [
        "coverage",
        "width",
        "crps",
    ]
    if "BALSA_KL_Pairs" in cfg.training.active_learning.policies: #"MC_Dropout_Variance" in cfg.training.active_learning.policies:
        for uncertainty_policy_name in uncertainty_policy_names:
            for uncertainty_metric_name in uncertainty_metric_names:
                metric_key = (
                    f"{uncertainty_policy_name}_{uncertainty_metric_name}"
                )

                metric_histories = all_uncertainty_metrics[metric_key]

                max_metric_len = max(
                    len(sim)
                    for sim in metric_histories
                )

                median_metric = []
                p25_metric = []
                p75_metric = []

                for i in range(max_metric_len):
                    values = [
                        sim[i]
                        for sim in metric_histories
                        if i < len(sim)
                    ]

                    median_metric.append(float(np.median(values)))
                    p25_metric.append(float(np.percentile(values, 25)))
                    p75_metric.append(float(np.percentile(values, 75)))

                plot_uncertainty_metrics[metric_key] = {
                    "policy_name": uncertainty_policy_name,
                    "metric_name": uncertainty_metric_name,
                    "timesteps": eval_timesteps[:max_metric_len],
                    "median": median_metric,
                    "p25": p25_metric,
                    "p75": p75_metric,
                }
    
    # # plot for training loss of all policies
    # train_comparison_fig = create_policy_comparison_plot(
    #     policy_train_histories,
    #     metric_name="Training loss",
    #     lookback_length=cfg.dataset.num_lookback_steps,
    # )

    # # plot for evaluation loss of all policies (exclude the ablated model)
    # eval_comparison_fig = create_policy_comparison_plot(
    #     {name: history for name, history in policy_eval_histories.items() if (name != "Ablated_Model")}, # and name != "Pretrained model")}
    #     metric_name="Next-timestep evaluation MSE",
    #     lookback_length=cfg.dataset.num_lookback_steps,
    # )

    # plot for mean true incidence across all simulations
    # incidence_fig = create_policy_comparison_plot(
    #     incidence_history,
    #     metric_name="Mean true incidence",
    #     lookback_length=cfg.dataset.num_lookback_steps,
    #     show_quantiles=False
    # )

    incidence_fig = create_policy_comparison_plot(
        incidence_history,
        metric_name="True incidence",
        lookback_length=cfg.dataset.num_lookback_steps,
        show_quantiles=True
    )

    # # plot for comparison of pretrained model, MC_Dropout_Variance policy, and BALSA_KL_Pairs policy
    # baseline_fig = create_policy_comparison_plot(
    #     {name: history for name, history in policy_eval_histories.items() if name in ["No_Fine_Tuning", 
    #                                                                                   "True_MSE_Oracle",
    #                                                                                    "Predicted_Incidence",
    #                                                                                    "True_MSE_Oracle_Variable"]},
    #     metric_name="Next-timestep evaluation MSE",
    #     lookback_length=cfg.dataset.num_lookback_steps,
    # )

    # plot comparison of full_model and ablated_model
    if ("Predicted_Incidence" in policy_eval_histories and "Ablated_Model" in policy_eval_histories):
        ablated_fig = create_policy_comparison_plot(
            {
                "Full graph model": policy_eval_histories[
                    "Predicted_Incidence"],
                "Ablated graph model": policy_eval_histories[
                    "Ablated_Model"],
            },
            metric_name="Next-timestep evaluation MSE",
            lookback_length=cfg.dataset.num_lookback_steps,
        )
        if cfg.seed == cfg.plot_seed and False:
            wandb.log({
                "deployment/full_vs_ablated/eval_mse": ablated_fig,
            })

    for metric_key, metric_history in plot_uncertainty_metrics.items():
        uncertainty_policy_name = metric_history["policy_name"]
        uncertainty_metric_name = metric_history["metric_name"]
        # uncertainty_fig = create_policy_comparison_plot(
        #     {
        #         uncertainty_policy_name: metric_history,
        #     },
        #     metric_name=uncertainty_metric_name,
        #     lookback_length=cfg.dataset.num_lookback_steps,
        #     show_quantiles=False,
        # )
        if cfg.seed == cfg.plot_seed and False:
            wandb.log({
                (
                    f"deployment/uncertainty_metrics/"
                    f"{uncertainty_policy_name}/{uncertainty_metric_name}"
                ): uncertainty_fig,
            })

    # results_table_fig = create_policy_results_table_plot(
    #     policy_eval_histories=policy_eval_histories,
    #     max_timestep=cfg.dataset.SIR.max_T,
    #     lookback_length=cfg.dataset.num_lookback_steps,
    #     timestep_spacing=50,
    # )
    if cfg.seed == cfg.plot_seed:
        wandb.log({
            #"deployment/policy_comparison/train_loss": train_comparison_fig,
            #"deployment/policy_comparison/eval_mse": eval_comparison_fig,
            "deployment/policy_comparison/mean_true_incidence": incidence_fig,
            #"deployment/pretrained_vs_mc_dropout/eval_mse": baseline_fig,
            #"deployment/results_table": results_table_fig,
        })

    return policy_eval_histories, plot_uncertainty_metrics


def evaluate_surrogate_model(AL_dataset, dataset_idx, hidden_states, AL_mask, model, device, cfg):
    """
    Evaluate the surrogate model on a test dataset. In particular, the function computes the MSE loss
    between the model's predictions and the true values for the timestep after the current timestep.

    Args:
        AL_dataset: The dataset to evaluate the model on.
        dataset_idx: The index of the snapshot in the dataset to evaluate the model on.
        hidden_states: The hidden states of the model from the previous timestep.
        AL_mask: The active learning mask from the current timestep, indicating which nodes were selected by the policy on the previous iteration.
        model: The surrogate model to be evaluated.
        device: The device to run the evaluation on.
        cfg: The configuration object containing model and training parameters.

    Returns:
        The average mean squared error (MSE) loss over the test dataset for the timestep specified by dataset_idx.
    """
    if hidden_states is None:
        hidden_states = [torch.zeros(cfg.model.model.num_nodes, cfg.model.model.gru_hidden_dim, device=device) for _ in range(cfg.model.model.num_layers)]
    model = model.to(device)
    model.eval()
    model.embed.make_test_mask_equal_to_train_mask()  # Ensure test mask is consistent with training mask
    eval_hidden_states = [h.detach().clone()for h in hidden_states]
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        snapshot = AL_dataset[dataset_idx].to(device)
        snapshot = snapshot.to(device)
        x_t = snapshot.x
        y_hat, _, hidden_states, _ = model(x_t, snapshot.edge_index, snapshot.edge_attr, eval_hidden_states, external_mask=AL_mask)
        y_hat = y_hat.view(-1)
        y_true = snapshot.y.view(-1)
        loss = mse_fn(y_hat.squeeze(), y_true).item()
    return loss


def evaluate_pretrained_baseline(AL_dataset, sim_ids_AL, model, pretrained_model_state, cfg, device):
    """
    Run the pretrained model over every simulation without active-learning
    acquisition or model fine-tuning, for comparison with the active learning deployment phase.
    Returns a history dictionary containing the median and interquartile range
    of the next-timestep evaluation MSE across simulations.
    """
    model.load_state_dict(
        copy.deepcopy(pretrained_model_state)
    )
    model.eval()
    all_sim_eval_losses = []
    update_timesteps = None
    truncated_BPTT_length = getattr(cfg.training.active_learning, 'truncated_BPTT_length', 1024)
    for sim_id in sorted(set(sim_ids_AL)):
        print(f"Starting AL deployment of pretrained baseline for simulation {sim_id}")
        # Reset to the same pretrained weights for each simulation
        model.load_state_dict(copy.deepcopy(pretrained_model_state))
        model.eval()
        model.embed.reset_train_temporal_masks()
        sim_indices = [i for i, sid in enumerate(sim_ids_AL)if sid == sim_id]
        hidden_states = None
        sim_eval_losses = []
        sim_update_timesteps = []
        with torch.no_grad():
            for local_time_of_sim, dataset_idx in enumerate(sim_indices[:-1]):
                snapshot = AL_dataset[dataset_idx].to(device)
                #print(f"Size of the snapshot: {snapshot.x.shape}")
                random_activation_mask = (torch.rand(snapshot.x.shape[0], device=device) < cfg.training.active_learning.num_nodes_to_test/snapshot.x.shape[0]).float()
                # Prediction only: no gradient and no weight update
                _, _, hidden_states, temporal_mask = model(snapshot.x,snapshot.edge_index,snapshot.edge_attr,hidden_states, external_mask=random_activation_mask)
                evaluation_loss = evaluate_surrogate_model(
                    AL_dataset=AL_dataset,
                    dataset_idx=dataset_idx,
                    hidden_states=hidden_states,
                    AL_mask=temporal_mask,
                    model=model,
                    device=device,
                    cfg=cfg
                )
                if (local_time_of_sim + 1) % truncated_BPTT_length == 0:
                    print(f"Pretrained baseline evaluation - timestep {local_time_of_sim}, evaluation error: {evaluation_loss:.4f}")
                sim_eval_losses.append(evaluation_loss)
                sim_update_timesteps.append(local_time_of_sim + 1)
                hidden_states = [h.detach() for h in hidden_states]
        all_sim_eval_losses.append(sim_eval_losses)
        if update_timesteps is None:
            update_timesteps = sim_update_timesteps
    median_eval_loss = []
    p25_eval_loss = []
    p75_eval_loss = []
    max_eval_len = max(len(sim_losses)for sim_losses in all_sim_eval_losses)
    for i in range(max_eval_len):
        losses = [
            sim_losses[i]
            for sim_losses in all_sim_eval_losses
            if i < len(sim_losses)
        ]
        median_eval_loss.append(float(np.median(losses)))
        p25_eval_loss.append(float(np.percentile(losses, 25)))
        p75_eval_loss.append(float(np.percentile(losses, 75)))
    return {
        "timesteps": update_timesteps,
        "median": median_eval_loss,
        "p25": p25_eval_loss,
        "p75": p75_eval_loss,
    }