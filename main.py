from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch.nn as nn
from tqdm import tqdm
import hydra
import torch
import random
import numpy as np
import logging
import copy
import wandb
from src.data_utils.preprocess_raw import convert_simulations_to_dataset

from src.data_utils.simulate_sir import get_multiple_sir_sims
from src.GCN_only import GCN_Model
from src.pre_training import train_surrogate_model, evaluate_surrogate_model
from src.visualization_utils import aggregate_policy_histories_across_seeds, create_policy_comparison_plot, create_policy_results_table_plot, aggregate_uncertainty_histories_across_seeds, create_pretrained_ablation_results_table, create_budget_allocation_plot
from src.data_utils.visualise import generate_outbreak_gifs_for_multiple_simulations
#from src.AL_dep_temporary import AL_deployment
from src.AL_deployment import AL_deployment
import os
from omegaconf import OmegaConf


def get_device(cfg):
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(cfg.device)

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default"
)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.data_seed)
    random.seed(cfg.data_seed)
    np.random.seed(cfg.data_seed)
    device = get_device(cfg)

    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.wandb_mode  # NOTE: disabled by default
    )
    wandb.define_metric("pretrain/epoch")
    wandb.define_metric("pretrain/*", step_metric="pretrain/epoch")

    wandb.define_metric("deployment/global_update")
    wandb.define_metric(
        "deployment/raw/*",
        step_metric="deployment/global_update",
    )

    # ==================================================================
    # Construct the dataset for the main experiment
    # ==================================================================

    print("Loading dataset...")
    sir_simulations, populations, edge_index, edge_attr = get_multiple_sir_sims(
        cfg
    )

    # Visualize raw SIR simulations (multiple) as GIFs over the mobility graph
    if cfg.dataset.visualise:
        out_dir = os.path.join("cache", f"outbreaks_{cfg.graph}") if hasattr(cfg, 'graph') else os.path.join("cache", "outbreaks")
        theta_csv_path = cfg.dataset.dataset.mobility_matrix_path
        print(f"Generating outbreak GIFs to: {out_dir}")
        generate_outbreak_gifs_for_multiple_simulations(
            simulations=sir_simulations,
            theta_csv_path=theta_csv_path,
            out_dir=out_dir,
            layout="spring",
            threshold=0.0
        )

    print("Converting simulations to dataset...")
    (
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
    ) = convert_simulations_to_dataset(
        sir_simulations, populations,
        edge_index, edge_attr, cfg
    )

    print("Instantiating model...")
    cfg.model.model.num_nodes = populations.shape[0] # NOTE: fix num_nodes to be consistent to graph info
    cfg.model.model.probability_of_selecting_nodes = (
        cfg.training.active_learning.num_nodes_to_test
        / cfg.model.model.num_nodes
    )

    # ==================================================================
    # Construct datasets for the temporal-ablation experiment
    # ==================================================================

    gcn_only_cfg = copy.deepcopy(cfg)
    # No temporal look-back
    gcn_only_cfg.dataset.num_lookback_steps = 1
    # The GCN-only model receives one timestep
    gcn_only_cfg.model.model.input_dim = 1
    print(
        "Constructing datasets for GCN-only temporal ablation..."
    )
    (
        dataset_train_gcn_only,
        _,
        sim_ids_train_gcn_only,
        dataset_test_gcn_only,
        _,
        sim_ids_test_gcn_only,
        _,
        _,
        _,
        simulation_params_gcn_only,
    ) = convert_simulations_to_dataset(
        sir_simulations,
        populations,
        edge_index,
        edge_attr,
        gcn_only_cfg,
    )

    # ---------------------------------------------------------
    # Run main experiment pipeline for multiple random seeds
    # ---------------------------------------------------------

    # Run multiple seeds for the entire pipeline (pre-training + active learning deployment)
    all_seed_policy_eval_histories = []
    all_seed_uncertainty_histories = []
    # Store pretrained-model test performance across seeds
    all_seed_full_pretrained_mse = []
    all_seed_ablated_pretrained_mse = []
    all_seed_gcn_only_pretrained_mse = []
    # Store pretraining train and eval MSE across seeds
    all_seed_pretraining_histories = []
    # store variable budget allocation across seeds
    all_seed_num_nodes_to_test_lists = []
    for seed_idx in [cfg.seed_to_run]:

        seed = seed_idx
        print()
        print("=" * 70)
        print(
            f"Running experimental seed "
            f"{seed + 1}/{cfg.num_seeds}: {seed}"
        )
        print("=" * 70)
        print()

        # ---------------------------------------------------------
        # Set experimental random seed
        # ---------------------------------------------------------
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cfg.seed = seed

        # ---------------------------------------------------------
        # Common initial model state for this seed
        # ---------------------------------------------------------
        initial_model = instantiate(
            cfg.model.model,
            device=device,
        ).to(device)

        common_initial_model_state = copy.deepcopy(
            initial_model.state_dict()
        )

        # ---------------------------------------------------------
        # Full model
        # ---------------------------------------------------------
        full_model = instantiate(
            cfg.model.model,
            device=device,
        ).to(device)

        full_model.load_state_dict(
            copy.deepcopy(
                common_initial_model_state
            )
        )

        full_optimizer = torch.optim.AdamW(
            full_model.parameters(),
            lr=1e-3,
            weight_decay=cfg.model.weight_decay,
        )

        full_lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                full_optimizer,
                T_max=cfg.training.pretraining.max_epoch,
                eta_min=1e-4,
            )
        )

        # ---------------------------------------------------------
        # Pretraining
        # ---------------------------------------------------------
        print(
            f"Starting pre-training for seed {seed}..."
        )

        full_model, num_nodes_to_test_list, train_loss_history, eval_mse_history = (
            train_surrogate_model(
                dataset_train_full,
                sim_ids_train,
                dataset_test_full,
                sim_ids_test,
                full_model,
                full_optimizer,
                full_lr_scheduler,
                simulation_params,
                cfg,
                device,
            )
        )

        all_seed_pretraining_histories.append({
            "Training Loss": {
                "median": train_loss_history,
            },
            "Evaluation MSE": {
                "median": eval_mse_history,
            },
        })

        all_seed_num_nodes_to_test_lists.append(
            num_nodes_to_test_list
        )

        # Evaluate the pretrained full model on the test set 
        (
            full_pretrained_mse,
            _,
            _,
            _,
        ) = evaluate_surrogate_model(
            dataset_test_full,
            sim_ids_test,
            full_model,
            simulation_params,
            device,
        )

        all_seed_full_pretrained_mse.append(
            float(full_pretrained_mse.item())
        )

        # ---------------------------------------------------------
        # Mobility network ablated model
        # ---------------------------------------------------------

        print(
            f"Starting ablated model training for seed {seed}..."
        )

        
        ablated_model = instantiate(
            cfg.model.model,
            device=device,
        ).to(device)

        ablated_model.load_state_dict(
            copy.deepcopy(
                common_initial_model_state
            )
        )

        ablated_optimizer = torch.optim.AdamW(
            ablated_model.parameters(),
            lr=1e-3,
            weight_decay=cfg.model.weight_decay,
        )

        ablated_pretraining_lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                ablated_optimizer,
                T_max=cfg.training.pretraining.max_epoch,
                eta_min=1e-4,
            )
        )

        ablated_model, _, _, _ = train_surrogate_model(
            dataset_train_ablated,
            sim_ids_train,
            dataset_test_ablated,
            sim_ids_test,
            ablated_model,
            ablated_optimizer,
            ablated_pretraining_lr_scheduler,
            simulation_params,
            cfg,
            device,
        )

        # evaluate the pretrained ablated model on the test set
        (
            ablated_pretrained_mse,
            _,
            _,
            _,
        ) = evaluate_surrogate_model(
            dataset_test_ablated,
            sim_ids_test,
            ablated_model,
            simulation_params,
            device,
        )

        all_seed_ablated_pretrained_mse.append(
            float(ablated_pretrained_mse.item())
        )


        # ==================================================================
        # GCN-only temporal ablation
        # ==================================================================

        print(
            f"Starting GCN-only temporal ablation for seed {seed}..."
        )

        gcn_only_model = GCN_Model(
            input_dim=1,
            embed_dim=cfg.model.model.embed_dim,
            gru_hidden_dim=cfg.model.model.gru_hidden_dim,
            gcn_hidden_dim=cfg.model.model.gcn_hidden_dim,
            output_dim=cfg.model.model.output_dim,
            probability_of_selecting_nodes=(
                cfg.model.model.probability_of_selecting_nodes
            ),
            num_nodes=cfg.model.model.num_nodes,
            return_temporal_mask=(
                cfg.model.model.return_temporal_mask
            ),
            device=device,
            dropout=cfg.model.model.dropout,
            num_layers=cfg.model.model.num_layers,
        ).to(device)

        gcn_only_optimizer = torch.optim.AdamW(
            gcn_only_model.parameters(),
            lr=1e-3,
            weight_decay=cfg.model.weight_decay,
        )

        gcn_only_lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                gcn_only_optimizer,
                T_max=(
                    cfg.training.pretraining.max_epoch
                ),
                eta_min=1e-4,
            )
        )

        gcn_only_model, _, _, _ = train_surrogate_model(
            dataset_train_gcn_only,
            sim_ids_train_gcn_only,
            dataset_test_gcn_only,
            sim_ids_test_gcn_only,
            gcn_only_model,
            gcn_only_optimizer,
            gcn_only_lr_scheduler,
            simulation_params_gcn_only,
            gcn_only_cfg,
            device,
            gcn_only=True,
        )

        (
            gcn_only_pretrained_mse,
            _,
            _,
            _,
        ) = evaluate_surrogate_model(
            dataset_test_gcn_only,
            sim_ids_test_gcn_only,
            gcn_only_model,
            simulation_params_gcn_only,
            device,
        )


        gcn_only_pretrained_mse = float(
            gcn_only_pretrained_mse.item()
        )

        all_seed_gcn_only_pretrained_mse.append(
            gcn_only_pretrained_mse
        )


        # ---------------------------------------------------------
        # Active-learning deployment
        # ---------------------------------------------------------
        full_deployment_lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                full_optimizer,
                T_max=cfg.training.active_learning.max_epoch,
                eta_min=1e-4,
            )
        )

        ablated_deployment_lr_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                ablated_optimizer,
                T_max=cfg.training.active_learning.max_epoch,
                eta_min=1e-4,
            )
        )

        print(
            f"Starting AL deployment for seed {seed}..."
        )

        policy_eval_histories, uncertainty_histories = AL_deployment(
            dataset_AL_full,
            dataset_AL_ablated,
            sim_ids_AL,
            full_model,
            ablated_model,
            full_optimizer,
            ablated_optimizer,
            full_deployment_lr_scheduler,
            ablated_deployment_lr_scheduler,
            simulation_params,
            num_nodes_to_test_list,
            cfg,
            device,
        )

        all_seed_policy_eval_histories.append(
            policy_eval_histories
        )

        all_seed_uncertainty_histories.append(
            uncertainty_histories
        )

        # # save results for this seed
        # seed_results = {
        #     "policy_eval_histories": policy_eval_histories,
        #     "uncertainty_histories": uncertainty_histories,

        #     "pretraining_histories": {
        #         "Training Loss": {
        #             "median": train_loss_history,
        #         },
        #         "Evaluation MSE": {
        #             "median": eval_mse_history,
        #         },
        #     },

        #     "full_pretrained_mse": float(
        #         full_pretrained_mse.item()
        #     ),

        #     "ablated_pretrained_mse": float(
        #         ablated_pretrained_mse.item()
        #     ),

        #     "gcn_only_pretrained_mse":
        #         gcn_only_pretrained_mse,
        # }

        # torch.save(
        #     seed_results,
        #     (
        #         "/content/drive/MyDrive/"
        #         "MSc_Thesis_Code/DynADS-main/"
        #         f"seed_{seed}_results.pt"
        #     ),
        # )

        # Free GPU memory before next seed
        del full_model
        del ablated_model
        del full_optimizer
        del ablated_optimizer
        del gcn_only_model
        del gcn_only_optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ==================================================================
    # Variable testing budget allocation across seeds
    # ==================================================================

    budget_array = np.asarray(
        all_seed_num_nodes_to_test_lists,
        dtype=float,
    )

    mean_num_nodes_to_test_list = np.mean(
        budget_array,
        axis=0,
    ).tolist()

    std_num_nodes_to_test_list = np.std(
        budget_array,
        axis=0,
        ddof=1,
    ).tolist()

    budget_allocation_fig = create_budget_allocation_plot(
            mean_num_nodes_to_test_list,
            std_num_nodes_to_test_list,
            cfg.dataset.num_lookback_steps,
    )

    wandb.log({
        (
            "final_across_seeds/"
            "variable_budget/"
            "budget_allocation"
        ): budget_allocation_fig,
    })
    

    # ==================================================================
    # Pertaining train and eval MSE across seeds of full model
    # ==================================================================

    aggregated_pretraining_histories = (
        aggregate_policy_histories_across_seeds(
            all_seed_pretraining_histories
        )
    )

    train_loss_fig = create_policy_comparison_plot(
        {
            "Training Loss":
                aggregated_pretraining_histories[
                    "Training Loss"
                ]
        },
        metric_name="Training Loss",
        lookback_length=1,
        xaxis_title="Epoch",
    )

    eval_mse_fig = create_policy_comparison_plot(
        {
            "Evaluation MSE":
                aggregated_pretraining_histories[
                    "Evaluation MSE"
                ]
        },
        metric_name="Evaluation MSE",
        lookback_length=1,
        xaxis_title="Epoch",
    )

    wandb.log({
        "final_across_seeds/pretraining/train_loss":
            train_loss_fig,

        "final_across_seeds/pretraining/eval_mse":
            eval_mse_fig,
    })

    # ==================================================================
    # Pretrained full vs ablated model comparison across seeds
    # ==================================================================

    mean_full_pretrained_mse = float(
        np.mean(all_seed_full_pretrained_mse)
    )

    mean_ablated_pretrained_mse = float(
        np.mean(all_seed_ablated_pretrained_mse)
    )

    mean_gcn_only_pretrained_mse = float(
            np.mean(
                all_seed_gcn_only_pretrained_mse
            )
        )

    pretrained_ablation_table_fig = (
        create_pretrained_ablation_results_table(
            full_model_mse=mean_full_pretrained_mse,
            ablated_model_mse=mean_ablated_pretrained_mse,
            gcn_only_model_mse=mean_gcn_only_pretrained_mse,
        )
    )

    wandb.log({
        (
            "final_across_seeds/"
            "pretrained_ablation/"
            "results_table"
        ): pretrained_ablation_table_fig,
    })

    # ==================================================================
    # Create final across-seed figures and tables
    # ==================================================================

    # aggregate policy evaluation histories across seeds for final analysis and visualization
    aggregated_policy_eval_histories = (
        aggregate_policy_histories_across_seeds(
            all_seed_policy_eval_histories
        )
    )

    aggregated_uncertainty_histories = (
        aggregate_uncertainty_histories_across_seeds(
            all_seed_uncertainty_histories
        )
    )


    # ------------------------------------------------------------------
    # Identify fixed-budget, adaptive-budget, and paired policies
    # ------------------------------------------------------------------
    fixed_policy_names = [
        policy_name
        for policy_name in aggregated_policy_eval_histories
        if (
            policy_name != "No_Fine_Tuning"
            and not policy_name.endswith("_Variable")
        )
    ]

    adaptive_policy_names = [
        policy_name
        for policy_name in aggregated_policy_eval_histories
        if policy_name.endswith("_Variable")
    ]

    paired_policy_names = [
        policy_name
        for policy_name in fixed_policy_names
        if (
            f"{policy_name}_Variable"
            in aggregated_policy_eval_histories
        )
    ]


    # ==================================================================
    # FIXED-BUDGET POLICIES + PRETRAINED MODEL
    # ==================================================================
    if fixed_policy_names:

        fixed_budget_histories = {
            "No_Fine_Tuning":
                aggregated_policy_eval_histories[
                    "No_Fine_Tuning"
                ]
        }

        for policy_name in fixed_policy_names:
            fixed_budget_histories[
                policy_name
            ] = aggregated_policy_eval_histories[
                policy_name
            ]

        fixed_budget_comparison_fig = (
            create_policy_comparison_plot(
                fixed_budget_histories,
                metric_name="Next-timestep evaluation MSE",
                lookback_length=(
                    cfg.dataset.num_lookback_steps
                ),
            )
        )

        fixed_budget_results_table_fig = (
            create_policy_results_table_plot(
                policy_eval_histories=(
                    fixed_budget_histories
                ),
                max_timestep=(
                    cfg.dataset.SIR.max_T
                ),
                lookback_length=(
                    cfg.dataset.num_lookback_steps
                ),
                timestep_spacing=50,
            )
        )

        wandb.log({
            (
                "final_across_seeds/"
                "fixed_budget/"
                "policy_comparison/eval_mse"
            ): fixed_budget_comparison_fig,

            (
                "final_across_seeds/"
                "fixed_budget/"
                "results_table"
            ): fixed_budget_results_table_fig,
        })


    # ==================================================================
    # ADAPTIVE-BUDGET POLICIES + PRETRAINED MODEL
    # ==================================================================
    if adaptive_policy_names:

        adaptive_budget_histories = {
            "No_Fine_Tuning":
                aggregated_policy_eval_histories[
                    "No_Fine_Tuning"
                ]
        }

        for policy_name in adaptive_policy_names:
            adaptive_budget_histories[
                policy_name
            ] = aggregated_policy_eval_histories[
                policy_name
            ]

        adaptive_budget_comparison_fig = (
            create_policy_comparison_plot(
                adaptive_budget_histories,
                metric_name="Next-timestep evaluation MSE",
                lookback_length=(
                    cfg.dataset.num_lookback_steps
                ),
            )
        )

        adaptive_budget_results_table_fig = (
            create_policy_results_table_plot(
                policy_eval_histories=(
                    adaptive_budget_histories
                ),
                max_timestep=(
                    cfg.dataset.SIR.max_T
                ),
                lookback_length=(
                    cfg.dataset.num_lookback_steps
                ),
                timestep_spacing=50,
            )
        )

        wandb.log({
            (
                "final_across_seeds/"
                "adaptive_budget/"
                "policy_comparison/eval_mse"
            ): adaptive_budget_comparison_fig,

            (
                "final_across_seeds/"
                "adaptive_budget/"
                "results_table"
            ): adaptive_budget_results_table_fig,
        })


    # ==================================================================
    # FIXED VS ADAPTIVE FOR EACH POLICY
    # ==================================================================
    for policy_name in paired_policy_names:

        adaptive_name = (
            f"{policy_name}_Variable"
        )

        comparison_histories = {
            "No_Fine_Tuning":
                aggregated_policy_eval_histories[
                    "No_Fine_Tuning"
                ],

            policy_name:
                aggregated_policy_eval_histories[
                    policy_name
                ],

            adaptive_name:
                aggregated_policy_eval_histories[
                    adaptive_name
                ],
        }

        comparison_fig = (
            create_policy_comparison_plot(
                comparison_histories,
                metric_name="Next-timestep evaluation MSE",
                lookback_length=(
                    cfg.dataset.num_lookback_steps
                ),
                color_overrides={
                    "No_Fine_Tuning": "#636EFA",
                    policy_name: "#EF553B",
                    adaptive_name: "#00CC96",
                },
            )
        )

        comparison_table_fig = (
            create_policy_results_table_plot(
                policy_eval_histories=(
                    comparison_histories
                ),
                max_timestep=(
                    cfg.dataset.SIR.max_T
                ),
                lookback_length=(
                    cfg.dataset.num_lookback_steps
                ),
                timestep_spacing=50,
            )
        )

        wandb.log({
            (
                "final_across_seeds/"
                f"fixed_vs_adaptive/"
                f"{policy_name}/eval_mse"
            ): comparison_fig,

            (
                "final_across_seeds/"
                f"fixed_vs_adaptive/"
                f"{policy_name}/results_table"
            ): comparison_table_fig,
        })

   
    # ==================================================================
    # Quality of uncertainty quantification plots
    # ==================================================================
    for (
        metric_key,
        metric_history,
    ) in aggregated_uncertainty_histories.items():

        policy_name = (
            metric_history["policy_name"]
        )

        metric_name = (
            metric_history["metric_name"]
        )

        uncertainty_fig = (
            create_policy_comparison_plot(
                {
                    policy_name:
                        metric_history
                },
                metric_name=metric_name,
                lookback_length=(
                    cfg.dataset.num_lookback_steps
                ),
                show_quantiles=True,
            )
        )

        wandb.log({
            (
                "final_across_seeds/"
                "uncertainty_metrics/"
                f"{policy_name}/"
                f"{metric_name}"
            ): uncertainty_fig,
        })



if __name__ == '__main__':
    main()
