# Realistic Disease Surveillance with Graph Neural Networks

This is the code repo that accompanies the thesis for the MSc in Advanced Computer Science at
the University of Oxford.

## Thesis Abstract

Effective disease surveillance is essential for informing public health interventions, yet allocating limited testing resources remains a fundamental challenge. Recent work has formulated disease surveillance as an active learning problem on graphs, using a conditional autoregressive (CAR) statistical surrogate model to adaptively allocate tests based on binary estimates of infection status. However, this approach relies on simplified assumptions about disease transmission that may limit its applicability to real-world outbreaks. The present thesis harnesses recent advances in deep learning to develop more realistic models for disease surveillance under limited resources. Specifically, it proposes a Graph Neural Network (GNN) based disease surveillance framework that combines graph structured data with recurrent components to model the spatio-temporal dynamics of infectious diseases. The GNN captures interactions between regions through a directed and weighted mobility network, while the recurrent component models the temporal evolution of disease spread. Together, these components enable the surrogate model to estimate disease incidence from sparse observations and guide the allocation of limited testing resources. By reducing reliance on hand-crafted transmission features, the proposed framework provides a more flexible and realistic approach to resource-constrained disease surveillance. Furthermore, this work extends the literature on variable-batch active learning by proposing a novel method for allocating a fixed total testing budget across the course of an epidemic. Specifically, the problem is reformulated as a classical problem in social choice theory, enabling the use of well-established apportionment methods for budget allocation. Finally, we demonstrate that this approach yields substantial improvements in predictive performance. 

## Repo Structure

```text
├── src/                                         # Core source code
│   ├── data_utils/                              # Constructing the dataset
│   │   ├── preprocess_raw.py                    # Construct PyG dataset from simulations
│   │   └── simulate_sir.py                      # Create epidemic simulations
│   ├── acquisition_funcs.py                     # AL acquisition functions
│   ├── adaptive_batch_allocation.py             # Variable batch allocation algorithm
│   ├── AL_deployment.py                         # Runs AL deployment phase
│   ├── GCN_only.py                              # Model for ablation
│   ├── GRU_GCN.py                               # Main surrogate model
│   ├── pre_training.py                          # Runs the pre-training phase
│   ├── quality_of_quantified_uncertainty.py     # Code for the metrics of QQU
│   ├── temporal_masking.py                      # Implements temporal masking mechanism
│   └── visualization_utils.py                   # Code for plots and tables
├── data/                                        # Raw mobility network
├── configs/                                     # Experiment configurations
├── requirements.txt                             # Dependencies
├── main.py                                      # Runs main
├── MSc_Thesis_Colab_Mount.ipynb                 # For running in Colab with GPUs
└── README.md
```     
