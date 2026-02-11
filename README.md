# Local Lane Graph Conditioning for Trajectory Prediction

<p align="left">
<a href="https://obsicat.com/lane-conditioning.html">
    <img src="https://img.shields.io/badge/Project_Page-obsicat.com-blue" /></a>
<a href="https://obsicat.com/assets/lane-conditioning-paper.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg?style=flat" /></a>
</p>

> **Local Lane Graph Conditioning as a General Inductive Bias for Trajectory Prediction: A Multi-Architecture Study on the Waymo Open Motion Dataset**
> [Xingnan Zhou](https://obsicat.com), Ciprian Alecsandru
> Department of Building, Civil and Environmental Engineering, Concordia University, Montreal
> Submitted to MDPI Sustainability, 2026

<p align="center">
  <img src="docs/project-page/assets/anim_scene_1_left_turn.gif" width="45%">
  <img src="docs/project-page/assets/anim_scene_0_straight.gif" width="45%">
</p>
<p align="center"><em>Left turn (minADE: 8.95m → 3.09m) and straight-through (4.35m → 1.19m) — Baseline (left) vs Lane-Conditioned (right)</em></p>

## Overview

We propose a **waterflow lane graph extraction** method that constructs a local, ego-centric lane topology through breadth-first traversal of the HD map, and a lightweight lane encoder with **graph message passing** and **cross-attention fusion**. The module is architecture-agnostic and improves both LSTM and Transformer backbones.

**Key results on 89,258 Waymo intersection scenarios:**

| Setting | Model | minADE | minFDE | MR@5m | vs Baseline |
|---------|-------|--------|--------|-------|-------------|
| 3s, single-modal | LSTM-LC | **0.507** | — | — | **+9.3%** |
| 8s, single-modal | LSTM-LC | **3.075** | **8.688** | — | **+18.7%** |
| 8s, single-modal | TF-LC | **3.303** | **8.956** | — | **+32.0%** |
| 8s, K=6 multi-modal | LSTM-LC | **1.337** | **3.289** | **19.4%** | **+27.3% / +33.7% / +42.7%** |

Our lane-conditioned LSTM achieves **minADE = 1.34m**, matching the [Waymo official LSTM baseline](https://arxiv.org/abs/2104.10133) that uses the full feature set — while using only 2D position + local lane features.

## Method

<p align="center">
  <img src="docs/project-page/assets/architecture.svg" width="80%">
</p>

1. **Waterflow Lane Graph Extraction** — 3-hop BFS from the ego lane, reducing graph size by ~80%
2. **Graph Message Passing** — 2 rounds of lane feature propagation along connectivity edges
3. **Cross-Attention Fusion** — Lane embeddings attend to trajectory features
4. **CV-Residual Decoder** — Predicts residuals relative to constant-velocity baseline; K=6 heads for multi-modal

The lane module adds only **+47K parameters (+7.5%)** for LSTM, achieving 27.3% minADE improvement.

## Repository Structure

```
├── models/                  # Model definitions
│   ├── lstm_baseline.py     # LSTM encoder-decoder baseline
│   ├── lane_conditioned_lstm.py  # LSTM + lane conditioning
│   ├── transformer_baseline.py   # Transformer baseline
│   ├── transformer_lane_cond.py  # Transformer + lane conditioning
│   ├── multimodal_lstm.py        # Multi-modal K=6 baseline
│   ├── multimodal_lane_cond.py   # Multi-modal + lane conditioning
│   └── flow_matching.py          # Flow matching (experimental)
├── training/                # Training infrastructure
│   ├── train.py             # Hydra-based entry point
│   ├── lightning_module.py  # PyTorch Lightning module
│   ├── multimodal_lightning_module.py
│   └── metrics.py           # ADE, FDE, MR evaluation
├── datasets/trajectory/     # Data loading
│   ├── traj_dataset.py      # Dataset with lane graph loading
│   └── lane_feature_utils.py
├── tools/                   # Core utilities
│   ├── lane_graph/          # Waterflow extraction & graph building
│   ├── scene_loader.py      # Unified scene loading
│   └── encoder/             # Trajectory & lane tokenization
├── configs/                 # Hydra configs
│   └── config.yaml          # Base configuration
├── docs/project-page/       # Project page (obsicat.com)
└── environment.yml          # Conda environment
```

## Setup

```bash
git clone https://github.com/Jynxzzz/scenario-dreamer-jynxzzz.git
cd scenario-dreamer-jynxzzz

conda env create -f environment.yml
conda activate scenario-dreamer
```

### Data

This project uses the [Waymo Open Motion Dataset v1.1.0](https://waymo.com/open/). We preprocess scenes into per-scenario pickle files containing trajectory data, lane graphs, and traffic light states.

```bash
# Preprocess Waymo TFRecords (requires waymo-open-dataset-tf)
bash data_processing/prepare_waymo_data_with_traffic_light.sh
```

## Training

```bash
# LSTM Baseline (8s, single-modal)
python training/train.py model.name=lstm_baseline data.future_len=80

# LSTM + Lane Conditioning (8s, single-modal)
python training/train.py model.name=lane_conditioned data.future_len=80

# Multi-Modal K=6 Baseline (8s)
python training/train.py model.name=multimodal_lstm_baseline data.future_len=80

# Multi-Modal K=6 + Lane Conditioning (8s)
python training/train.py model.name=multimodal_lane_cond data.future_len=80

# Transformer + Lane Conditioning (8s)
python training/train.py model.name=tf_lane_cond data.future_len=80
```

All training uses PyTorch Lightning with cosine annealing LR. We recommend **100 epochs** — the lane-conditioned model converges slower but reaches a lower asymptote.

## Error Decomposition

| Component | Baseline (m) | Lane-Cond (m) | Improvement |
|-----------|-------------|----------------|-------------|
| Avg Longitudinal | 1.238 | 0.924 | +25.4% |
| Avg Lateral | 0.919 | 0.675 | +26.5% |
| Endpoint Longitudinal | 3.561 | 2.577 | +27.6% |
| **Endpoint Lateral** | **2.687** | **1.867** | **+30.5%** |

Lane conditioning provides balanced improvements across both error axes, with endpoint lateral error (+30.5%) showing the strongest gain.

## Acknowledgements

This project builds on the [Scenario Dreamer](https://github.com/RLuke22/scenario-dreamer-waymo) framework (CVPR 2025) for Waymo data processing.

## Citation

```bibtex
@article{zhou2026lanegraph,
  title={Local Lane Graph Conditioning as a General Inductive Bias for Trajectory Prediction: A Multi-Architecture Study on the Waymo Open Motion Dataset},
  author={Zhou, Xingnan and Alecsandru, Ciprian},
  journal={Sustainability},
  year={2026},
  publisher={MDPI}
}
```
