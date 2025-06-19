#!/bin/bash
export SCRATCH_ROOT=/home/xingnan/VideoDataInbox
export PROJECT_ROOT=/home/xingnan/scenario-dreamer
export DATASET_ROOT=$SCRATCH_ROOT
export CONFIG_PATH=$PROJECT_ROOT/cfgs
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Run preprocessing
cd $PROJECT_ROOT/data_processing/waymo

python generate_waymo_dataset_with_traffic.py \
  generate_waymo_dataset.mode=train \
  scratch_root=$SCRATCH_ROOT \
  dataset_root=$SCRATCH_ROOT

python preprocess_dataset_waymo.py \
  preprocess_waymo.mode=train \
  scratch_root=$SCRATCH_ROOT \
  dataset.waymo.dataset_path=$SCRATCH_ROOT/scenario_dreamer_waymo
