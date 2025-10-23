#!/bin/bash
# ======================
# USER CONFIGURATION
# ======================
project_name='ml_on_full_graph_extended'
exp_name="ml"   # Set your experiment name here

models_classification=("dt_c" "hgb_c" "lgbm_c" "logistic_regression" "mlp_c" "xgb_c", "rf_c")
models_regression=("rf_r" "dt_r" "hgb_r" "lgbm_r" "linear_regression" "mlp_r" "xgb_r")

# --- Classification (binary) ---
datasets_classification_binary=(
  "graph/tolokers"
  "graph/tolokers-2"
  "graph/minesweeper"
  "graph/questions"
  "graph/web-fraud"
  "graph/artnet-exp"
  "graph/city-reviews"
)

# --- Classification (multiclass) ---
datasets_classification_multiclass=(
  "graph/roman_empire"
  "graph/amazon_ratings"
  "graph/cocitation_citeseer"
  "graph/cocitation_cora"
  "graph/cocitation_pubmed"
  "graph/hm-categories"
  "graph/pokec-regions"
  "graph/web-topics"
  "graph/wiki_cs"
)

# --- Regression ---
datasets_regression=(
  "graph/artnet-views"
  "graph/avazu-ctr"
  "graph/city-roads-L"
  "graph/city-roads-M"
  "graph/hm-prices"
  "graph/web-traffic"
  "graph/twitch-views"
  "graph/US-BachelorRate"
  "graph/US-BirthRate"
  "graph/US-DeathRate"
  "graph/US-Election"
  "graph/US-MedianIncome"
  "graph/US-MigraRate"
  "graph/US-UnemploymentRate"
)


seeds="0,1,2,3,4"

# ---- CLASSIFICATION: BINARY ----
for d in "${datasets_classification_binary[@]}"; do
  for m in "${models_classification[@]}"; do
    echo "Running BINARY classification dataset=$d model=$m"
    python -m topobench -m \
      dataset=$d \
      evaluator=classification_extended \
      model=non_relational/sklearn_classifier \
      model/non_relational/sklearn@model.backbone=$m \
      model/non_relational/sklearn/samplers@model.backbone_wrapper.sampler=graph_hop \
      train=False \
      dataset.split_params.data_seed=$seeds \
      logger.wandb.project=$project_name \
      trainer=cpu \
      loss=no_loss
  done
done



# ---- CLASSIFICATION: MULTICLASS ----
for d in "${datasets_classification_multiclass[@]}"; do
  for m in "${models_classification[@]}"; do
    echo "Running MULTICLASS classification dataset=$d model=$m"
    python -m topobench -m \
      dataset=$d \
      evaluator=classification_extended \
      model=non_relational/sklearn_classifier \
      model/non_relational/sklearn@model.backbone=$m \
      model/non_relational/sklearn/samplers@model.backbone_wrapper.sampler=graph_hop \
      train=False \
      dataset.split_params.data_seed=$seeds \
      logger.wandb.project=$project_name \
      trainer=cpu \
      loss=no_loss
  done
done

# ---- REGRESSION ----
for d in "${datasets_regression[@]}"; do
  for m in "${models_regression[@]}"; do
    echo "Running REGRESSION dataset=$d model=$m"
    python -m topobench -m \
      dataset=$d \
      evaluator=regression_extended \
      model=non_relational/sklearn_regressor \
      model/non_relational/sklearn@model.backbone=$m \
      model/non_relational/sklearn/samplers@model.backbone_wrapper.sampler=graph_hop \
      train=False \
      dataset.split_params.data_seed=$seeds \
      logger.wandb.project=$project_name \
      trainer=cpu \
      loss=no_loss
  done
done

