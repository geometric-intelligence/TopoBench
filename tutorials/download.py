#!/usr/bin/env python3

import ast
import glob
import os
import pandas as pd
import wandb
from datetime import date

def download_wandb_runs(project_name, user, overwrite=False):
    """
    Downloads and saves Weights & Biases runs for a given project.
    
    Args:
        project_name (str): The wandb project name.
        user (str): The wandb username.
        overwrite (bool): Whether to overwrite existing CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing run summaries and configs.
    """
    today = date.today()
    api = wandb.Api()
    csv_file = f"{project_name}.csv"

    if not os.path.exists(csv_file) or overwrite:
        print(f"Downloading runs for project: {project_name}...")
        runs = api.runs(f"{user}/{project_name}")

        summary_list, config_list, name_list = [], [], []
        for run in runs:
            summary_list.append(run.summary._json_dict)
            config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
            name_list.append(run.name)

        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
        })

        runs_df.to_csv(csv_file)
        print(f"Saved to {csv_file}")
    else:
        print(f"Found existing CSV: {csv_file}")
        runs_df = pd.read_csv(csv_file, index_col=0)

        # Convert stringified dicts back to dictionaries
        runs_df["summary"] = runs_df["summary"].apply(ast.literal_eval)
        runs_df["config"] = runs_df["config"].apply(ast.literal_eval)

    return runs_df


if __name__ == "__main__":
    # Set your user and project name here
    #PROJECT_NAME = "HOPSE_M_cell_ablation_rebutal_MUTAG"
    for PROJECT_NAME in  ['rebuttal_cell_fix_CSL']:
        USER = "levsap" #"gbg141-hopse"

        # Set overwrite=True if you want to re-download and overwrite the CSV
        df = download_wandb_runs(project_name=PROJECT_NAME, user=USER, overwrite=False)
        
        print(df.head())  # Print a preview of the dataframe
