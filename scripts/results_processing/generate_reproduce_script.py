import pandas as pd
from constants import sweeped_columns, optimization_metrics
from generate_scores import gen_scores
from preprocess import preprocess_df


def generate(
    df, collect_subsets, sweeped_columns, all_seeds=[0, 3, 5, 7, 9], cpu=False
):
    datasets = list(df["dataset.loader.parameters.data_name"].unique())
    # Get unique models
    models = list(df["model.model_name"].unique())
    domains = list(df['model.model_domain'].unique())

    # 1) Dictionary mapping old model names to new ones (fill in as needed)
    model_name_mapping = {
        "HOPSE_MANUAL_PE": "sann",
        "HOPSE_GPSE": "sann",
        "HOPSE_GPSE_PCQM4MV2": "sann",
        "HOPSE_GPSE_GEOM": "sann",
        "HOPSE_GPSE_ZINC": "sann",
        "HOPSE_GPSE_MOLPCBA": "sann",
        "SANN": "sann",
    }


    with open("scripts/best_runs.sh", "w") as f:
        # Shebang so we can run `./best_runs.sh` directly if desired
        f.write("#!/usr/bin/env bash\n\n")
        f.write('# Define log files\n')
        f.write('LOG_FILE="scripts/script_output.log"\n')
        f.write('ERROR_LOG_FILE="scripts/script_error.log"\n')
        f.write('FAILED_LOG_FILE="scripts/failed_runs.log"\n\n')

        f.write('# Clear previous log files\n')
        f.write('> "$LOG_FILE"\n')
        f.write('> "$ERROR_LOG_FILE"\n')
        f.write('> "$FAILED_LOG_FILE"\n\n')

        f.write('# Function to run a command and check for failure\n')
        f.write('run_command() {\n')
        f.write('\tlocal cmd="$1"\n')
        f.write('\t# Run the command and capture the output and error\n')
        f.write('\t{ eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; } 2>> "$ERROR_LOG_FILE"\n')
        f.write('\t# Check if the command failed\n')
        f.write('\tif [ ${PIPESTATUS[0]} -ne 0 ]; then\n')
        f.write('\t\techo "Command failed: $cmd" >> "$FAILED_LOG_FILE"\n')
        f.write('\t\techo "Check $ERROR_LOG_FILE for details." >> "$FAILED_LOG_FILE"\n')
        f.write('\tfi\n')
        f.write('}\n\n')
        f.write('commands=(\n')

        for dataset in datasets:
            if 'IMDB' in dataset:
                continue
            # 'collect_subsets[dataset]' is the sorted, aggregated DataFrame
            aggregated = collect_subsets[dataset]
            direction = optimization_metrics[dataset]["direction"]
            optim_metric = optimization_metrics[dataset]["optim_metric"]

            for model in models:
                for domain in domains:
                    # Filter to rows for this model
                    model_agg = aggregated[
                        (aggregated["model.model_name"] == model)
                        &
                        (aggregated["model.model_domain"] == domain)]
                    if len(model_agg) == 0:
                        continue

                    model_agg = model_agg.sort_values(by=(optim_metric, 'mean'), ascending=(direction == "min"))
                    # Get the best row for this model
                    best_params_row = model_agg.iloc[0]

                    # -----------------------------------------------------------------
                    # EXTRACT AND CORRECT MODEL DOMAIN IF NEEDED
                    # If 'model.model_domain' is a single value, great; if it's a Series,
                    # we extract scalar. If it’s missing, adapt as needed.
                    # -----------------------------------------------------------------
                    model_domain_value = best_params_row["model.model_domain"]
                    if isinstance(model_domain_value, (pd.Series, pd.DataFrame)):
                        model_domain_value = model_domain_value.iloc[0]
                    # Convert to string if needed
                    model_domain_value = str(model_domain_value)

                    # -----------------------------------------------------------------
                    # REMAP THE MODEL NAME IF NEEDED
                    # We look in our mapping dict. If not found, fallback to original.
                    # -----------------------------------------------------------------
                    actual_model_name = model_name_mapping.get(model, model)

                    # -----------------------------------------------------------------
                    # BUILD A DICT OF BEST HYPERPARAMETERS
                    # For each column in `sweeped_columns`, we:
                    #   - Extract the scalar if needed
                    #   - Skip if NaN
                    #   - Convert based on type (int, float, list, or other => quotes)
                    # -----------------------------------------------------------------
                    best_params_dict = {}

                    for col in sweeped_columns:
                        if 'redefine' in col and (model == 'scn' or model == 'sccnn'):
                            continue
                        value = best_params_row[col]

                        # If it’s a Pandas object, extract the scalar
                        if (
                            isinstance(value, (pd.Series, pd.DataFrame))
                            and not value.empty
                        ):
                            value = value.iloc[0]

                        # Skip if NaN
                        if pd.isna(value):
                            continue

                        # Type-checking for Hydra-friendly output
                        if isinstance(value, int):
                            param_val = str(value)  # keep integer as plain number
                        elif isinstance(value, float):
                            if col.startswith("optimizer.parameters") and (
                                "proj_dropout" in col
                                or "lr" in col
                                or "weight_decay"
                            ):
                                param_val = str(value)
                            else:
                                param_val = str(
                                    int(value)
                                )  # float as plain number
                        elif isinstance(value, (list, tuple)):
                            # Convert list or tuple to a quoted string
                            param_val = f'{",".join(value)}'
                        else:
                            # For strings (or anything else), quote them
                            param_val = f'{str(value).replace(" ", "")}'

                        best_params_dict[col] = param_val

                    # -----------------------------------------------------------------
                    # DATASET OVERRIDE AND MODEL OVERRIDE
                    #  e.g. "dataset=graph/MNIST"  or "model=gnn/GAT"
                    # Depending on your logic, we choose domain if 'MANTRA' in dataset
                    # -----------------------------------------------------------------
                    data_domain = "simplicial" if "MANTRA" in dataset else "graph"
                    # We actually use sccnn_custom since TopoModelX implementation is incorrect
                    actual_model_name = actual_model_name + '_custom' if actual_model_name == 'sccnn' else actual_model_name
                    dataset_name_str = dataset.lower() if 'MANTRA' in dataset else dataset
                    dataset_str = f"dataset={data_domain}/{dataset_name_str}"
                    model_str = f"model={model_domain_value}/{actual_model_name}"

                    # -----------------------------------------------------------------
                    # CONVERT HYPERPARAMETERS TO key=value STRINGS
                    # e.g. "transforms.R.loops='2'" or "model.backbone.num_layers='3'"
                    # -----------------------------------------------------------------
                    if 'sccnn' == model or 'scn' == model:
                        if "transforms.redefine_simplicial_neighborhoods.signed" in best_params_dict:
                            del best_params_dict[
                                "transforms.redefine_simplicial_neighborhoods.signed"
                            ]
                        if "transforms.redefine_simplicial_neighborhoods.neighborhoods" in best_params_dict:
                            del best_params_dict[
                                "transforms.redefine_simplicial_neighborhoods.neighborhoods"
                            ]
                    param_strs = []

                    for key, val in best_params_dict.items():
                        # Value is nan so not set
                        if pd.isna(val):
                            continue
                        if val == 'nan':
                            continue
                        if 'neighbourhood' in key:
                            key = key.replace('neighbourhood', 'neighborhood')
                        param_strs.append(f"{key}={val}")

                    additional_parameters = {}
                    unset_parameters = {}

                    dataset_additional_parameters = {}
                    if dataset == "ZINC":
                        dataset_additional_parameters[
                            "transforms.one_hot_node_degree_features.degrees_field"
                        ] = "x"
                        dataset_additional_parameters[
                            "transforms.one_hot_node_degree_features.features_field"
                        ] = "x"
                    # TODO Change to TRUE
                    elif 'MANTRA' in dataset:
                        dataset_additional_parameters[
                            'dataset.loader.parameters.slice'
                        ] = False
                    if 'betti_numbers' in dataset:
                        dataset_additional_parameters["evaluator"] = "betti_numbers"


                    if "GPSE" in model:
                        additional_parameters[
                            "transforms/data_manipulations@transforms.sann_encoding"
                        ] = "add_gpse_information"
                        additional_parameters[
                            "transforms.sann_encoding.copy_initial"
                        ] = True
                        if model_domain_value == "cell" and data_domain == "graph":
                            additional_parameters[
                                "transforms.graph2cell_lifting.neighborhoods"
                            ] = best_params_dict[
                                "transforms.sann_encoding.neighborhoods"
                            ]
                        elif model_domain_value == "simplicial" and data_domain == "graph":
                            additional_parameters[
                                "transforms.graph2simplicial_lifting.neighborhoods"
                            ] = best_params_dict[
                                "transforms.sann_encoding.neighborhoods"
                            ]
                        if model_domain_value == 'simplicial' and data_domain == "simplicial":
                            assert model != 'sccnn'
                            additional_parameters[
                                "transforms.redefine_simplicial_neighborhoods.neighborhoods"
                            ] = best_params_dict['transforms.sann_encoding.neighborhoods']
                            additional_parameters[
                                "transforms.redefine_simplicial_neighborhoods.signed"
                            ] = True

                    elif "HOPSE" in model:
                        additional_parameters[
                            "transforms/data_manipulations@transforms.sann_encoding"
                        ] = "hopse_ps_information"
                        if model_domain_value == "cell" and data_domain == "graph":
                            additional_parameters[
                                "transforms.graph2cell_lifting.neighborhoods"
                            ] = best_params_dict[
                                "transforms.sann_encoding.neighborhoods"
                            ]
                        elif model_domain_value == "simplicial" and data_domain == "graph":
                            additional_parameters[
                                "transforms.graph2simplicial_lifting.neighborhoods"
                            ] = best_params_dict[
                                "transforms.sann_encoding.neighborhoods"
                            ]

                        if model_domain_value == 'simplicial' and data_domain == "simplicial":
                            assert model != 'sccnn'
                            additional_parameters[
                                "transforms.redefine_simplicial_neighborhoods.neighborhoods"
                            ] = best_params_dict['transforms.sann_encoding.neighborhoods']
                            additional_parameters[
                                "transforms.redefine_simplicial_neighborhoods.signed"
                            ] = True
                        additional_parameters[
                            "transforms.sann_encoding.kernel_param_HKdiagSE"
                        ] = "[1,22]"
                        additional_parameters[
                            "transforms.sann_encoding.kernel_param_RWSE"
                        ] = "[2,20]"

                        additional_parameters[
                            "transforms.sann_encoding.laplacian_norm_type"
                        ] = "sym"
                        additional_parameters[
                            "transforms.sann_encoding.posenc_LapPE_eigen_max_freqs"
                        ] = 18
                        additional_parameters[
                            "transforms.sann_encoding.posenc_LapPE_eigen_eigvec_norm"
                        ] = "L2"
                        additional_parameters[
                            "transforms.sann_encoding.posenc_LapPE_eigen_skip_zero_freq"
                        ] = True
                        additional_parameters[
                            "transforms.sann_encoding.posenc_LapPE_eigen_eigvec_abs"
                        ] = True
                        additional_parameters[
                            "transforms.sann_encoding.target_pe_dim"
                        ] = 20
                        additional_parameters[
                            "transforms.sann_encoding.pe_types"
                        ] = '["RWSE","ElstaticPE","HKdiagSE","LapPE"]'
                    elif "SANN" in model:
                        additional_parameters[
                            "transforms/data_manipulations@transforms.sann_encoding"
                        ] = "precompute_khop_features"
                        if 'MANTRA' in dataset:
                            unset_parameters[
                                "transforms/data_manipulations@transforms.redefine_simplicial_neighborhoods"
                            ] = True


                    additional_param_strs = [
                        f"{key}={val}"
                        for key, val in additional_parameters.items()
                    ]
                    dataset_additional_param_strs = [
                        f"{key}={val}"
                        for key, val in dataset_additional_parameters.items()
                    ]
                    unset_params_strs = [
                        f"~{key}"
                        for key, val in unset_parameters.items()
                    ]

                    # -----------------------------------------------------------------
                    # GENERATE THE COMMAND
                    # We pass multiple seeds via multirun
                    # dataset.split_params.data_seed=0,1,2,...  plus --multirun
                    # -----------------------------------------------------------------
                    cpu_str = (
                        "trainer.devices=\\[6\\]"
                        if not cpu
                        else "trainer.accelerator=cpu trainer.devices=1"
                    )
                    trainer_epochs_str =  "trainer.max_epochs=500 trainer.min_epochs=50 trainer.check_val_every_n_epoch=5"
                    trainer_patience_str = "callbacks.early_stopping.patience=10"
                    model_tag = f"logger.wandb.tags=[\"{model}\",\"{domain}\"]"
                    wandb_str = "logger.wandb.project=HOPSE_reproducibility" if not cpu else "logger.wandb.project=HOPSE_test"
                    cmd = (
                        "\'python -m topobench "
                        + dataset_str
                        + " "
                        + model_str
                        + " "
                        + " ".join(param_strs)
                        + " "
                        + " ".join(dataset_additional_param_strs)
                        + " "
                        + " ".join(additional_param_strs)
                        + " "
                        + f"dataset.split_params.data_seed={','.join([str(i) for i in all_seeds])}"
                        + " "
                        + " ".join(unset_params_strs)
                        + " "
                        + trainer_epochs_str
                        + " "
                        + cpu_str
                        + " "
                        + trainer_patience_str
                        + " "
                        + model_tag
                        + " "
                        + wandb_str
                        + " " 
                        + "--multirun\'"
                        + f" # {model},{dataset},{data_domain}"
                    )

                    f.write(cmd + "\n")
        f.write(')\n\n')
        f.write('# Iterate over the commands and run them\n')
        f.write('for cmd in "${commands[@]}"; do\n')
        f.write('\techo "Running: $cmd"\n')
        f.write('\trun_command "$cmd"\n')
        f.write('done\n')

    print("Done! The best runs have been saved to scripts/best_runs.sh.")


if __name__ == "__main__":
    # Load merged normalized
    df = pd.read_csv("merged_csv/merged_normalized.csv")
    df = preprocess_df(df, split_mantra=False)
    # Keep only relevant columns
    # df = df[keep_columns]
    # Generate best scores per hyperparameter sweep
    scores = gen_scores(df)

    # Generate the best runs script
    generate(df, scores, sweeped_columns, cpu=False)
