import warnings
from collections import defaultdict

import pandas as pd

from scripts.results_processing.constants import sweeped_columns


def generate_times_dictionary(df):
    # Identify unique models in DataFrame
    unique_models = df["model.model_name"].unique()

    # Identify unique datasets in DataFrame
    unique_datasets = df["dataset.loader.parameters.data_name"].unique()

    collected_results_time = defaultdict(dict)
    collected_results_time_run = defaultdict(dict)

    collected_non_aggregated_results = defaultdict(dict)

    # Got over each dataset and model and find the best result
    for dataset in unique_datasets:
        for model in unique_models:
            # Get the subset of the DataFrame for the current dataset and model
            subset = df[
                (df["dataset.loader.parameters.data_name"] == dataset)
                & (df["model.model_name"] == model)
            ]

            if subset.empty:
                print("---------")
                print(f"No results for {model} on {dataset}")
                print("---------")
                continue
            # Suppress all warnings
            warnings.filterwarnings("ignore")
            subset["Model"] = model
            warnings.filterwarnings("default")

            # def get_metric(df):
            #     metric_ = df["callbacks.early_stopping.monitor"].unique()
            #     assert len(metric_) == 1, "There should be only one metric to optimize"
            #     metric = metric_[0]
            #     return metric.split("/")[-1]

            # # Cols to get statistics later
            # # TODO: log maximum validation value for optimized metric
            # performance_cols = [f"test/{get_metric(subset)}"]

            # Get the unique values for each config column
            unique_colums_values = {}
            for col in sweeped_columns:
                try:
                    unique_colums_values[col] = subset[col].unique()
                except:
                    print(
                        f"Attention the columns: {col}, has issues with unique values"
                    )

            # Keep only those keys that have more than one unique value
            unique_colums_values = {
                k: v for k, v in unique_colums_values.items() if len(v) > 1
            }

            # Print the unique values for each config column

            # print(f"Unique values for each config column for {model} on {dataset}:")
            # for col, unique in unique_colums_values.items():
            #     print(f"{col}: {unique}")
            #     print()
            # print("---------")

            # Check if "special colums" are not in unique_colums_values
            # For example dataset.parameters.data_seed should not be in aggregation columns
            # If it is, then we should remove it from the list
            special_columns = ["dataset.parameters.data_seed"]

            for col in special_columns:
                if col in unique_colums_values:
                    unique_colums_values.pop(col)

            # Obtain the aggregation columns
            aggregation_columns = ["Model"] + list(unique_colums_values.keys())

            collected_non_aggregated_results[dataset][model] = {
                "df": subset.copy(),
                "aggregation_columns": aggregation_columns,
                # "performance_cols": performance_cols,
            }

            # Get average epoch run time
            collected_results_time[dataset][model] = {
                "mean": subset["AvgTime/train_epoch_mean"].mean(),
                "std": subset["AvgTime/train_epoch_mean"].std(),
            }

            collected_results_time_run[dataset][model] = {
                "mean": subset["_runtime"].mean(),
                "std": subset["_runtime"].std(),
            }
    return collected_results_time


def build_table(
    collected_results_time,
    selected_datasets=["MUTAG", "NCI1", "NCI109", "PROTEINS", "ZINC"],
):
    nested_dict = dict(collected_results_time)
    result_dict = pd.DataFrame.from_dict(
        {
            (i, j): nested_dict[i][j]
            for i in nested_dict
            for j in nested_dict[i].keys()
        },
        orient="index",
    )

    result_dict = result_dict.round(2)
    result_dict["performance"] = result_dict.apply(
        lambda x: f"{x['mean']} ± {x['std']}", axis=1
    )
    result_dict = result_dict.drop(["mean", "std"], axis=1)

    # Reset multiindex
    result_dict = result_dict.reset_index()
    # rename columns
    result_dict.columns = ["Dataset", "Model", "Average Time per Epoch"]

    table = result_dict.pivot_table(
        index="Model",
        columns="Dataset",
        values="Average Time per Epoch",
        aggfunc="first",
    )[selected_datasets]
    return table


if __name__ == "__main__":
    df = pd.read_csv("merged_csv/merged_normalized.csv")
    collected_results_time = generate_times_dictionary(df)
    table = build_table(collected_results_time)
    print(table)
