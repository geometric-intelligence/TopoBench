from collections import defaultdict

from constants import (
    dataset_model_columns,
    optimization_metrics,
    run_columns,
    sweeped_columns,
)


def gen_scores(df):
    # Get unique datasets
    datasets = list(df["dataset.loader.parameters.data_name"].unique())
    # Get unique models
    models = list(df["model.model_name"].unique())

    collect_subsets = defaultdict(dict)
    # Got over each dataset and model and find the best result
    for dataset in datasets:
        for model in models:
            # Get the subset of the DataFrame for the current dataset and model
            subset = df.loc[
                (df["dataset.loader.parameters.data_name"] == dataset)
            ]

            optim_metric = optimization_metrics[dataset]["optim_metric"]
            eval_metric = optimization_metrics[dataset]["eval_metric"]
            direction = optimization_metrics[dataset]["direction"]

            # Keep metrics that matters for dataset
            performance_columns = optimization_metrics[dataset][
                "performance_columns"
            ]
            subset = subset[
                dataset_model_columns
                + sweeped_columns
                + performance_columns
                + run_columns
            ]
            aggregated = subset.groupby(
                sweeped_columns + ["model.model_name", "model.model_domain"],
                dropna=False,
            ).agg(
                {col: ["mean", "std", "count"] for col in performance_columns},
            )

            # aggregated = subset.groupby(sweeped_columns, dropna=False).count()

            # Go from MultiIndex to Index
            aggregated = aggregated.reset_index()
            print(f"Dataset: {dataset}, Model: {model}")
            print(aggregated[(eval_metric, "count")].unique())
            # print(aggregated['dataset.split_params.data_seed'].unique())
            print(
                (aggregated[(eval_metric, "count")] >= 4).sum()
                / len(aggregated)
                * 100
            )
            aggregated = aggregated[aggregated[(eval_metric, "count")] >= 4]
            # print(len(aggregated[aggregated['seed'] > 4]))
            aggregated = aggregated.sort_values(
                by=(optim_metric, "mean"), ascending=(direction == "min")
            )

            # Git percent in case of classification
            if "test/accuracy" in performance_columns:
                # Go over all the performance columns and multiply by 100
                for col in performance_columns:
                    aggregated[(col, "mean")] *= 100
                    aggregated[(col, "std")] *= 100

                # Round performance columns values up to 2 decimal points
                for col in performance_columns:
                    aggregated[(col, "mean")] = aggregated[
                        (col, "mean")
                    ].round(4)
                    aggregated[(col, "std")] = aggregated[(col, "std")].round(
                        4
                    )

            else:
                # Round all values up to 4 decimal points
                # Round performance columns values up to 4 decimal points
                for col in performance_columns:
                    aggregated[(col, "mean")] = aggregated[
                        (col, "mean")
                    ].round(4)
                    aggregated[(col, "std")] = aggregated[(col, "std")].round(
                        4
                    )

            collect_subsets[dataset] = aggregated
    return collect_subsets
