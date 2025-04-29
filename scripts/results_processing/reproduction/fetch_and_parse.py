import wandb
import pandas as pd
from ast import literal_eval

columns_to_normalize = [
    "model",
    "dataset",
    "transforms",
    "optimizer",
    "callbacks",
]
def normalize_columns(df, columns_to_normalize):
    # Gather the new DataFrames to be concatenated
    flattened_dfs = []

    for col in columns_to_normalize:
        df[col] = df[col].apply(lambda x: str(x).replace("nan", "None"))
        df[col] = df[col].apply(literal_eval)

        # Flatten
        flat = pd.json_normalize(df[col])

        # Rename
        flat.columns = [f"{col}.{c}" for c in flat.columns]
        flattened_dfs.append(flat)

    # Drop all nested columns in one shot
    df = df.drop(columns=columns_to_normalize)

    # Concatenate once at the end
    return pd.concat([df] + flattened_dfs, axis=1)


def normalize_df(df, columns_to_normalize):
    # Config columns to normalize
    df = normalize_columns(df, columns_to_normalize)

    return df

def fetch():
    user = "telyatnikov_sap"
    project = "HOPSE_reproducibility"
    api = wandb.Api(overrides={"base_url": "https://api.wandb.ai"}, timeout=40)
    runs = api.runs(f"{user}/{project}")
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {
                k: v
                for k, v in run.config.items()
                if not k.startswith("_")
            }
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
        }
    )
    # Merge the dicts in a vectorized way:
    merged_dicts = [
        {**s, **c}
        for s, c in zip(runs_df["summary"], runs_df["config"], strict=False)
    ]

    # Now expand them into a DataFrame:
    df_merged = pd.DataFrame.from_records(merged_dicts)
    return df_merged

def main():
    df = fetch()
    df = normalize_df(df, columns_to_normalize)
    return df

if __name__ == "__main__":
    main()