from ast import literal_eval

import pandas as pd


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

if __name__ == "__main__":
    columns_to_normalize = [
        "model",
        "dataset",
        "transforms",
        "optimizer",
        "callbacks",
    ]
    df = pd.read_csv("merged_csv/merged.csv")
    df = normalize_df(df, columns_to_normalize)
    df.to_csv("merged_csv/merged_normalized.csv")
