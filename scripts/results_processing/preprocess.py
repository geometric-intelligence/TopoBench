import pandas as pd
from ast import literal_eval


def map_name(row):
    if isinstance(row["transforms.sann_encoding.pe_types"], list):
        return "HOPSE_MANUAL_PE"
    elif row["model.model_name"] == "sann":
        if type(
            row["transforms.sann_encoding.pretrain_model"]
        ) == float and pd.isna(row["transforms.sann_encoding.pretrain_model"]):
            return "SANN"
        else:
            return "HOPSE_GPSE"
    else:
        return row["model.model_name"]


def preprocess_df(df):
    columns_to_eval = ["transforms.sann_encoding.pe_types"]
    for col in columns_to_eval:
        df[col] = df[col].apply(lambda x: str(x).replace("nan", "None"))
        df[col] = df[col].apply(literal_eval)
    df["model.model_name"] = df.apply(map_name, axis=1)
    df["transforms.sann_encoding.neighborhoods"] = df[
        "transforms.sann_encoding.neighborhoods"
    ].astype(str)
    # Remove rows with missing data_seed
    df = df[~(df["dataset.split_params.data_seed"].isna())]
    return df


if __name__ == "__main__":
    df = pd.read_csv("merged_csv/merged_normalized.csv", low_memory=False)
    preprocess_df(df)
