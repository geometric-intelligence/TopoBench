from ast import literal_eval

import pandas as pd


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


def split_evaluation_metrics(df):
    scores_df_list = []
    for i, row in df.iterrows():
        if (
            row["dataset.loader.parameters.data_name"]
            == "MANTRA_betti_numbers"
        ):
            rows = []
            for i in range(3):
                row_dict = row.to_dict()
                row_dict["dataset.loader.parameters.data_name"] = (
                    row_dict["dataset.loader.parameters.data_name"] + f"_{i}"
                )
                row_dict["val/f1"] = row_dict[f"val/f1_{i}"]
                row_dict["test/f1"] = row_dict[f"test/f1_{i}"]
                rows.append(pd.DataFrame.from_records([row_dict]))
            scores_df_list.append(pd.concat(rows))
    scores_df = pd.concat(scores_df_list)
    df = df[
        df["dataset.loader.parameters.data_name"] != "MANTRA_betti_numbers"
    ]
    return pd.concat([df, scores_df])


def preprocess_df(df, split_mantra=True):
    columns_to_eval = ["transforms.sann_encoding.pe_types"]
    for col in columns_to_eval:
        df[col] = df[col].apply(lambda x: str(x).replace("nan", "None"))
        df[col] = df[col].apply(literal_eval)
    df["model.model_name"] = df.apply(map_name, axis=1)
    df["transforms.sann_encoding.neighborhoods"] = df[
        "transforms.sann_encoding.neighborhoods"
    ].astype(str)
    if split_mantra:
        df = split_evaluation_metrics(df)
    # Remove rows with missing data_seed
    df = df[~(df["dataset.split_params.data_seed"].isna())]
    return df


if __name__ == "__main__":
    df = pd.read_csv("merged_csv/merged_normalized.csv", low_memory=False)
    preprocess_df(df)
