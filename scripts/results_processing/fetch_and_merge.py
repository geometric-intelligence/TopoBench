import ast
import glob
from datetime import date

import pandas as pd
import wandb


def fetch(user, main_csv_path, PROJ_PREFIX, PROJ_TYPES, PROJ_DS):
    today = date.today()
    api = wandb.Api(overrides={"base_url": "https://api.wandb.ai"}, timeout=40)

    # # Find all csv files in the current directory
    csv_files = glob.glob(f"{main_csv_path}/*.csv")
    # # Collect all the names of the csv files without the extension
    csv_names = [csv_file[4:-4] for csv_file in csv_files]

    for project_dataset in PROJ_DS:
        runs_df = None
        for project_type in PROJ_TYPES:
            if len(PROJ_PREFIX):
                project_name = (
                    f"{PROJ_PREFIX}_{project_type}_{project_dataset}"
                )
            else:
                project_name = f"{project_type}_{project_dataset}"
            project_name_plus_user = f"{user}_{project_name}"
            if project_name not in csv_names:
                print(project_name)
                runs = api.runs(f"{user}/{project_name}")
                # Project not found
                if runs == None:
                    print("No project found")
                    continue
                try:
                    list(runs)
                except Exception as e:
                    print("Exception:", e)
                    continue

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

                runs_df.to_csv(f"{main_csv_path}/{project_name}.csv")
            else:
                runs_df = pd.read_csv(f"csv/{project_name}.csv", index_col=0)

                for row in runs_df.iloc:
                    row["summary"] = ast.literal_eval(row["summary"])
                    row["config"] = ast.literal_eval(row["config"])


def merge(main_csv_path="csv", merge_csv_path="merged_csv"):
    csv_files = glob.glob(f"{main_csv_path}/*.csv")
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)

        # Parse JSON-like columns at once instead of row by row:
        df["summary"] = df["summary"].apply(ast.literal_eval)
        df["config"] = df["config"].apply(ast.literal_eval)

        # Merge the dicts in a vectorized way:
        merged_dicts = [
            {**s, **c} for s, c in zip(df["summary"], df["config"], strict=False)
        ]

        # Now expand them into a DataFrame:
        df_merged = pd.DataFrame.from_records(merged_dicts)

        df_list.append(df_merged)

    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(f"{merge_csv_path}/merged.csv")


if __name__ == "__main__":
    users = ["levsap", "telyatnikov_sap"]

    # Main experiments
    MAIN_EXP_DICT = {
        "PROJ_PREFIX": "main_exp",
        "PROJ_TYPES": ["GPSE", "SANN"],
        "PROJ_DS": [
            "NCI1",
            "NCI109",
            "MUTAG",
            "PROTEINS",
            "ZINC",
            "IMDB-BINARY",
            "IMDB-MULTI",
        ],
    }
    MAIN_EXP_SIMP_DICT = {
        "PROJ_PREFIX": "main_simplicial",
        "PROJ_TYPES": ["GPSE", "SANN"],
        "PROJ_DS": [
            "NCI1",
            "NCI109",
            "MUTAG",
            "PROTEINS",
            "ZINC",
            "IMDB-BINARY",
            "IMDB-MULTI",
        ],
    }
    HOPSE_EXP_DICT = {
        "PROJ_PREFIX": "HOPSE",
        "PROJ_TYPES": ["cell"],
        "PROJ_DS": [
            "NCI1",
            "NCI109",
            "MUTAG",
            "PROTEINS",
            "ZINC",
            "IMDB-BINARY",
            "IMDB-MULTI",
        ],
    }
    MANTRA_EXP_DICT = {
        "PROJ_PREFIX": "",
        "PROJ_TYPES": ["GRAPH", "SCCNN", "HOPSE_simplicial", "SCN", "GPSE", "SANN"],
        "PROJ_DS": ["mantra_name", "mantra_orientation", "mantra_betti_numbers"],
    }

    EXPERIMENTS = [
        MAIN_EXP_DICT,
        MAIN_EXP_SIMP_DICT,
        HOPSE_EXP_DICT,
        MANTRA_EXP_DICT,
    ]

    for user in users:
        for EXP in EXPERIMENTS:
            fetch(user, "csv", **EXP)

    merge("csv", "merged_csv")
