from ast import literal_eval

import numpy as np
import pandas as pd
from constants import (
    optimization_metrics,
)

import wandb

columns_to_normalize = [
    "model",
    "dataset",
    "transforms",
    "optimizer",
    "callbacks",
]
NBHD_ORDER = [
 "$\\scriptstyle A_{ 0 , 1 }$",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 }$",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 },A_{ 3,2 }$",
 "$\\scriptstyle A_{ 0 , 1 },I_{ 0 \\to 1 },I_{ 1 \\to 2 }$",
 "$\\scriptstyle A_{ 0 , 1 },I_{ 2 \\to 1 },I_{ 3 \\to 2 }$",
 "$\\scriptstyle A_{ 0 , 1 },I_{ 0 \\to 1 },I_{ 1 \\to 2 },$\\\\$\\scriptstyle I_{ 2 \\to 1 },I_{ 3 \\to 2 }$",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 },A_{ 2,1 },$\\\\$\\scriptstyle A_{ 3,2 },I_{ 0 \\to 1 },I_{ 1 \\to 2 },$\\\\$\\scriptstyle I_{ 2 \\to 1 },I_{ 3 \\to 2 }$",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 },A_{ 0 , 2 },$\\\\$\\scriptstyle A_{ 2,1 },A_{ 3,2 },A_{ 4,2 }$",
]

NB_MAP = {
 "$\\scriptstyle A_{ 0 , 1 }$": "Adj-1",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 }$": "Adj-2",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 },A_{ 3,2 }$": "Adj-3",
 "$\\scriptstyle A_{ 0 , 1 },I_{ 0 \\to 1 },I_{ 1 \\to 2 }$": "Inc-1",
 "$\\scriptstyle A_{ 0 , 1 },I_{ 2 \\to 1 },I_{ 3 \\to 2 }$": "Inc-2",
 "$\\scriptstyle A_{ 0 , 1 },I_{ 0 \\to 1 },I_{ 1 \\to 2 },$\\\\$\\scriptstyle I_{ 2 \\to 1 },I_{ 3 \\to 2 }$": "Inc-3",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 },A_{ 2,1 },$\\\\$\\scriptstyle A_{ 3,2 },I_{ 0 \\to 1 },I_{ 1 \\to 2 },$\\\\$\\scriptstyle I_{ 2 \\to 1 },I_{ 3 \\to 2 }$": "Mix-1",
 "$\\scriptstyle A_{ 0 , 1 },A_{ 1 , 2 },A_{ 0 , 2 },$\\\\$\\scriptstyle A_{ 2,1 },A_{ 3,2 },A_{ 4,2 }$": "Mix-2",
}
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

def fetch(proj):
    user = "telyatnikov_sap"
    project = proj
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

def main():
    df = fetch(
        "HOPSE_reproducibility_neighbourhoods")
    df = normalize_df(df, columns_to_normalize)
    df_2 = fetch(
        "HOPSE_reproducibility_neighborhoods"
    )
    df_2 = normalize_df(df_2, columns_to_normalize)
    df = pd.concat([df, df_2], ignore_index=True)
    #df = split_evaluation_metrics(df)
    df = df[~(df["dataset.split_params.data_seed"].isna())]
    df["transforms.sann_encoding.neighborhoods"] = df["transforms.sann_encoding.neighborhoods"].astype(str)

    datasets = df["dataset.loader.parameters.data_name"].unique()
    models = df["model.model_name"].unique()
    domains = df["model.model_domain"].unique()
    neighborhoods = df["transforms.sann_encoding.neighborhoods"].unique()

    df_dict = {
        "dataset": [],
        "model": [],
        "domain": [],
        "neighborhood": [],
        "score": [],
    }

    for dataset in datasets:
        for model in models:
            for domain in domains:
                for nbhd in neighborhoods:
                    if nbhd == "nan":
                        continue
                    subset = df[
                        (df["dataset.loader.parameters.data_name"] == dataset)
                        & (df["model.model_name"] == model)
                        & (df["model.model_domain"] == domain)
                        & (df["transforms.sann_encoding.neighborhoods"] == nbhd)
                    ]
                    if subset.empty:
                        # print("Empty subset")
                        # print(dataset, model, domain, nbhd)
                        # print("")
                        continue
                    str_nbhd = inc_list_to_name(nbhd)

                    mu = subset["AvgTime/val_epoch_mean"].mean()
                    sigma = subset["AvgTime/val_epoch_mean"].std()

                    # if eval_metric in ["test/accuracy", "test/f1"]:
                    #     mu *= 100
                    #     sigma *= 100
                    #     mu = round(mu, 2)
                    #     sigma = round(sigma, 2)
                    # if eval_metric in ["test/mse", "test/mae", "test/loss"]:
                    #     mu = round(mu, 4)
                    #     sigma = round(sigma, 4)
                        
                    df_dict["dataset"].append(dataset)
                    df_dict["model"].append(model)
                    df_dict["domain"].append(domain)
                    df_dict["neighborhood"].append(str_nbhd)
                    df_dict["score"].append(f"{mu}_{sigma}")
    df_nbhd = pd.DataFrame(df_dict)
    return df_nbhd


def parse_all_dfs(selected_datasets=[]):
    df_nbhd = main()

    # df_hopse = df_hopse[~df_hopse.isna()]
    # df_topotune = parse_topotune_results()
    # df_tb = parse_tb_results()
    # cat_df = pd.concat([df_hopse, df_topotune, df_tb], ignore_index=True)

    df_nbhd = df_nbhd[df_nbhd["dataset"].isin(["NCI1", "NCI109", "MUTAG", "PROTEINS", "ZINC", "MANTRA_name", "MANTRA_orientation", "MANTRA_betti_numbers"])]

    # Load your DataFrame
    # df = pd.read_csv("your_data.csv")

    # Replace _ with - in dataset and model ONLY
    df_nbhd["model"] = df_nbhd["model"].str.replace("hopse_m", "HOPSE-M")
    df_nbhd["model"] = df_nbhd["model"].str.replace("hopse_g", "HOPSE-G")
    df_nbhd["dataset"] = df_nbhd["dataset"].str.replace("MANTRA_name", "NAME", regex=False)
    df_nbhd["dataset"] = df_nbhd["dataset"].str.replace("MANTRA_orientation", "ORIENT", regex=False)
    df_nbhd["dataset"] = df_nbhd["dataset"].str.replace("MANTRA_betti_numbers_0", "MANTRA-BN-0", regex=False)
    df_nbhd["dataset"] = df_nbhd["dataset"].str.replace("MANTRA_betti_numbers_1", "MANTRA-BN-1", regex=False)
    df_nbhd["dataset"] = df_nbhd["dataset"].str.replace("MANTRA_betti_numbers_2", "MANTRA-BN-2", regex=False)
    df_nbhd["dataset"] = df_nbhd["dataset"].str.replace("MANTRA_betti_numbers", r"$(\beta_0, \beta_1, \beta_2)$", regex=False)
    df_nbhd["model"] = df_nbhd["model"].str.replace("_", "-", regex=False)
    return df_nbhd

def inc_list_to_name(inc_list):
    inc_list = eval(inc_list)
    inc_strs = []
    for i, inc in enumerate(inc_list):
        inc_num = inc.split("-")
        inc_val = int(inc_num[0]) if len(inc_num) == 3 else 1
        dim = int(inc_num[-1])

        key = ""
        if "incidence" in inc:
            if "up" in inc:
                key = f"I_{{ {dim} \\to {dim+inc_val} }}"
            elif "down" in inc:
                key = f"I_{{ {dim+inc_val} \\to {dim} }}"
            else:
                raise Exception("Unknown NHBD")
        elif "adjacency" in inc:
            key = "A_"
            if "up" in inc:
                key = f"A_{{ {dim} , {dim+inc_val} }}"
            elif "down" in inc:
                key  = f"A_{{ {dim+inc_val},{dim} }}"
            else:
                raise Exception("Unknown NHBD")
        else:
            raise Exception("Unknown NHBD")
        if i % 3 == 0 and i != 0:
            key = f"$\\\\$\\scriptstyle {key}"
        inc_strs.append(key)
    inc_name = rf"$\scriptstyle {','.join(inc_strs)}$"
    return inc_name

def generate_table(df_nbhd, optimization_metrics):
    # 1) Split your combined 'score' into numeric 'mean' and 'std'
    df_nbhd[["mean", "std"]] = df_nbhd["score"].str.split("_", expand=True).astype(float)

    for domain in df_nbhd["domain"].unique():
        df_domain = df_nbhd[df_nbhd["domain"] == domain].reset_index(drop=True)

        def compute_best(g):
            ds = g.name[1]   # g.name == (model, dataset)
            idx = g["mean"].idxmin()
            best_val = g["mean"].min()
            return pd.Series({
                "best_mean": best_val,
                "best_std":  g.loc[idx, "std"]
            })

        best_md = (
            df_domain
            .groupby(["model", "dataset"])
            .apply(compute_best)
            .reset_index()
        )


        df_domain = df_domain.merge(best_md, on=["model", "dataset"], how="left")

        # 3) LaTeX table setup
        datasets = df_domain["dataset"].unique()
        dataset_order = ["MUTAG", "PROTEINS", "NCI1", "NCI109", "ZINC", "NAME", "ORIENT", r"$(\beta_0, \beta_1, \beta_2)$"]
        model_order = ["HOPSE-M", "HOPSE-G"]
        column_format = r"{@{}" + "ll" + "c"*len([d for d in datasets if d in dataset_order]) + "@{}}"

        print(f"% LaTeX table for domain: {domain}")
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        print(r"\small")
        if domain == "simplicial":
            print(rf"\caption{{Runtime per epoch per neighborhood on \emph{{{domain}}}. Results are shown as mean and standard deviation. The best result per domain and per model is in \colorbox{{bestgray}}{{\textbf{{bold}}}} and results within one standard deviation of the lowest per domain are highlighted in \colorbox{{stdblue}}{{blue}}. We omit MANTRA-BN-0 as the accuracy is perfect on all neighborhoods.}}")
        else:
            print(rf"\caption{{Runtime per epoch per neighborhood on \emph{{{domain}}}. Results are shown as mean and standard deviation. The best result per domain and per model is in \colorbox{{bestgray}}{{\textbf{{bold}}}} and results within one standard deviation of the lowest per domain are highlighted in \colorbox{{stdblue}}{{blue}}.}}")
        print(f"\\label{{tab:{domain.lower()}_scores}}")
        print(r"\begin{adjustbox}{width=1.\textwidth}")
        print(r"\begin{tabular}" + column_format)
        print(r"\toprule")

        header_cells = [r"\textbf{Model}", r"\textbf{Nbhd}"] + [
            rf"{{\centering \scriptsize {col}}}" for col in dataset_order if col in datasets
        ]
        print(" & ".join(header_cells) + r" \\")
        print(r"\midrule")

        # 4) Print rows, highlighting per-model bests
        for cnt_model, model in enumerate([m for m in model_order if m in df_domain["model"].unique()]):
            sub = df_domain[df_domain["model"] == model]
            pivot = sub.pivot(index="neighborhood",
                            columns="dataset",
                            values=["mean", "std", "best_mean", "best_std"])

            first_row = True
            #pivot.sort_values(by=['neighborhood'], inplace=True, key=sort_nbhd_order)
            for nbhd in [n for n in NBHD_ORDER if n in pivot.index]:
                cnt = pivot.index.get_loc(nbhd)
                cell_strs = []
                for ds in dataset_order:
                    if ds in pivot.columns.levels[1]:
                        mean_val      = pivot.loc[nbhd, ("mean",      ds)]
                        std_val       = pivot.loc[nbhd, ("std",       ds)]
                        best_mean_val = pivot.loc[nbhd, ("best_mean", ds)]
                        best_std_val  = pivot.loc[nbhd, ("best_std",  ds)]

                        # base cell
                        txt = rf"\scriptsize {mean_val:.2f} $\pm$ {std_val:.2f}"
                        is_best   = np.isclose(mean_val, best_mean_val)
                        within_sd = (mean_val <= best_mean_val + best_std_val)

                        # apply highlighting
                        if is_best:
                            cell = rf"\cellcolor{{bestgray}}{{\textbf{{{txt}}}}}"
                        elif within_sd:
                            cell = rf"\cellcolor{{stdblue}}{{{txt}}}"
                        else:
                            cell = txt

                        cell_strs.append(cell)

                row_cells = " & ".join(cell_strs)

                if first_row:
                    span = len(pivot)
                    print(
                        rf"\multirow{{{span}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{model}}}}}}}"
                        + rf" & \emph{{{NB_MAP[nbhd]}}} & {row_cells} \\"
                    )
                    first_row = False
                else:
                    print(rf" & \emph{{{NB_MAP[nbhd]}}} & {row_cells} \\")
                if cnt < len(pivot.index) - 1:
                    print(r"\cmidrule(lr){2-" + str(2 + len([d for d in datasets if d in dataset_order])) + "}")
            # if cnt < len(pivot.index) - 1:
            #     print(r"\midrule")
            if cnt_model < len(df_domain["model"].unique()) - 1:
                print(r"\midrule")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{adjustbox}")
        print(r"\end{table}\n\n")



if __name__ == "__main__":
    # Define the datasets to include in the table
    selected_datasets = [
        "MUTAG",
        "PROTEINS",
        "NCI1",
        "NCI109",
        "ZINC",
        "MANTRA_orientation",
        "MANTRA_name",
        "MANTRA_betti_numbers_1",
        "MANTRA_betti_numbers_2",
    ]

    # Parse the dataframes
    df = parse_all_dfs(selected_datasets)
    #df.drop(['variant'], inplace=True, axis=1)
    # mask = (df['model'] == 'HOPSE-M') & (df['dataset'] == 'ZINC')

    # Generate the LaTeX table
    generate_table(df, optimization_metrics)
