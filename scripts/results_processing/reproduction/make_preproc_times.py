from ast import literal_eval

import pandas as pd
from constants import DATASET_ORDER, MODEL_ORDER

import wandb

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

def fetch(project):
    user = "telyatnikov_sap"
    project = project
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


def generate_table(df):
    df_best = df.copy()
    print(df_best)


    def build_table(subset_df, caption_text):
        if subset_df.empty:
            return (
                r"\begin{table}[ht]"
                "\n"
                r"\centering"
                "\n"
                r"\small"
                "\n"
                r"\textit{No data for this subset.}"
                "\n"
                r"\end{table}"
                "\n"
            )
        unique_datasets = subset_df["dataset"].unique()
        all_datasets = [d for d in DATASET_ORDER if d in unique_datasets]
        domain_groups = {}
        for _, row in subset_df.iterrows():
            dom = row["domain"]
            domain_groups.setdefault(dom, []).append(row)
        for dom in domain_groups:
            domain_groups[dom] = pd.DataFrame(domain_groups[dom])

        directions = {dset: "min" for dset in all_datasets}

        val = "max"
        best_vals_by_domain = {
            dom: {
                dset: dom_df[dom_df["dataset"] == dset][val].max()
                for dset in all_datasets
                if not dom_df[dom_df["dataset"] == dset].empty
            }
            for dom, dom_df in domain_groups.items()
        }

        def style_cell(mn, st, dset, dom):
            best_val = best_vals_by_domain[dom].get(dset, None)
            direction = directions[dset]

            content = f"\\scriptsize {mn:.0f} "

            if best_val is None:
                return content

            is_best = abs(mn - best_val) < 1e-12
            within_std = False

            if is_best:
                return f"\\cellcolor{{bestgray}}\\textbf{{{content}}}"
            elif within_std:
                return f"\\cellcolor{{stdblue}}{content}"
            else:
                return content

        latex_lines = []
        latex_lines.append(r"\begin{table}[ht]")
        latex_lines.append(r"\caption{" + caption_text + "}")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\small")

        col_spec = "@{}ll" + "c" * len(all_datasets) + "@{}"
        latex_lines.append(r"\begin{adjustbox}{width=1.\textwidth}")
        latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
        latex_lines.append(r"\toprule")

        header_cells = [r"& \textbf{Model}"]
        for dset in all_datasets:
            header_cells.append(r"\scriptsize " + dset)
        latex_lines.append(" & ".join(header_cells) + r" \\")

        all_domains = sorted(domain_groups.keys())
        if len(all_domains) == 3:
            all_domains = ["graph", "simplicial", "cell"]

        for dom in all_domains:
            dom_df = domain_groups[dom]
            dom_models = [m for m in MODEL_ORDER[dom] if m in dom_df["model"].unique()]
            latex_lines.append(r"\midrule")
            latex_lines.append(
                rf"\multirow{{{len(dom_models)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{dom.capitalize()}}}}}}}"
            )

            for model in dom_models:
                row_elems = [model]
                for dset in all_datasets:
                    sel = dom_df[(dom_df["model"] == model) & (dom_df["dataset"] == dset)]
                    if sel.empty:
                        cell_val = "-"
                    else:
                        r = sel.iloc[0]
                        mn, st = r["max"], r["max"]
                        cell_val = style_cell(mn, st, dset, dom)
                    row_elems.append(cell_val)
                latex_lines.append("& " + " & ".join(row_elems) + r" \\")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{adjustbox}")
        latex_lines.append(r"\end{table}")
        return "\n".join(latex_lines)

    # latex_mantra = build_table(df_mantra, "Topological datasets. Results are shown as mean and standard deviation. The best result is bold and shaded in grey, while those within one standard deviation are in blue-shaded boxes.")
    # latex_others = build_table(df_other, "Benchmarking datasets. Results are shown as mean and standard deviation. The best result is bold and shaded in grey, while those within one standard deviation are in blue-shaded boxes.")
    # return latex_mantra + "\n\n" + latex_others
    latex_all = build_table(df_best, "Total")
    return latex_all

def parse_hops():
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

    df.rename({
        "dataset.loader.parameters.data_name": "dataset",
        "model.model_name": "model",
        "model.model_domain": "domain",
    }, inplace=True, axis=1)

    df["model"] = df["model"].str.replace("hopse_m", "HOPSE-M")
    df["model"] = df["model"].str.replace("hopse_g", "HOPSE-G")
    df["dataset"] = df["dataset"].str.replace("MANTRA_name", "NAME", regex=False)
    df["dataset"] = df["dataset"].str.replace("MANTRA_orientation", "ORIENT", regex=False)
    df["dataset"] = df["dataset"].str.replace("MANTRA_betti_numbers_0", "MANTRA-BN-0", regex=False)
    df["dataset"] = df["dataset"].str.replace("MANTRA_betti_numbers_1", "MANTRA-BN-1", regex=False)
    df["dataset"] = df["dataset"].str.replace("MANTRA_betti_numbers_2", "MANTRA-BN-2", regex=False)
    df["dataset"] = df["dataset"].str.replace("MANTRA_betti_numbers", r"$(\beta_0, \beta_1, \beta_2)$", regex=False)
    df["model"] = df["model"].str.replace("_", "-", regex=False)

    grouped = df.groupby(["domain", "dataset", "model"], group_keys=True)
    df_best = grouped.agg({
        "preprocessor_time": ["max"]
    }).reset_index(drop=False)
    df_best["max"] = df_best[("preprocessor_time", "max")]
    df_best.drop([("preprocessor_time", "max")], axis=1, inplace=True)
    df_best.columns = [col if isinstance(col, str) else col[0] for col in df_best.columns]

    return df_best


if __name__ == "__main__":
    df = parse_hops()
    table = generate_table(df)
    print(table)
    #c_p, c_e, c_t = processing_times(df)
    # collected_results_time = generate_times_dictionary(df)
    # table = build_table(collected_results_time)
