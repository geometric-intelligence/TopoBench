import pandas as pd
from constants import (
    DATASET_ORDER,
    MODEL_ORDER,
    optimization_metrics,
)
from fetch_and_parse import main

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

def parse_gnn_results(df, selected_datasets):
    df_dict = {
        "model": [],
        "pe_type": [],
        "dataset": [],
        "mean": [],
        "std": [],
    }

    df = split_evaluation_metrics(df)

    models = df["model.model_name"].unique()
    datasets = df["dataset.loader.parameters.data_name"].unique()
    pe_types = df["transform.sann_encodigs.pe_types"].unique()


    for dataset in datasets:
        for model in models:
            for pe_type in pe_types:
                subset = df[
                    (df["dataset.loader.parameters.data_name"] == dataset)
                    & (df["model.model_name"] == model)
                    & (df["transform.sann_encodigs.pe_types"] == pe_type)
                ]
                if subset.empty:
                    print(dataset, model, pe_type)
                    continue
                eval_metric = optimization_metrics[dataset]["eval_metric"]

                if eval_metric in ["test/accuracy", "test/f1"]:
                    subset[eval_metric] *= 100

                subset[eval_metric] = subset[
                    eval_metric
                ].round(4)


                df_dict["pe_type"].append(pe_type)
                df_dict["model"].append(model)
                df_dict["dataset"].append(dataset)
                df_dict["mean"].append(subset[eval_metric].mean())
                df_dict["std"].append(subset[eval_metric].std())
    df_res = pd.DataFrame(df_dict)
    return df_res



def parse_all_dfs(selected_datasets=[]):
    dfs = []
    for ds in ['MUTAG', 'PROTEINS', 'NCI1', 'NCI109', 'ZINC']:
        df = main(user='levsap', project=f'rebuttal_cell_{ds}')
        dfs.append(df)
    df = pd.concat(dfs)

    df_gen = parse_gnn_results(df, selected_datasets)
    #mask = (df_hopse["model"] == "sann") & (df_hopse["dataset"] == "ZINC")

    df_gen = df_gen[~df_gen.isna()]

    filtered_df = df_gen[df_gen["dataset"].isin(selected_datasets)]

    # FIX MAntra naming
    filtered_df.loc[
        filtered_df["dataset"] == "MANTRA_betti_numbers_0", "dataset"
    ] = "MANTRA-BN-0"
    filtered_df.loc[
        filtered_df["dataset"] == "MANTRA_betti_numbers_1", "dataset"
    ] = "MANTRA-BN-1"
    filtered_df.loc[
        filtered_df["dataset"] == "MANTRA_betti_numbers_2", "dataset"
    ] = "MANTRA-BN-2"
    filtered_df.loc[filtered_df["dataset"] == "MANTRA_name", "dataset"] = (
        "MANTRA-N"
    )
    filtered_df.loc[filtered_df["dataset"] == "MANTRA_orientation", "dataset"] = (
        "MANTRA-O"
    )
    return filtered_df


def generate_table(df, optimization_metrics):
    """
    Generate TWO LaTeX tables:

    1) One table for MANTRA datasets (MANTRA_name, MANTRA_orientation).
    2) One table for all OTHER datasets.

    Each table is "flipped": columns = datasets (with smaller labels + up/down arrow), rows = models.

    For each domain in a table, we:
      - Print two midrules, one before the domain label and one after, to "sandwich" it.
      - Show that domain as a small subtitle.
      - Then list models in that domain below.

    When grouping by (domain, dataset, model), pick the best row among variants using:
      - direction='max' => highest mean
      - direction='min' => lowest mean

    Coloring logic:
      - If a cell's score is the absolute best for that dataset, color the cell gray and make it bold.
      - If a cell's score is within 1 standard deviation of the best, color the cell light blue.
        Specifically:
          * direction='max': highlight if (best_val - mean) <= std
          * direction='min': highlight if (mean - best_val) <= std
      - All numeric values use \\scriptsize to be smaller.

    Requirements in LaTeX:
      - \\usepackage{booktabs} (\\toprule, \\midrule, \\bottomrule)
      - \\usepackage[table]{xcolor} (for \\cellcolor)
    """

    # Make a copy to avoid altering the original DataFrame
    df = df.copy()

    # 2) Among multiple variants for (domain, dataset, model), pick best according to direction
    def pick_best_variant(group):
        dataset = group["dataset"].iloc[0]
        direction = optimization_metrics.get(dataset, {}).get(
            "direction", "max"
        )
        if len(group["mean"]) == 1:
            return group.iloc[0]
        if "BN" in dataset:
            direction = "max"
        if direction == "min":
            return group.loc[group["mean"].idxmin()]
        else:
            return group.loc[group["mean"].idxmax()]

    grouped = df.groupby(["domain", "dataset", "model"], group_keys=False)
    df_best = grouped.apply(pick_best_variant).reset_index(drop=True)

    # 3) Split into MANTRA vs. Other
    mantra_dsets = [
        "MANTRA-N",
        "MANTRA-O",
        "MANTRA-BN-0",
        "MANTRA-BN-1",
        "MANTRA-BN-2",
    ]
    df_mantra = df_best[df_best["dataset"].isin(mantra_dsets)]
    df_other = df_best[~df_best["dataset"].isin(mantra_dsets)]

    def build_table(subset_df, caption_text):
        """
        Build a single "flipped" table from subset_df.
          - columns = sorted datasets, each with an up/down arrow in \\scriptsize
          - rows = models, grouped by domain with a domain subtitle sandwiched by two midrules
          - color the best cell gray (bold text),
            color cells within 1 std of the best in light blue,
            all numbers in \\scriptsize
        """
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

        # Collect all datasets in this subset, sorted
        unique_datasets = subset_df["dataset"].unique()
        all_datasets = [d for d in DATASET_ORDER if d in unique_datasets]

        # We'll group by domain to produce domain blocks
        domain_groups = {}
        for _, row in subset_df.iterrows():
            dom = row["pe_type"]
            domain_groups.setdefault(dom, []).append(row)
        # Convert each domain list to a DataFrame
        for dom in domain_groups:
            domain_groups[dom] = pd.DataFrame(domain_groups[dom])

        # Determine best values for each dataset (max or min), to know who is best
        # also used for "within 1 std" checks
        best_vals = {}
        directions = {}
        for dset in all_datasets:
            direction = optimization_metrics.get(dset, {}).get(
                "direction", "max"
            )
            directions[dset] = direction
            dset_rows = subset_df[subset_df["dataset"] == dset]
            if dset_rows.empty:
                best_vals[dset] = None
                continue
            if direction == "min":
                best_vals[dset] = dset_rows["mean"].min()
            else:
                best_vals[dset] = dset_rows["mean"].max()

        # Function to style a single cell (mean, std) for dataset dset
        def style_cell(mn, st, dset):
            direction = directions[dset]
            best_val = best_vals[dset]

            # always use scriptsize
            content = f"\\scriptsize {mn:.2f} $\\pm$ {st:.2f}"

            if best_val is None:
                return content

            # is this the best?
            is_best = abs(mn - best_val) < 1e-12

            # within std logic
            if direction == "max":
                within_std = (best_val - mn) <= st
            else:  # direction == 'min'
                within_std = (mn - best_val) <= st

            if is_best:
                # best => gray + bold
                return f"\\cellcolor{{bestgray}}\\textbf{{{content}}}"
            elif within_std:
                # within std => blue
                return f"\\cellcolor{{stdblue}}{content}"
            else:
                return content

        # Build LaTeX lines
        latex_lines = []
        latex_lines.append(r"\begin{table}[ht]")
        latex_lines.append(r"\caption{" + caption_text + "}")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\small")

        # columns: 1 for model + len(all_datasets)
        col_spec = "@{}ll" + "c" * len(all_datasets) + "@{}"
        latex_lines.append(r"\begin{adjustbox}{width=1.\textwidth}")
        latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
        latex_lines.append(r"\toprule")

        # Header row: "Model" + each dataset with arrow
        header_cells = [r"& \textbf{Model}"]
        for dset in all_datasets:
            arrow = (
                r"($\uparrow$)"
                if directions[dset] == "max"
                else r"($\downarrow$)"
            )
            header_cells.append(r"\scriptsize " + dset + " " + arrow)
        latex_lines.append(" & ".join(header_cells) + r" \\")

        # sort domains to have consistent ordering
        all_domains = sorted(domain_groups.keys())
        # if len(all_domains) == 3:
        #     all_domains = ["graph", "simplicial", "cell"]

        # For each domain, we do the "sandwiching" with midrules
        for dom in all_domains:
            dom_df = domain_groups[dom]
            dom_models = [m for m in MODEL_ORDER[dom] if m in dom_df["model"].unique()]
            # domain subtitle row
            latex_lines.append(r"\midrule")
            latex_lines.append(
                rf"\multirow{{{len(dom_models)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{dom.capitalize()}}}}}}}"
            )

            # each row => [Model, val(dset1), val(dset2), ...]
            for model in dom_models:
                row_elems = [model]
                for dset in all_datasets:
                    sel = dom_df[
                        (dom_df["model"] == model)
                        & (dom_df["dataset"] == dset)
                    ]
                    if sel.empty:
                        print(model, dset)
                        cell_val = "-"
                    else:
                        r = sel.iloc[0]
                        mn, st = r["mean"], r["std"]
                        cell_val = style_cell(mn, st, dset)
                    row_elems.append(cell_val)
                latex_lines.append("& " + " & ".join(row_elems) + r" \\")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{adjustbox}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)

    # Build the two separate tables
    # latex_mantra = build_table(
    #     df_mantra,
    #     "Topological datasets. Results are shown as mean and standard deviation. The best result is bold and shaded in grey, while those within one standard deviation are in blue-shaded boxes.",
    # )
    # latex_others = build_table(df_other, "Benchmarking datasets. Results are shown as mean and standard deviation. The best result is bold and shaded in grey, while those within one standard deviation are in blue-shaded boxes.")



    # Return them combined with some spacing
    # return latex_mantra + "\n\n" + latex_others
    latex_all = build_table(
        df_best,
        "Benchmarking datasets. Results are shown as mean and standard deviation. The best result is bold and shaded in grey, while those within one standard deviation are in blue-shaded boxes."
    )
    return latex_all


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
    latex_table = generate_table(df, optimization_metrics)
    print(latex_table)
