import pandas as pd
from constants import keep_columns, optimization_metrics, DATASET_ORDER
from generate_scores import gen_scores
from preprocess import preprocess_df


NAME_DICT_PE = {
    "rwse,elstaticpe,hkdiagse,lappe": "All",
    "rwse": "RWSE",
    "hkdiagse": "HKDiag",
    "lappe": "LapPE",
    "elstaticpe": "EStat",
}


def parse_pse_results(datasets, collect_subsets):
    df_dict = {
        "model": [],
        "dataset": [],
        "mean": [],
        "std": [],
        "pe_type": [],
        "domain": [],
    }

    for dataset in datasets:
        aggregated = collect_subsets[dataset]
        if len(aggregated) == 0:
            continue
        domains = aggregated["model.model_domain"].unique()
        for m_name in aggregated["model.model_name"].unique():
            for pe_type in aggregated[
                "transforms.sann_encoding.pe_types"
            ].unique():
                for domain in domains:
                    agg_sub = aggregated[
                        (aggregated["model.model_name"] == m_name)
                        & (
                            aggregated["transforms.sann_encoding.pe_types"]
                            == pe_type
                        )
                        & (aggregated["model.model_domain"] == domain)
                    ].copy()
                    if len(agg_sub) == 0:
                        print("Not found:", dataset, m_name, pe_type, domain)
                        continue
                    optim_metric = optimization_metrics[dataset][
                        "optim_metric"
                    ]
                    eval_metric = optimization_metrics[dataset]["eval_metric"]
                    optim_dir = optimization_metrics[dataset]["direction"]
                    agg_sub.sort_values(
                        by=(optim_metric, "mean"),
                        ascending=(optim_dir == "min"),
                        inplace=True,
                    )
                    # pe_type = NAME_DICT_PE[pe_type.lower()]
                    df_dict["pe_type"].append(pe_type)
                    df_dict["model"].append(m_name)
                    df_dict["dataset"].append(dataset)
                    df_dict["domain"].append(domain)
                    df_dict["mean"].append(
                        agg_sub.iloc[0][(eval_metric, "mean")]
                    )
                    df_dict["std"].append(
                        agg_sub.iloc[0][(eval_metric, "std")]
                    )
    df_res = pd.DataFrame(df_dict)
    return df_res


def parse_all_dfs(selected_datasets=[]):
    csv = "merged_gnn_rebutal"
    df = pd.read_csv(f"{csv}/merged_normalized.csv")
    df = preprocess_df(df, gnn=True, split_mantra=False)
    print(df)
    # Keep only relevant columns
    df = df[keep_columns["gnn"]]
    # Generate best scores per hyperparameter sweep
    scores = gen_scores(df, gnn=True)

    df_pse = parse_pse_results(selected_datasets, scores)

    # FIX MAntra naming
    # df_pse.loc[df_pse.dataset == "MANTRA_betti_numbers_0", "dataset"] = "MANTRA-BN-0"
    # df_pse.loc[df_pse.dataset == "MANTRA_betti_numbers_1", "dataset"] = "MANTRA-BN-1"
    # df_pse.loc[df_pse.dataset == "MANTRA_betti_numbers_2", "dataset"] = "MANTRA-BN-2"
    # df_pse.loc[df_pse.dataset == "MANTRA_name", "dataset"] = "MANTRA-N"
    # df_pse.loc[df_pse.dataset == "MANTRA_orientation", "dataset"] = "MANTRA-O"

    # Only grab the datasets we are interested in

    return df_pse


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
        if "BN" in dataset:
            direction = "max"
        if direction == "min":
            return group.loc[group["mean"].idxmin()]
        else:
            return group.loc[group["mean"].idxmax()]

    grouped = df.groupby(["pe_type", "dataset", "model"], group_keys=False)
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
            dom = row["model"]
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
        header_cells = [r"& \textbf{PE}"]
        for dset in all_datasets:
            arrow = (
                r"($\uparrow$)"
                if directions[dset] == "max"
                else r"($\downarrow$)"
            )
            header_cells.append(r"\scriptsize " + dset + " " + arrow)
        latex_lines.append(" & ".join(header_cells) + r" \\")

        # sort domains to have consistent ordering
        order_list = ["GCN", "GAT", "GIN"]
        all_domains = [d for d in order_list if d in domain_groups.keys()]

        # For each domain, we do the "sandwiching" with midrules
        for dom in all_domains:
            dom_df = domain_groups[dom]
            # dom_models = [m for m in MODEL_ORDER[dom] if m in dom_df["pe"].unique()]
            dom_pes = list(dom_df["pe_type"].unique())

            # domain subtitle row
            latex_lines.append(r"\midrule")
            latex_lines.append(
                rf"\multirow{{{len(dom_pes)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{dom.upper()}}}}}}}"
            )

            # each row => [Model, val(dset1), val(dset2), ...]
            for pe in dom_pes:
                row_elems = [NAME_DICT_PE[pe.lower()]]
                for dset in all_datasets:
                    sel = dom_df[
                        (dom_df["pe_type"] == pe) & (dom_df["dataset"] == dset)
                    ]
                    if sel.empty:
                        print(pe, dset)
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

    # def build_table(subset_df, caption_text):
    #     """
    #     Build a single "flipped" table from subset_df.
    #       - columns = sorted datasets, each with an up/down arrow in \\scriptsize
    #       - rows = models, grouped by domain with a domain subtitle sandwiched by two midrules
    #       - color the best cell gray (bold text),
    #         color cells within 1 std of the best in light blue,
    #         all numbers in \\scriptsize
    #     """
    #     if subset_df.empty:
    #         return (
    #             r"\begin{table}[ht]"
    #             "\n"
    #             r"\centering"
    #             "\n"
    #             r"\small"
    #             "\n"
    #             r"\textit{No data for this subset.}"
    #             "\n"
    #             r"\end{table}"
    #             "\n"
    #         )

    #     # Collect all datasets in this subset, sorted

    #     unique_datasets = subset_df["dataset"].unique()
    #     all_datasets = [ds for ds in DATASET_ORDER if ds in unique_datasets]

    #     # We'll group by domain to produce domain blocks
    #     pe_groups = {}
    #     for _, row in subset_df.iterrows():
    #         pe = row["model"]
    #         pe_groups.setdefault(pe, []).append(row)
    #     # Convert each domain list to a DataFrame
    #     for pe in pe_groups:
    #         pe_groups[pe] = pd.DataFrame(pe_groups[pe])

    #     # Determine best values for each dataset (max or min), to know who is best
    #     # also used for "within 1 std" checks
    #     best_vals = {}
    #     directions = {}
    #     for dset in all_datasets:
    #         direction = optimization_metrics.get(dset, {}).get(
    #             "direction", "max"
    #         )
    #         directions[dset] = direction
    #         dset_rows = subset_df[subset_df["dataset"] == dset]
    #         if dset_rows.empty:
    #             best_vals[dset] = None
    #             continue
    #         if direction == "min":
    #             best_vals[dset] = dset_rows["mean"].min()
    #         else:
    #             best_vals[dset] = dset_rows["mean"].max()

    #     # Function to style a single cell (mean, std) for dataset dset
    #     def style_cell(mn, st, dset):
    #         direction = directions[dset]
    #         best_val = best_vals[dset]

    #         # always use scriptsize
    #         content = f"\\scriptsize {mn:.2f} $\\pm$ {st:.2f}"

    #         if best_val is None:
    #             return content

    #         # is this the best?
    #         is_best = abs(mn - best_val) < 1e-12

    #         # within std logic
    #         if direction == "max":
    #             within_std = (best_val - mn) <= st
    #         else:  # direction == 'min'
    #             within_std = (mn - best_val) <= st

    #         if is_best:
    #             # best => gray + bold
    #             return f"\\cellcolor{{bestgray}}\\textbf{{{content}}}"
    #         elif within_std:
    #             # within std => blue
    #             return f"\\cellcolor{{stdblue}}{content}"
    #         else:
    #             return content

    #     # Build LaTeX lines
    #     latex_lines = []
    #     latex_lines.append(r"\begin{table}[ht]")
    #     latex_lines.append(r"\centering")
    #     latex_lines.append(r"\small")

    #     # columns: 1 for model + len(all_datasets)
    #     col_spec = "@{}ll" + "c" * len(all_datasets) + "@{}"
    #     latex_lines.append(r"\begin{adjustbox}{width=1.\textwidth}")
    #     latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    #     latex_lines.append(r"\toprule")

    #     # Header row: "Model" + each dataset with arrow
    #     header_cells = [r"& \textbf{PE}"]
    #     for dset in all_datasets:
    #         arrow = (
    #             r"($\uparrow$)"
    #             if directions[dset] == "max"
    #             else r"($\downarrow$)"
    #         )
    #         header_cells.append(r"\scriptsize " + dset + " " + arrow)
    #     latex_lines.append(" & ".join(header_cells) + r" \\")

    #     # sort domains to have consistent ordering
    #     all_pe = sorted(pe_groups.keys())

    #     # if len(all_domains) == 3:
    #     #     all_domains = ["graph", "simplicial", "cell"]

    #     # For each domain, we do the "sandwiching" with midrules
    #     for pe in all_pe:
    #         pe_df = pe_groups[pe]
    #         pe_models = sorted(pe_df["pe_type"].unique())
    #         # domain subtitle row
    #         latex_lines.append(r"\midrule")
    #         latex_lines.append(
    #             rf"\multirow{{{len(pe_models)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{pe}}}}}}}"
    #         )

    #         # each row => [Model, val(dset1), val(dset2), ...]
    #         for model in pe_models:
    #             row_elems = [model]
    #             for dset in all_datasets:
    #                 sel = pe_df[
    #                     (pe_df["pe_type"] == model)
    #                     & (pe_df["dataset"] == dset)
    #                 ]
    #                 if sel.empty:
    #                     print(model, dset)
    #                     cell_val = "-"
    #                 else:
    #                     r = sel.iloc[0]
    #                     mn, st = r["mean"], r["std"]
    #                     cell_val = style_cell(mn, st, dset)
    #                 row_elems.append(cell_val)
    #             latex_lines.append("& " + " & ".join(row_elems) + r" \\")

    #     latex_lines.append(r"\bottomrule")
    #     latex_lines.append(r"\end{tabular}")
    #     latex_lines.append(r"\end{adjustbox}")
    #     latex_lines.append(r"\caption{" + caption_text + "}")
    #     latex_lines.append(r"\end{table}")

    #     return "\n".join(latex_lines)

    # Build the two separate tables
    latex_mantra = build_table(
        df_mantra,
        "Results for MANTRA datasets (MANTRA-N, MANTRA-O, MANTRA-BN).",
    )
    latex_others = build_table(df_other, "Ablation study over GNNs + PEs.")

    # Return them combined with some spacing
    return latex_mantra + "\n\n" + latex_others


if __name__ == "__main__":
    # Define the datasets to include in the table
    selected_datasets = [
        "MUTAG",
        "PROTEINS",
        "NCI1",
        "NCI109",
        # "IMDB-BINARY",
        # "IMDB-MULTI",
        "ZINC",
        # "MANTRA_betti_numbers_0",
        # "MANTRA_betti_numbers_1",
        # "MANTRA_betti_numbers_2",
        # "MANTRA_name",
        # "MANTRA_orientation",
    ]

    # Parse the dataframes
    df = parse_all_dfs(selected_datasets)

    # Generate the LaTeX table
    latex_table = generate_table(df, optimization_metrics)
    print(latex_table)
