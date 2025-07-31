import pandas as pd
from constants import (
    DATASET_ORDER,
    MODEL_ORDER,
    optimization_metrics,
)
from fetch_and_parse import main
from make_time_tables import fix_domain


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

def parse_hopse_results(df, selected_datasets):
    df_dict = {
        "model": [],
        "dataset": [],
        "size_k": [],
        "domain": [],
    }

    # df = split_evaluation_metrics(df)
    df["model.model_domain"] = df.apply(fix_domain, axis=1)

    models = df["model.model_name"].unique()
    datasets = df["dataset.loader.parameters.data_name"].unique()
    domains = df["model.model_domain"].unique()


    for dataset in datasets:
        for model in models:
            for domain in domains:
                subset = df[
                    (df["dataset.loader.parameters.data_name"] == dataset)
                    & (df["model.model_name"] == model)
                    & (df["model.model_domain"] == domain)
                ]
                if subset.empty:
                    print(dataset, model, domain)
                    continue

                df_dict["domain"].append(domain)
                df_dict["model"].append(model)
                df_dict["dataset"].append(dataset)
                df_dict["size_k"].append(subset["model/params/trainable"].iloc[0] / 1000)
    df_res = pd.DataFrame(df_dict)
    return df_res


def parse_topotune_results():
    data = []
    data.extend([
        {
            "model": "GCCN with GAT",
            "domain": "cell",
            "dataset": "MUTAG",
            "size_k": 15.11,
        },
        {
            "model": "GCCN with GAT",
            "domain": "cell",
            "dataset": "PROTEINS",
            "size_k": 46.27,
        },
        {
            "model": "GCCN with GAT",
            "domain": "cell",
            "dataset": "NCI1",
            "size_k": 68.99,
        },
        {
            "model": "GCCN with GAT",
            "domain": "cell",
            "dataset": "NCI109",
            "size_k": 49.63,
        },
        {
            "model": "GCCN with GAT",
            "domain": "cell",
            "dataset": "ZINC",
            "size_k": 39.78,
        },

        {
            "model": "GCCN with GCN",
            "domain": "cell",
            "dataset": "MUTAG",
            "size_k": 45.44,
        },
        {
            "model": "GCCN with GCN",
            "domain": "cell",
            "dataset": "PROTEINS",
            "size_k": 45.25,
        },
        {
            "model": "GCCN with GCN",
            "domain": "cell",
            "dataset": "NCI1",
            "size_k": 65.92,
        },
        {
            "model": "GCCN with GCN",
            "domain": "cell",
            "dataset": "NCI109",
            "size_k": 30.69,
        },
        {
            "model": "GCCN with GCN",
            "domain": "cell",
            "dataset": "ZINC",
            "size_k": 29.54,
        },

        {
            "model": "GCCN with GIN",
            "domain": "cell",
            "dataset": "MUTAG",
            "size_k": 63.62,
        },
        {
            "model": "GCCN with GIN",
            "domain": "cell",
            "dataset": "PROTEINS",
            "size_k": 23.49,
        },
        {
            "model": "GCCN with GIN",
            "domain": "cell",
            "dataset": "NCI1",
            "size_k": 49.03,
        },
        {
            "model": "GCCN with GIN",
            "domain": "cell",
            "dataset": "NCI109",
            "size_k": 66.79,
        },
        {
            "model": "GCCN with GIN",
            "domain": "cell",
            "dataset": "ZINC",
            "size_k": 64.35,
        },

        {
            "model": "GCCN with GraphSAGE",
            "domain": "cell",
            "dataset": "MUTAG",
            "size_k": 44.42,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "cell",
            "dataset": "PROTEINS",
            "size_k": 76.99,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "cell",
            "dataset": "NCI1",
            "size_k": 47.49,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "cell",
            "dataset": "NCI109",
            "size_k": 115.17,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "cell",
            "dataset": "ZINC",
            "size_k": 79.71,
        },

        {
            "model": "GCCN with Transformer",
            "domain": "cell",
            "dataset": "MUTAG",
            "size_k": 112.26,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "cell",
            "dataset": "PROTEINS",
            "size_k": 78.79,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "cell",
            "dataset": "NCI1",
            "size_k": 82.05,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "cell",
            "dataset": "NCI109",
            "size_k": 115.43,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "cell",
            "dataset": "ZINC",
            "size_k": 317.02,
        },

        {
            "model": "GCCN with Hasse",
            "domain": "cell",
            "dataset": "MUTAG",
            "size_k": 14.98,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "cell",
            "dataset": "PROTEINS",
            "size_k": 18.88,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "cell",
            "dataset": "NCI1",
            "size_k": 18.05,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "cell",
            "dataset": "NCI109",
            "size_k": 15.91,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "cell",
            "dataset": "ZINC",
            "size_k": 20.83,
        },
    ])


    data.extend([
        {
            "model": "GCCN with GAT",
            "domain": "simplicial",
            "dataset": "MUTAG",
            "size_k": 15.11,
        },
        {
            "model": "GCCN with GAT",
            "domain": "simplicial",
            "dataset": "PROTEINS",
            "size_k": 46.27,
        },
        {
            "model": "GCCN with GAT",
            "domain": "simplicial",
            "dataset": "NCI1",
            "size_k": 68.99,
        },
        {
            "model": "GCCN with GAT",
            "domain": "simplicial",
            "dataset": "NCI109",
            "size_k": 49.63,
        },
        {
            "model": "GCCN with GAT",
            "domain": "simplicial",
            "dataset": "ZINC",
            "size_k": 67.42,
        },

        {
            "model": "GCCN with GCN",
            "domain": "simplicial",
            "dataset": "MUTAG",
            "size_k": 45.44,
        },
        {
            "model": "GCCN with GCN",
            "domain": "simplicial",
            "dataset": "PROTEINS",
            "size_k": 45.25,
        },
        {
            "model": "GCCN with GCN",
            "domain": "simplicial",
            "dataset": "NCI1",
            "size_k": 65.92,
        },
        {
            "model": "GCCN with GCN",
            "domain": "simplicial",
            "dataset": "NCI109",
            "size_k": 30.69,
        },
        {
            "model": "GCCN with GCN",
            "domain": "simplicial",
            "dataset": "ZINC",
            "size_k": 64.35,
        },

        {
            "model": "GCCN with GIN",
            "domain": "simplicial",
            "dataset": "MUTAG",
            "size_k": 63.62,
        },
        {
            "model": "GCCN with GIN",
            "domain": "simplicial",
            "dataset": "PROTEINS",
            "size_k": 23.49,
        },
        {
            "model": "GCCN with GIN",
            "domain": "simplicial",
            "dataset": "NCI1",
            "size_k": 49.03,
        },
        {
            "model": "GCCN with GIN",
            "domain": "simplicial",
            "dataset": "NCI109",
            "size_k": 66.79,
        },
        {
            "model": "GCCN with GIN",
            "domain": "simplicial",
            "dataset": "ZINC",
            "size_k": 118.11,
        },

        {
            "model": "GCCN with GraphSAGE",
            "domain": "simplicial",
            "dataset": "MUTAG",
            "size_k": 44.42,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "simplicial",
            "dataset": "PROTEINS",
            "size_k": 76.99,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "simplicial",
            "dataset": "NCI1",
            "size_k": 47.49,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "simplicial",
            "dataset": "NCI109",
            "size_k": 115.17,
        },
        {
            "model": "GCCN with GraphSAGE",
            "domain": "simplicial",
            "dataset": "ZINC",
            "size_k": 147.30,
        },

        {
            "model": "GCCN with Transformer",
            "domain": "simplicial",
            "dataset": "MUTAG",
            "size_k": 113.15,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "simplicial",
            "dataset": "PROTEINS",
            "size_k": 213.70,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "simplicial",
            "dataset": "NCI1",
            "size_k": 82.05,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "simplicial",
            "dataset": "NCI109",
            "size_k": 166.24,
        },
        {
            "model": "GCCN with Transformer",
            "domain": "simplicial",
            "dataset": "ZINC",
            "size_k": 148.83,
        },

        {
            "model": "GCCN with Hasse",
            "domain": "simplicial",
            "dataset": "MUTAG",
            "size_k": 19.07,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "simplicial",
            "dataset": "PROTEINS",
            "size_k": 14.66,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "simplicial",
            "dataset": "NCI1",
            "size_k": 31.11,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "simplicial",
            "dataset": "NCI109",
            "size_k": 15.91,
        },
        {
            "model": "GCCN with Hasse",
            "domain": "simplicial",
            "dataset": "ZINC",
            "size_k": 29.54,
        },
    ])

    # Simplicial models
    topotune_df = pd.DataFrame(data)
    topotune_df["variant"] = topotune_df["model"].map(
        lambda x: x.split(" ")[-1]
    )
    topotune_df["model"] = topotune_df["model"].map(lambda x: x.split(" ")[0])

    return topotune_df


def parse_tb_results():
    # Define the raw table data
    raw_table_data = {
        "MUTAG": {
            "GCN":  67.97,  "GAT":  22.02,  "GIN":  38.40,  "AST":  80.77,
            "EDGNN":  5.73, "UniGNN2": 84.10,"CWN": 334.72,  "CCCN":284.29,
            "CCXN": 73.86,  "SCN": 20.03,   "SCCN":398.85, "SCCNN": 27.11,
        },
        "PROTEINS": {
            "GCN":  13.19,  "GAT":  10.11,  "GIN":  13.19,  "AST":  14.34,
            "EDGNN":  5.60, "UniGNN2": 21.31,"CWN": 101.12,  "CCCN": 34.56,
            "CCXN": 86.53,  "SCN": 10.24,   "SCCN":397.31, "SCCNN": 26.72,
        },
        "NCI1": {
            "GCN":   6.72,  "GAT":  11.20,  "GIN": 154.37,  "AST":  57.47,
            "EDGNN":88.19,  "UniGNN2":104.32,"CWN": 124.10,  "CCCN": 63.87,
            "CCXN": 15.87,  "SCN": 94.98,   "SCCN":131.84, "SCCNN":188.99,
        },
        "NCI109": {
            "GCN":  23.75,  "GAT":  11.23,  "GIN": 154.50,  "AST": 221.57,
            "EDGNN":88.32,  "UniGNN2":  4.61,"CWN": 412.29,  "CCCN": 17.67,
            "CCXN": 71.36,  "SCN": 26.08,   "SCCN":135.75, "SCCNN": 49.54,
        },
        "ZINC": {
            "GCN":  22.59,  "GAT":  22.85,  "GIN":  10.40,  "AST":106.82,
            "EDGNN":22.53,  "UniGNN2":102.14,"CWN":  88.06,  "CCCN":287.74,
            "CCXN": 16.48,  "SCN": 24.42,   "SCCN":617.86, "SCCNN":1453.82,
        },
    }

    # Process the data to select the best of DR and SDP for each method
    additional_data = []

    for dataset, entries in raw_table_data.items():
        optim_dir = optimization_metrics[dataset]["direction"]

        # Group data by method prefix (before the underscore)
        method_results = {}
        standard_methods = []
        for method, size_k in entries.items():
            if method in ["CWN", "CCCN", "SCCNN", "SCN", "GCN", "GIN", "GAT"]:
                # These are the standard methods - just add them directly
                standard_methods.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "size_k": size_k
                    }
                )

        # Add all standard methods
        additional_data.extend(standard_methods)

    # Map method names to their proper format in the data
    method_mappings = {
        "CWN": "CWN",
        "CCCN": "CCCN",  # Cell CCNN
        "SCCNN": "SCCNN",
        "SCN": "SCN",
        "GCN": "GCN",
        "GIN": "GIN",
        "GAT": "GAT",
    }

    # Domain mappings
    domain_mappings = {
        "CWN": "cell",
        "CCCN": "cell",
        "SCCNN": "simplicial",
        "SCN": "simplicial",
        "GCN": "graph",
        "GIN": "graph",
        "GAT": "graph",
    }

    # Convert the additional data to the proper format for the dataframe
    formatted_data = []
    for item in additional_data:
        formatted_data.append(
            {
                "model": method_mappings[item["method"]],
                "domain": domain_mappings[item["method"]],
                "dataset": item["dataset"],
                "size_k": item["size_k"],
            }
        )

    # This data can now be added to your existing dataframe or used to create a new one
    tbx_df = pd.DataFrame(formatted_data)

    return tbx_df


def parse_all_dfs(selected_datasets=[]):
    df = main()
    df_hopse = parse_hopse_results(df, selected_datasets)
    mask = (df_hopse["model"] == "sann") & (df_hopse["dataset"] == "ZINC")

    df_hopse = df_hopse[~df_hopse.isna()]
    df_topotune = parse_topotune_results()
    df_tb = parse_tb_results()
    cat_df = pd.concat([df_hopse, df_topotune, df_tb], ignore_index=True)

    # FIX Hopse naming
    cat_df.loc[cat_df["model"] == "hopse_m", "model"] = "HOPSE-M"
    cat_df.loc[cat_df["model"] == "hopse_g", "model"] = "HOPSE-G"
    cat_df.loc[cat_df["model"] == "sccnn", "model"] = "SCCNN"
    cat_df.loc[cat_df["model"] == "scn", "model"] = "SCN"
    cat_df.loc[cat_df["model"] == "gcn", "model"] = "GCN"
    cat_df.loc[cat_df["model"] == "gin", "model"] = "GIN"
    cat_df.loc[cat_df["model"] == "gat", "model"] = "GAT"
    cat_df.loc[cat_df["model"] == "sann", "model"] = "SaNN"
    cat_df.loc[cat_df["model"] == "topotune", "model"] = "GCCN"

    filtered_df = cat_df[cat_df["dataset"].isin(selected_datasets)]

    # FIX MAntra naming
    # filtered_df.loc[
    #     filtered_df["dataset"] == "MANTRA_betti_numbers_0", "dataset"
    # ] = "MANTRA-BN-0"
    # filtered_df.loc[
    #     filtered_df["dataset"] == "MANTRA_betti_numbers_1", "dataset"
    # ] = "MANTRA-BN-1"
    # filtered_df.loc[
    #     filtered_df["dataset"] == "MANTRA_betti_numbers_2", "dataset"
    # ] = "MANTRA-BN-2"
    filtered_df.loc[
        filtered_df["dataset"] == "MANTRA_betti_numbers", "dataset"
    ] = "MANTRA-BN"
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
    df_best = df.copy()

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
            dom = row["domain"]
            domain_groups.setdefault(dom, []).append(row)
        # Convert each domain list to a DataFrame
        for dom in domain_groups:
            domain_groups[dom] = pd.DataFrame(domain_groups[dom])

        # Determine best values for each dataset (max or min), to know who is best
        # also used for "within 1 std" checks
        best_vals_by_domain = {
            dom: {
                dset: dom_df[dom_df["dataset"] == dset]["size_k"].max()
                for dset in all_datasets
                if not dom_df[dom_df["dataset"] == dset].empty
            }
            for dom, dom_df in domain_groups.items()
        }

        # Function to style a single cell (mean, std) for dataset dset
        def style_cell(size, dset, dom):
            best_val = best_vals_by_domain[dom].get(dset, None)

            # always use scriptsize
            content = f"\\scriptsize {size:.2f}K"

            if best_val is None:
                return content

            # is this the best?
            is_best = abs(size - best_val) < 1e-12

            if is_best:
                # best => gray + bold
                return f"\\cellcolor{{bestgray}}\\textbf{{{content}}}"
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
            header_cells.append(r"\scriptsize " + dset)
        latex_lines.append(" & ".join(header_cells) + r" \\")

        # sort domains to have consistent ordering
        all_domains = sorted(domain_groups.keys())
        if len(all_domains) == 3:
            all_domains = ["graph", "simplicial", "cell"]

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
                        size_k = r["size_k"]
                        cell_val = style_cell(size_k, dset, dom)
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
        "Model size per datasets. Results are shown as model size in thousand (K). The best result is bold and shaded in grey."
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
        "MANTRA_betti_numbers",
    ]

    # Parse the dataframes
    df = parse_all_dfs(selected_datasets)
    #df.drop(['variant'], inplace=True, axis=1)
    # mask = (df['model'] == 'HOPSE-M') & (df['dataset'] == 'ZINC')

    # Generate the LaTeX table
    latex_table = generate_table(df, optimization_metrics)
    print(latex_table)
