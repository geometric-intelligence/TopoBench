
import pandas as pd
from constants import DATASET_ORDER, MODEL_ORDER
from fetch_and_parse import main

PRETTY_MAP = {
    "preprocessor_time": "Preprocessing Time",
    "_runtime": "Total Runtime",
    "AvgTime/val_batch_mean": "Average Time per Batch",
    "AvgTime/val_epoch_mean": "Average Time per Epoch",
}

def fix_domain(row):
    if "transforms.graph2simplicial_lifting.transform_name" in row:
        if not pd.isna(row["transforms.graph2simplicial_lifting.transform_name"]):
            return "simplicial"
    if "transforms.redefine_simplicial_neighborhoods.transform_name" in row:
        if not pd.isna(row["transforms.redefine_simplicial_neighborhoods.transform_name"]):
            return "simplicial"
    if "transforms.graph2cell_lifting.transform_name" in row:
        if not pd.isna(row["transforms.graph2cell_lifting.transform_name"]):
            return "cell"
    return row["model.model_domain"]
    

def generate_table(df, time_column):
    df_best = df.copy()

    mantra_dsets = ["MANTRA-N", "MANTRA-O", "MANTRA-BN"]
    df_mantra = df_best[df_best["dataset"].isin(mantra_dsets)]
    df_other = df_best[~df_best["dataset"].isin(mantra_dsets)]

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

        val = "mean" if time_column != "preprocessor_time" else "max"
        best_vals_by_domain = {
            dom: {
                dset: dom_df[dom_df["dataset"] == dset][val].min()
                for dset in all_datasets
                if not dom_df[dom_df["dataset"] == dset].empty
            }
            for dom, dom_df in domain_groups.items()
        }

        def style_cell(mn, st, dset, dom):
            best_val = best_vals_by_domain[dom].get(dset, None)
            direction = directions[dset]

            if time_column != "preprocessor_time":
                content = f"\\scriptsize {mn:.2f} $\\pm$ {st:.2f}"
            else:
                content = f"\\scriptsize {mn:.0f} "

            if best_val is None:
                return content

            if time_column != "preprocessor_time":
                is_best = abs(mn - best_val) < 1e-12
                within_std = (mn - best_val) <= st if direction == "min" else (best_val - mn) <= st
            else:
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
                        if time_column != "preprocessor_time":
                            mn, st = r["mean"], r["std"]
                        else:
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

def parse_hops(time_column):
    df = main().reset_index(drop=True)
    df["model.model_domain"] = df.apply(fix_domain, axis=1)
    df.rename({
        "dataset.loader.parameters.data_name": "dataset",
        "model.model_name": "model",
        "model.model_domain": "domain",
    }, inplace=True, axis=1)
    df.loc[
        df["dataset"] == "MANTRA_betti_numbers", "dataset"
    ] = "MANTRA-BN"
    df.loc[df["dataset"] == "MANTRA_name", "dataset"] = (
        "MANTRA-N"
    )
    df.loc[df["dataset"] == "MANTRA_orientation", "dataset"] = (
        "MANTRA-O"
    )
    df.loc[df["model"] == "hopse_m", "model"] = "HOPSE-M"
    df.loc[df["model"] == "hopse_g", "model"] = "HOPSE-G"
    df.loc[df["model"] == "sccnn", "model"] = "SCCNN"
    df.loc[df["model"] == "scn", "model"] = "SCN"
    df.loc[df["model"] == "gcn", "model"] = "GCN"
    df.loc[df["model"] == "gin", "model"] = "GIN"
    df.loc[df["model"] == "gat", "model"] = "GAT"
    df.loc[df["model"] == "sann", "model"] = "SaNN"
    df.loc[df["model"] == "topotune", "model"] = "GCCN"

    grouped = df.groupby(["domain", "dataset", "model"], group_keys=True)
    if time_column == "preprocessor_time":
        df_best = grouped.agg({
            time_column: ["max"]
        }).reset_index(drop=False)
        df_best["max"] = df_best[(time_column, "max")]
        df_best.drop([(time_column, "max")], axis=1, inplace=True)
    else:
        df_best = grouped.agg({
            time_column: ["mean", "std"]
        }).reset_index(drop=False)
        df_best["mean"] = df_best[(time_column, "mean")]
        df_best["std"] = df_best[(time_column, "std")]
        df_best.drop([(time_column, "mean"), (time_column, "std")], axis=1, inplace=True)
    df_best.columns = [col if isinstance(col, str) else col[0] for col in df_best.columns]

    return df_best

def parse_tb(t_c):
    if t_c == "AvgTime/val_epoch_mean":
        data =  [
            {"model": "CCCN", "domain": "cell", "dataset": "MUTAG", "mean": 0.11, "std": 0.0},
            {"model": "CCCN", "domain": "cell", "dataset": "NCI1", "mean": 1.33, "std": 0.02},
            {"model": "CCCN", "domain": "cell", "dataset": "NCI109", "mean": 1.33, "std": 0.02},
            {"model": "CCCN", "domain": "cell", "dataset": "PROTEINS", "mean": 0.32, "std": 0.01},
            {"model": "CCCN", "domain": "cell", "dataset": "ZINC", "mean": 6.14, "std": 0.04},
            {"model": "CCXN", "domain": "cell", "dataset": "MUTAG", "mean": 0.09, "std": 0.0},
            {"model": "CCXN", "domain": "cell", "dataset": "NCI1", "mean": 1.25, "std": 0.02},
            {"model": "CCXN", "domain": "cell", "dataset": "NCI109", "mean": 1.39, "std": 0.04},
            {"model": "CCXN", "domain": "cell", "dataset": "PROTEINS", "mean": 0.32, "std": 0.01},
            {"model": "CCXN", "domain": "cell", "dataset": "ZINC", "mean": 5.92, "std": 0.09},
            {"model": "CWN", "domain": "cell", "dataset": "MUTAG", "mean": 0.1, "std": 0.0},
            {"model": "CWN", "domain": "cell", "dataset": "NCI1", "mean": 1.38, "std": 0.03},
            {"model": "CWN", "domain": "cell", "dataset": "NCI109", "mean": 1.37, "std": 0.01},
            {"model": "CWN", "domain": "cell", "dataset": "PROTEINS", "mean": 0.37, "std": 0.02},
            {"model": "CWN", "domain": "cell", "dataset": "ZINC", "mean": 5.86, "std": 0.05},
            {"model": "GAT", "domain": "graph", "dataset": "MUTAG", "mean": 0.04, "std": 0.0},
            {"model": "GAT", "domain": "graph", "dataset": "NCI1", "mean": 0.34, "std": 0.02},
            {"model": "GAT", "domain": "graph", "dataset": "NCI109", "mean": 0.33, "std": 0.02},
            {"model": "GAT", "domain": "graph", "dataset": "PROTEINS", "mean": 0.07, "std": 0.0},
            {"model": "GAT", "domain": "graph", "dataset": "ZINC", "mean": 1.24, "std": 0.01},
            {"model": "GCN", "domain": "graph", "dataset": "MUTAG", "mean": 0.03, "std": 0.0},
            {"model": "GCN", "domain": "graph", "dataset": "NCI1", "mean": 0.26, "std": 0.01},
            {"model": "GCN", "domain": "graph", "dataset": "NCI109", "mean": 0.27, "std": 0.02},
            {"model": "GCN", "domain": "graph", "dataset": "PROTEINS", "mean": 0.05, "std": 0.0},
            {"model": "GCN", "domain": "graph", "dataset": "ZINC", "mean": 1.24, "std": 0.01},
            {"model": "GIN", "domain": "graph", "dataset": "MUTAG", "mean": 0.03, "std": 0.0},
            {"model": "GIN", "domain": "graph", "dataset": "NCI1", "mean": 0.27, "std": 0.02},
            {"model": "GIN", "domain": "graph", "dataset": "NCI109", "mean": 0.27, "std": 0.02},
            {"model": "GIN", "domain": "graph", "dataset": "PROTEINS", "mean": 0.06, "std": 0.0},
            {"model": "GIN", "domain": "graph", "dataset": "ZINC", "mean": 1.19, "std": 0.0},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "MUTAG", "mean": 0.09, "std": 0.0},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "NCI1", "mean": 1.65, "std": 0.02},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "NCI109", "mean": 1.66, "std": 0.03},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "PROTEINS", "mean": 0.43, "std": 0.01},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "ZINC", "mean": 10.4, "std": 0.06},
            {"model": "SCN", "domain": "simplicial", "dataset": "MUTAG", "mean": 0.09, "std": 0.0},
            {"model": "SCN", "domain": "simplicial", "dataset": "NCI1", "mean": 1.6, "std": 0.03},
            {"model": "SCN", "domain": "simplicial", "dataset": "NCI109", "mean": 1.59, "std": 0.02},
            {"model": "SCN", "domain": "simplicial", "dataset": "PROTEINS", "mean": 0.42, "std": 0.01},
            {"model": "SCN", "domain": "simplicial", "dataset": "ZINC", "mean": 6.93, "std": 0.09},
        ]
    elif t_c == "_runtime":
        data = [
            {"model": "CCCN", "domain": "cell", "dataset": "MUTAG", "mean": 12.04, "std": 2.21},
            {"model": "CCCN", "domain": "cell", "dataset": "NCI1", "mean": 372.36, "std": 109.47},
            {"model": "CCCN", "domain": "cell", "dataset": "NCI109", "mean": 272.3, "std": 20.89},
            {"model": "CCCN", "domain": "cell", "dataset": "PROTEINS", "mean": 41.63, "std": 7.23},
            {"model": "CCCN", "domain": "cell", "dataset": "ZINC", "mean": 1621.11, "std": 141.72},
            {"model": "CCXN", "domain": "cell", "dataset": "MUTAG", "mean": 10.76, "std": 1.89},
            {"model": "CCXN", "domain": "cell", "dataset": "NCI1", "mean": 244.72, "std": 46.09},
            {"model": "CCXN", "domain": "cell", "dataset": "NCI109", "mean": 225.72, "std": 86.57},
            {"model": "CCXN", "domain": "cell", "dataset": "PROTEINS", "mean": 51.98, "std": 8.5},
            {"model": "CCXN", "domain": "cell", "dataset": "ZINC", "mean": 1226.67, "std": 317.55},
            {"model": "CWN", "domain": "cell", "dataset": "MUTAG", "mean": 10.92, "std": 0.96},
            {"model": "CWN", "domain": "cell", "dataset": "NCI1", "mean": 302.34, "std": 63.44},
            {"model": "CWN", "domain": "cell", "dataset": "NCI109", "mean": 294.79, "std": 46.27},
            {"model": "CWN", "domain": "cell", "dataset": "PROTEINS", "mean": 53.6, "std": 17.7},
            {"model": "CWN", "domain": "cell", "dataset": "ZINC", "mean": 1390.21, "std": 96.86},
            {"model": "GAT", "domain": "graph", "dataset": "MUTAG", "mean": 4.16, "std": 1.05},
            {"model": "GAT", "domain": "graph", "dataset": "NCI1", "mean": 57.32, "std": 17.49},
            {"model": "GAT", "domain": "graph", "dataset": "NCI109", "mean": 56.44, "std": 9.05},
            {"model": "GAT", "domain": "graph", "dataset": "PROTEINS", "mean": 8.18, "std": 2.3},
            {"model": "GAT", "domain": "graph", "dataset": "ZINC", "mean": 171.15, "std": 64.67},
            {"model": "GCN", "domain": "graph", "dataset": "MUTAG", "mean": 3.83, "std": 0.89},
            {"model": "GCN", "domain": "graph", "dataset": "NCI1", "mean": 53.23, "std": 19.67},
            {"model": "GCN", "domain": "graph", "dataset": "NCI109", "mean": 37.4, "std": 8.63},
            {"model": "GCN", "domain": "graph", "dataset": "PROTEINS", "mean": 8.18, "std": 2.47},
            {"model": "GCN", "domain": "graph", "dataset": "ZINC", "mean": 146.89, "std": 27.63},
            {"model": "GIN", "domain": "graph", "dataset": "MUTAG", "mean": 4.6, "std": 0.56},
            {"model": "GIN", "domain": "graph", "dataset": "NCI1", "mean": 61.2, "std": 23.97},
            {"model": "GIN", "domain": "graph", "dataset": "NCI109", "mean": 50.32, "std": 7.98},
            {"model": "GIN", "domain": "graph", "dataset": "PROTEINS", "mean": 8.88, "std": 2.34},
            {"model": "GIN", "domain": "graph", "dataset": "ZINC", "mean": 168.43, "std": 107.1},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "MUTAG", "mean": 14.06, "std": 2.51},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "NCI1", "mean": 307.2, "std": 83.01},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "NCI109", "mean": 353.69, "std": 105.89},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "PROTEINS", "mean": 54.13, "std": 11.27},
            {"model": "SCCNN", "domain": "simplicial", "dataset": "ZINC", "mean": 2060.2, "std": 408.2},
            {"model": "SCN", "domain": "simplicial", "dataset": "MUTAG", "mean": 8.47, "std": 2.43},
            {"model": "SCN", "domain": "simplicial", "dataset": "NCI1", "mean": 276.32, "std": 63.21},
            {"model": "SCN", "domain": "simplicial", "dataset": "NCI109", "mean": 226.23, "std": 66.29},
            {"model": "SCN", "domain": "simplicial", "dataset": "PROTEINS", "mean": 42.78, "std": 13.41},
            {"model": "SCN", "domain": "simplicial", "dataset": "ZINC", "mean": 1209.83, "std": 571.93},
        ]
    else:
        return None
    return pd.DataFrame(data)

def parse_tt(t_c):
    if t_c == "_runtime":
        data = [
            {"model": "GCCN", "domain": "simplicial", "dataset": "MUTAG", "mean": 61, "std": 7},
            {"model": "GCCN", "domain": "simplicial", "dataset": "NCI1", "mean": 904, "std": 180},
            {"model": "GCCN", "domain": "simplicial", "dataset": "NCI109", "mean": 538, "std": 39},
            {"model": "GCCN", "domain": "simplicial", "dataset": "PROTEINS", "mean": 66, "std": 21},
            {"model": "GCCN", "domain": "simplicial", "dataset": "ZINC", "mean": 3603, "std": 475},

            {"model": "GCCN", "domain": "cell", "dataset": "MUTAG", "mean": 61, "std": 18},
            {"model": "GCCN", "domain": "cell", "dataset": "NCI1", "mean": 523, "std": 119},
            {"model": "GCCN", "domain": "cell", "dataset": "NCI109", "mean": 386, "std": 76},
            {"model": "GCCN", "domain": "cell", "dataset": "PROTEINS", "mean": 59, "std": 18},
            {"model": "GCCN", "domain": "cell", "dataset": "ZINC", "mean": 3301, "std": 440},

        ]
    else:
        return None
    return pd.DataFrame(data)


def parse_all_dfs(t_c):
    hopse_df = parse_hops(t_c)
    tb_df = parse_tb(t_c)
    tt_df = parse_tt(t_c)

    list_df = [hopse_df]
    if tb_df is not None:
        list_df.append(tb_df)
    if tt_df is not None:
        list_df.append(tt_df)

    df = pd.concat(list_df, ignore_index=True)
    if t_c == "_runtime":
        df["mean"] = df["mean"].round(0)
        df["std"] = df["std"].round(0)
    return df

if __name__ == "__main__":
    time_columns = [
        # '_runtime',
        # 'preprocessor_time',
        #'AvgTime/val_batch_mean',
        "AvgTime/val_epoch_mean",
    ]
    for t_c in time_columns:
        df = parse_all_dfs(t_c)
        table = generate_table(df, t_c)
        print(table)
        break
    #c_p, c_e, c_t = processing_times(df)
    # collected_results_time = generate_times_dictionary(df)
    # table = build_table(collected_results_time)
