import pandas as pd

from constants import optimization_metrics, keep_columns
from preprocess import preprocess_df
from generate_scores import gen_scores


def parse_hopse_results(datasets, collect_subsets):
    df_dict = {
        "model": [],
        "dataset": [],
        "mean": [],
        "std": [],
        "domain": [],
    }
    for dataset in datasets:
        aggregated = collect_subsets[dataset]
        for m_name in aggregated['model.model_name'].unique():
            for domain in aggregated['model.model_domain'].unique():
                agg_sub = aggregated[(aggregated['model.model_name']==m_name) & (aggregated['model.model_domain'] == domain)].copy()
                if len(agg_sub) == 0:
                    print(dataset, m_name, domain)
                    continue
                optim_metric = optimization_metrics[dataset]['optim_metric']
                eval_metric = optimization_metrics[dataset]['eval_metric']
                optim_dir = optimization_metrics[dataset]['direction']
                agg_sub.sort_values(by=(optim_metric,'mean'), ascending=(optim_dir == 'min'), inplace=True)

                df_dict['domain'].append(domain)
                df_dict['model'].append(m_name)
                df_dict['dataset'].append(dataset)
                df_dict['mean'].append(agg_sub.iloc[0][(eval_metric, 'mean')])
                df_dict['std'].append(agg_sub.iloc[0][(eval_metric, 'std')])
    df_res = pd.DataFrame(df_dict)
    return df_res


def parse_topotune_results():
    data = []
    data.extend(
        [
            {
                "model": "GCCN with GAT",
                "domain": "cell",
                "dataset": "MUTAG",
                "mean": 83.40,
                "std": 4.85,
            },
            {
                "model": "GCCN with GAT",
                "domain": "cell",
                "dataset": "PROTEINS",
                "mean": 74.05,
                "std": 2.16,
            },
            {
                "model": "GCCN with GAT",
                "domain": "cell",
                "dataset": "NCI1",
                "mean": 76.11,
                "std": 1.69,
            },
            {
                "model": "GCCN with GAT",
                "domain": "cell",
                "dataset": "NCI109",
                "mean": 75.62,
                "std": 0.76,
            },
            {
                "model": "GCCN with GAT",
                "domain": "cell",
                "dataset": "ZINC",
                "mean": 0.38,
                "std": 0.03,
            },
            # {'model': 'Cell with GAT', 'domain': 'cell', 'dataset': 'Cora', 'mean': 88.39, 'std': 0.65},
            # {'model': 'Cell with GAT', 'domain': 'cell', 'dataset': 'Citeseer', 'mean': 74.62, 'std': 1.95},
            # {'model': 'Cell with GAT', 'domain': 'cell', 'dataset': 'PubMed', 'mean': 87.68, 'std': 0.33},
            {
                "model": "GCCN with GCN",
                "domain": "cell",
                "dataset": "MUTAG",
                "mean": 85.11,
                "std": 6.73,
            },
            {
                "model": "GCCN with GCN",
                "domain": "cell",
                "dataset": "PROTEINS",
                "mean": 74.41,
                "std": 1.77,
            },
            {
                "model": "GCCN with GCN",
                "domain": "cell",
                "dataset": "NCI1",
                "mean": 76.42,
                "std": 1.67,
            },
            {
                "model": "GCCN with GCN",
                "domain": "cell",
                "dataset": "NCI109",
                "mean": 75.62,
                "std": 0.94,
            },
            {
                "model": "GCCN with GCN",
                "domain": "cell",
                "dataset": "ZINC",
                "mean": 0.36,
                "std": 0.01,
            },
            # {'model': 'Cell with GCN', 'domain': 'cell', 'dataset': 'Cora', 'mean': 88.51, 'std': 0.70},
            # {'model': 'Cell with GCN', 'domain': 'cell', 'dataset': 'Citeseer', 'mean': 75.41, 'std': 2.00},
            # {'model': 'Cell with GCN', 'domain': 'cell', 'dataset': 'PubMed', 'mean': 88.18, 'std': 0.26},
            {
                "model": "GCCN with GIN",
                "domain": "cell",
                "dataset": "MUTAG",
                "mean": 86.38,
                "std": 6.49,
            },
            {
                "model": "GCCN with GIN",
                "domain": "cell",
                "dataset": "PROTEINS",
                "mean": 72.54,
                "std": 3.07,
            },
            {
                "model": "GCCN with GIN",
                "domain": "cell",
                "dataset": "NCI1",
                "mean": 77.65,
                "std": 1.11,
            },
            {
                "model": "GCCN with GIN",
                "domain": "cell",
                "dataset": "NCI109",
                "mean": 77.19,
                "std": 0.21,
            },
            {
                "model": "GCCN with GIN",
                "domain": "cell",
                "dataset": "ZINC",
                "mean": 0.19,
                "std": 0.00,
            },
            # {'model': 'Cell with GIN', 'domain': 'cell', 'dataset': 'Cora', 'mean': 87.42, 'std': 1.85},
            # {'model': 'Cell with GIN', 'domain': 'cell', 'dataset': 'Citeseer', 'mean': 75.13, 'std': 1.17},
            # {'model': 'Cell with GIN', 'domain': 'cell', 'dataset': 'PubMed', 'mean': 88.47, 'std': 0.27},
            {
                "model": "GCCN with GraphSAGE",
                "domain": "cell",
                "dataset": "MUTAG",
                "mean": 85.53,
                "std": 6.80,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "cell",
                "dataset": "PROTEINS",
                "mean": 73.62,
                "std": 2.72,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "cell",
                "dataset": "NCI1",
                "mean": 78.23,
                "std": 1.47,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "cell",
                "dataset": "NCI109",
                "mean": 77.10,
                "std": 0.83,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "cell",
                "dataset": "ZINC",
                "mean": 0.24,
                "std": 0.00,
            },
            # {'model': 'Cell with GraphSAGE', 'domain': 'cell', 'dataset': 'Cora', 'mean': 88.57, 'std': 0.58},
            # {'model': 'Cell with GraphSAGE', 'domain': 'cell', 'dataset': 'Citeseer', 'mean': 75.89, 'std': 1.84},
            # {'model': 'Cell with GraphSAGE', 'domain': 'cell', 'dataset': 'PubMed', 'mean': 89.40, 'std': 0.57},
            {
                "model": "GCCN with Transformer",
                "domain": "cell",
                "dataset": "MUTAG",
                "mean": 83.83,
                "std": 6.49,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "cell",
                "dataset": "PROTEINS",
                "mean": 70.97,
                "std": 4.06,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "cell",
                "dataset": "NCI1",
                "mean": 73.00,
                "std": 1.37,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "cell",
                "dataset": "NCI109",
                "mean": 73.20,
                "std": 1.05,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "cell",
                "dataset": "ZINC",
                "mean": 0.45,
                "std": 0.02,
            },
            # {'model': 'Cell with Transformer', 'domain': 'cell', 'dataset': 'Cora', 'mean': 84.61, 'std': 1.32},
            # {'model': 'Cell with Transformer', 'domain': 'cell', 'dataset': 'Citeseer', 'mean': 75.05, 'std': 1.67},
            # {'model': 'Cell with Transformer', 'domain': 'cell', 'dataset': 'PubMed', 'mean': 88.37, 'std': 0.22},
            {
                "model": "GCCN with Hasse",
                "domain": "cell",
                "dataset": "MUTAG",
                "mean": 85.96,
                "std": 7.15,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "cell",
                "dataset": "PROTEINS",
                "mean": 73.73,
                "std": 2.95,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "cell",
                "dataset": "NCI1",
                "mean": 76.75,
                "std": 1.63,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "cell",
                "dataset": "NCI109",
                "mean": 76.94,
                "std": 0.82,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "cell",
                "dataset": "ZINC",
                "mean": 0.31,
                "std": 0.01,
            },
            # {'model': 'Cell with Hasse', 'domain': 'cell', 'dataset': 'Cora', 'mean': 87.24, 'std': 0.58},
            # {'model': 'Cell with Hasse', 'domain': 'cell', 'dataset': 'Citeseer', 'mean': 74.26, 'std': 1.47},
            # {'model': 'Cell with Hasse', 'domain': 'cell', 'dataset': 'PubMed', 'mean': 88.65, 'std': 0.55},
        ]
    )

    # Simplicial models
    data.extend(
        [
            {
                "model": "GCCN with GAT",
                "domain": "simplicial",
                "dataset": "MUTAG",
                "mean": 79.15,
                "std": 4.09,
            },
            {
                "model": "GCCN with GAT",
                "domain": "simplicial",
                "dataset": "PROTEINS",
                "mean": 74.62,
                "std": 1.95,
            },
            {
                "model": "GCCN with GAT",
                "domain": "simplicial",
                "dataset": "NCI1",
                "mean": 74.86,
                "std": 1.42,
            },
            {
                "model": "GCCN with GAT",
                "domain": "simplicial",
                "dataset": "NCI109",
                "mean": 74.81,
                "std": 1.14,
            },
            {
                "model": "GCCN with GAT",
                "domain": "simplicial",
                "dataset": "ZINC",
                "mean": 0.57,
                "std": 0.03,
            },
            # {'model': 'GCCN with GAT', 'domain': 'simplicial', 'dataset': 'Cora', 'mean': 88.33, 'std': 0.67},
            # {'model': 'GCCN with GAT', 'domain': 'simplicial', 'dataset': 'Citeseer', 'mean': 74.65, 'std': 1.93},
            # {'model': 'GCCN with GAT', 'domain': 'simplicial', 'dataset': 'PubMed', 'mean': 87.72, 'std': 0.36},
            {
                "model": "GCCN with GCN",
                "domain": "simplicial",
                "dataset": "MUTAG",
                "mean": 74.04,
                "std": 8.30,
            },
            {
                "model": "GCCN with GCN",
                "domain": "simplicial",
                "dataset": "PROTEINS",
                "mean": 74.91,
                "std": 2.51,
            },
            {
                "model": "GCCN with GCN",
                "domain": "simplicial",
                "dataset": "NCI1",
                "mean": 74.20,
                "std": 2.17,
            },
            {
                "model": "GCCN with GCN",
                "domain": "simplicial",
                "dataset": "NCI109",
                "mean": 74.13,
                "std": 0.53,
            },
            {
                "model": "GCCN with GCN",
                "domain": "simplicial",
                "dataset": "ZINC",
                "mean": 0.53,
                "std": 0.05,
            },
            # {'model': 'GCN', 'domain': 'simplicial', 'dataset': 'Cora', 'mean': 88.51, 'std': 0.70},
            # {'model': 'GCN', 'domain': 'simplicial', 'dataset': 'Citeseer', 'mean': 75.41, 'std': 2.00},
            # {'model': 'GCN', 'domain': 'simplicial', 'dataset': 'PubMed', 'mean': 88.19, 'std': 0.24},
            {
                "model": "GCCN with GIN",
                "domain": "simplicial",
                "dataset": "MUTAG",
                "mean": 85.96,
                "std": 4.66,
            },
            {
                "model": "GCCN with GIN",
                "domain": "simplicial",
                "dataset": "PROTEINS",
                "mean": 72.83,
                "std": 2.72,
            },
            {
                "model": "GCCN with GIN",
                "domain": "simplicial",
                "dataset": "NCI1",
                "mean": 76.67,
                "std": 1.62,
            },
            {
                "model": "GCCN with GIN",
                "domain": "simplicial",
                "dataset": "NCI109",
                "mean": 75.76,
                "std": 1.28,
            },
            {
                "model": "GCCN with GIN",
                "domain": "simplicial",
                "dataset": "ZINC",
                "mean": 0.35,
                "std": 0.01,
            },
            # {'model': 'GIN', 'domain': 'simplicial', 'dataset': 'Cora', 'mean': 87.27, 'std': 1.63},
            # {'model': 'GIN', 'domain': 'simplicial', 'dataset': 'Citeseer', 'mean': 75.05, 'std': 1.27},
            # {'model': 'GIN', 'domain': 'simplicial', 'dataset': 'PubMed', 'mean': 88.54, 'std': 0.21},
            {
                "model": "GCCN with GraphSAGE",
                "domain": "simplicial",
                "dataset": "MUTAG",
                "mean": 75.74,
                "std": 2.43,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "simplicial",
                "dataset": "PROTEINS",
                "mean": 74.70,
                "std": 3.10,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "simplicial",
                "dataset": "NCI1",
                "mean": 76.85,
                "std": 1.50,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "simplicial",
                "dataset": "NCI109",
                "mean": 75.64,
                "std": 1.94,
            },
            {
                "model": "GCCN with GraphSAGE",
                "domain": "simplicial",
                "dataset": "ZINC",
                "mean": 0.50,
                "std": 0.02,
            },
            # {'model': 'GraphSAGE', 'domain': 'simplicial', 'dataset': 'Cora', 'mean': 88.57, 'std': 0.59},
            # {'model': 'GraphSAGE', 'domain': 'simplicial', 'dataset': 'Citeseer', 'mean': 75.92, 'std': 1.85},
            # {'model': 'GraphSAGE', 'domain': 'simplicial', 'dataset': 'PubMed', 'mean': 89.34, 'std': 0.39},
            {
                "model": "GCCN with Transformer",
                "domain": "simplicial",
                "dataset": "MUTAG",
                "mean": 74.04,
                "std": 4.09,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "simplicial",
                "dataset": "PROTEINS",
                "mean": 70.97,
                "std": 4.06,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "simplicial",
                "dataset": "NCI1",
                "mean": 70.39,
                "std": 0.96,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "simplicial",
                "dataset": "NCI109",
                "mean": 69.99,
                "std": 1.13,
            },
            {
                "model": "GCCN with Transformer",
                "domain": "simplicial",
                "dataset": "ZINC",
                "mean": 0.64,
                "std": 0.01,
            },
            # {'model': 'Transformer', 'domain': 'simplicial', 'dataset': 'Cora', 'mean': 84.40, 'std': 1.16},
            # {'model': 'Transformer', 'domain': 'simplicial', 'dataset': 'Citeseer', 'mean': 74.60, 'std': 1.88},
            # {'model': 'Transformer', 'domain': 'simplicial', 'dataset': 'PubMed', 'mean': 88.55, 'std': 0.39},
            {
                "model": "GCCN with Hasse",
                "domain": "simplicial",
                "dataset": "MUTAG",
                "mean": 74.04,
                "std": 5.51,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "simplicial",
                "dataset": "PROTEINS",
                "mean": 74.48,
                "std": 1.89,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "simplicial",
                "dataset": "NCI1",
                "mean": 75.02,
                "std": 2.24,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "simplicial",
                "dataset": "NCI109",
                "mean": 73.91,
                "std": 3.90,
            },
            {
                "model": "GCCN with Hasse",
                "domain": "simplicial",
                "dataset": "ZINC",
                "mean": 0.56,
                "std": 0.02,
            },
            # {'model': 'Simplicial with Hasse', 'domain': 'simplicial', 'dataset': 'Cora', 'mean': 87.56, 'std': 0.66},
            # {'model': 'Simplicial with Hasse', 'domain': 'simplicial', 'dataset': 'Citeseer', 'mean': 74.50, 'std': 1.61},
            # {'model': 'Simplicial with Hasse', 'domain': 'simplicial', 'dataset': 'PubMed', 'mean': 88.61, 'std': 0.27},
        ]
    )
    topotune_df = pd.DataFrame(data)
    topotune_df["variant"] = topotune_df["model"].map(
        lambda x: x.split(" ")[-1]
    )
    topotune_df["model"] = topotune_df["model"].map(lambda x: x.split(" ")[0])

    return topotune_df


def parse_tb_results():
    # Define the raw table data
    raw_table_data = {
        # "Cora": {
        #     "GCN": (87.09, 0.20),
        #     "GIN": (87.21, 1.89),
        #     "GAT": (86.71, 0.95),
        #     "AST": (88.92, 0.44),
        #     "EDGNN": (87.06, 1.09),
        #     "UniGNN2": (86.97, 0.88),
        #     "CWN": (86.32, 1.38),
        #     "CCCN": (87.44, 1.28),
        #     "SCCNN": (82.19, 1.07),
        #     "SCN": (82.27, 1.34)
        # },
        # "Citeseer": {
        #     "GCN": (75.53, 1.27),
        #     "GIN": (73.73, 1.23),
        #     "GAT": (74.41, 1.75),
        #     "AST": (73.85, 2.21),
        #     "EDGNN": (74.93, 1.39),
        #     "UniGNN2": (74.72, 1.08),
        #     "CWN": (75.20, 1.82),
        #     "CCCN": (75.63, 1.58),
        #     "SCCNN": (70.23, 2.69),
        #     "SCN": (71.24, 1.68)
        # },
        # "Pubmed": {
        #     "GCN": (89.40, 0.30),
        #     "GIN": (89.29, 0.41),
        #     "GAT": (89.44, 0.24),
        #     "AST": (89.62, 0.25),
        #     "EDGNN": (89.04, 0.51),
        #     "UniGNN2": (89.34, 0.45),
        #     "CWN": (88.64, 0.36),
        #     "CCCN": (88.52, 0.44),
        #     "SCCNN": (88.18, 0.32),
        #     "SCN": (88.72, 0.50)
        # },
        # "Amazon": {
        #     "GCN": (49.56, 0.55),
        #     "GIN": (49.16, 1.02),
        #     "GAT": (50.17, 0.59),
        #     "AST": (50.50, 0.27),
        #     "EDGNN": (48.18, 0.09),
        #     "UniGNN2": (49.06, 0.08),
        #     "CWN": (51.90, 0.15),
        #     "CCCN": (50.26, 0.17),
        #     "SCCNN": None,
        #     "SCN": None
        # },
        # "Empire": {
        #     "GCN": (78.16, 0.32),
        #     "GIN": (79.56, 0.20),
        #     "GAT": (84.02, 0.51),
        #     "AST": (79.50, 0.13),
        #     "EDGNN": (81.01, 0.24),
        #     "UniGNN2": (77.06, 0.20),
        #     "CWN": (81.81, 0.62),
        #     "CCCN": (82.14, 0.00),
        #     "SCCNN": (89.15, 0.32),
        #     "SCN": (88.79, 0.46)
        # },
        # "Minesweeper": {
        #     "GCN": (87.52, 0.42),
        #     "GIN": (87.82, 0.34),
        #     "GAT": (89.64, 0.43),
        #     "AST": (81.14, 0.05),
        #     "EDGNN": (84.52, 0.05),
        #     "UniGNN2": (78.02, 0.00),
        #     "CWN": (88.62, 0.04),
        #     "CCCN": (89.42, 0.00),
        #     "SCCNN": (89.00, 0.00),
        #     "SCN": (90.32, 0.11)
        # },
        # "Tolokers": {
        #     "GCN": (83.02, 0.71),
        #     "GIN": (80.72, 1.19),
        #     "GAT": (84.43, 1.00),
        #     "AST": (83.26, 0.10),
        #     "EDGNN": (77.53, 0.01),
        #     "UniGNN2": (77.35, 0.03),
        #     "CWN": None,
        #     "CCCN": None,
        #     "SCCNN": None,
        #     "SCN": None
        # },
        # "Election": {
        #     "GCN": (0.31, 0.02),
        #     "GIN": (0.28, 0.02),
        #     "GAT": (0.29, 0.02),
        #     "AST": (0.29, 0.01),
        #     "EDGNN": (0.34, 0.02),
        #     "UniGNN2": (0.37, 0.02),
        #     "CWN": (0.34, 0.02),
        #     "CCCN": (0.31, 0.02),
        #     "SCCNN": (0.51, 0.03),
        #     "SCN": (0.46, 0.04)
        # },
        # "Bachelor": {
        #     "GCN": (0.29, 0.02),
        #     "GIN": (0.31, 0.03),
        #     "GAT": (0.28, 0.02),
        #     "AST": (0.30, 0.03),
        #     "EDGNN": (0.29, 0.02),
        #     "UniGNN2": (0.31, 0.02),
        #     "CWN": (0.33, 0.03),
        #     "CCCN": (0.31, 0.02),
        #     "SCCNN": (0.34, 0.03),
        #     "SCN": (0.32, 0.02)
        # },
        # "Birth": {
        #     "GCN": (0.72, 0.09),
        #     "GIN": (0.72, 0.09),
        #     "GAT": (0.71, 0.09),
        #     "AST": (0.71, 0.08),
        #     "EDGNN": (0.70, 0.07),
        #     "UniGNN2": (0.73, 0.10),
        #     "CWN": (0.72, 0.09),
        #     "CCCN": (0.71, 0.09),
        #     "SCCNN": (0.79, 0.12),
        #     "SCN": (0.71, 0.08)
        # },
        # "Death": {
        #     "GCN": (0.51, 0.04),
        #     "GIN": (0.52, 0.04),
        #     "GAT": (0.51, 0.04),
        #     "AST": (0.49, 0.05),
        #     "EDGNN": (0.52, 0.05),
        #     "UniGNN2": (0.51, 0.05),
        #     "CWN": (0.54, 0.06),
        #     "CCCN": (0.54, 0.06),
        #     "SCCNN": (0.55, 0.05),
        #     "SCN": (0.52, 0.05)
        # },
        # "Income": {
        #     "GCN": (0.22, 0.03),
        #     "GIN": (0.21, 0.02),
        #     "GAT": (0.20, 0.02),
        #     "AST": (0.21, 0.02),
        #     "EDGNN": (0.23, 0.02),
        #     "UniGNN2": (0.23, 0.02),
        #     "CWN": (0.25, 0.03),
        #     "CCCN": (0.23, 0.02),
        #     "SCCNN": (0.28, 0.03),
        #     "SCN": (0.25, 0.02)
        # },
        # "Migration": {
        #     "GCN": (0.80, 0.12),
        #     "GIN": (0.80, 0.10),
        #     "GAT": (0.77, 0.13),
        #     "AST": (0.78, 0.12),
        #     "EDGNN": (0.80, 0.12),
        #     "UniGNN2": (0.79, 0.12),
        #     "CWN": (0.84, 0.13),
        #     "CCCN": (0.84, 0.12),
        #     "SCCNN": (0.90, 0.14),
        #     "SCN": (0.92, 0.20)
        # },
        # "Unempl": {
        #     "GCN": (0.25, 0.03),
        #     "GIN": (0.22, 0.02),
        #     "GAT": (0.23, 0.03),
        #     "AST": (0.22, 0.02),
        #     "EDGNN": (0.26, 0.03),
        #     "UniGNN2": (0.28, 0.02),
        #     "CWN": (0.25, 0.03),
        #     "CCCN": (0.24, 0.03),
        #     "SCCNN": (0.43, 0.04),
        #     "SCN": (0.38, 0.04)
        # },
        "MUTAG": {
            "GCN": (69.79, 6.80),
            "GIN": (79.57, 6.13),
            "GAT": (72.77, 2.77),
            "AST": (71.06, 6.49),
            "EDGNN": (80.00, 4.90),
            "UniGNN2": (80.43, 4.09),
            "CWN": (80.43, 1.78),
            "CCCN": (77.02, 9.32),
            "SCCNN": (76.17, 6.63),
            "SCN": (73.62, 6.13),
        },
        "PROTEINS": {
            "GCN": (75.70, 2.14),
            "GIN": (75.20, 3.30),
            "GAT": (76.34, 1.66),
            "AST": (76.63, 1.74),
            "EDGNN": (73.91, 4.39),
            "UniGNN2": (75.20, 2.96),
            "CWN": (76.13, 2.70),
            "CCCN": (73.33, 2.30),
            "SCCNN": (74.19, 2.86),
            "SCN": (75.27, 2.14),
        },
        "NCI1": {
            "GCN": (72.86, 0.69),
            "GIN": (74.26, 0.96),
            "GAT": (75.00, 0.99),
            "AST": (75.18, 1.24),
            "EDGNN": (73.97, 0.82),
            "UniGNN2": (73.02, 0.92),
            "CWN": (73.93, 1.87),
            "CCCN": (76.67, 1.48),
            "SCCNN": (76.60, 1.75),
            "SCN": (74.49, 1.03),
        },
        "NCI109": {
            "GCN": (72.20, 1.22),
            "GIN": (74.42, 0.70),
            "GAT": (73.80, 0.73),
            "AST": (73.75, 1.09),
            "EDGNN": (74.93, 2.50),
            "UniGNN2": (70.76, 1.11),
            "CWN": (73.80, 2.06),
            "CCCN": (75.35, 1.50),
            "SCCNN": (77.12, 1.07),
            "SCN": (75.70, 1.04),
        },
        "IMDB-BINARY": {
            "GCN": (72.00, 2.48),
            "GIN": (70.96, 1.93),
            "GAT": (69.76, 2.65),
            "AST": (70.32, 3.27),
            "EDGNN": (69.12, 2.92),
            "UniGNN2": (71.04, 1.31),
            "CWN": (70.40, 2.02),
            "CCCN": (69.12, 2.82),
            "SCCNN": (70.88, 2.25),
            "SCN": (70.80, 2.38),
        },
        "IMDB-MULTI": {
            "GCN": (49.97, 2.16),
            "GIN": (47.68, 4.21),
            "GAT": (50.13, 3.87),
            "AST": (50.51, 2.92),
            "EDGNN": (49.17, 4.35),
            "UniGNN2": (49.76, 3.55),
            "CWN": (49.71, 2.83),
            "CCCN": (47.79, 3.45),
            "SCCNN": (48.75, 3.98),
            "SCN": (49.49, 5.08),
        },
        # "REDDIT": {
        #     "GCN": (76.24, 0.54),
        #     "GIN": (81.96, 1.36),
        #     "GAT": (75.68, 1.00),
        #     "AST": (74.84, 2.68),
        #     "EDGNN": (83.24, 1.45),
        #     "UniGNN2": (75.56, 3.19),
        #     "CWN": (85.52, 1.38),
        #     "CCCN": (85.12, 1.29),
        #     "SCCNN": (77.24, 1.87),
        #     "SCN": (71.28, 2.06)
        # },
        "ZINC": {
            "GCN": (0.62, 0.01),
            "GIN": (0.57, 0.04),
            "GAT": (0.61, 0.01),
            "AST": (0.59, 0.02),
            "EDGNN": (0.51, 0.01),
            "UniGNN2": (0.60, 0.01),
            "CWN": (0.34, 0.01),
            "CCCN": (0.34, 0.02),
            "SCCNN": (0.36, 0.02),
            "SCN": (0.53, 0.04),
        },
    }

    # Process the data to select the best of DR and SDP for each method
    additional_data = []

    for dataset, entries in raw_table_data.items():
        optim_dir = optimization_metrics[dataset]["direction"]

        # Group data by method prefix (before the underscore)
        method_results = {}
        standard_methods = []
        for method, (mean, std) in entries.items():
            if method in ["CWN", "CCCN", "SCCNN", "SCN", "GCN", "GIN", "GAT"]:
                # These are the standard methods - just add them directly
                standard_methods.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "mean": mean,
                        "std": std,
                    }
                )

        # Add all standard methods
        additional_data.extend(standard_methods)

    # Map method names to their proper format in the data
    method_mappings = {
        "CWN": "CWN",
        "CCCN": "CCNN",  # Cell CCNN
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
                "mean": item["mean"],
                "std": item["std"],
            }
        )

    # This data can now be added to your existing dataframe or used to create a new one
    tbx_df = pd.DataFrame(formatted_data)

    return tbx_df


def parse_all_dfs(selected_datasets=[]):
    df = pd.read_csv("merged_csv/merged_normalized.csv")
    df = preprocess_df(df)
    # Keep only relevant columns
    df = df[keep_columns]
    # Generate best scores per hyperparameter sweep
    scores = gen_scores(df)

    df_hopse = parse_hopse_results(selected_datasets, scores)
    df_topotune = parse_topotune_results()
    df_tb = parse_tb_results()
    cat_df = pd.concat([df_hopse, df_topotune, df_tb], ignore_index=True)

    # FIX Hopse naming
    cat_df["model"][cat_df["model"] == "HOPSE_MANUAL_PE"] = "HOPSE-M"
    cat_df["model"][cat_df["model"] == "HOPSE_GPSE"] = "HOPSE-G"

    # Only grab the datasets we are interested in
    filtered_df = cat_df[cat_df["dataset"].isin(selected_datasets)]

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

    # 1) Escape underscores in model names
    df["model"] = df["model"].str.replace("_", r"\_", regex=False)

    # 2) Among multiple variants for (domain, dataset, model), pick best according to direction
    def pick_best_variant(group):
        dataset = group["dataset"].iloc[0]
        direction = optimization_metrics.get(dataset, {}).get(
            "direction", "max"
        )
        if direction == "min":
            return group.loc[group["mean"].idxmin()]
        else:
            return group.loc[group["mean"].idxmax()]

    grouped = df.groupby(["domain", "dataset", "model"], group_keys=False)
    df_best = grouped.apply(pick_best_variant).reset_index(drop=True)

    # 3) Split into MANTRA vs. Other
    mantra_dsets = ["MANTRA_name", "MANTRA_orientation", "MANTRA_betti_numbers"]
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
        all_datasets = sorted(subset_df["dataset"].unique())

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
                r"($\uparrow$)" if directions[dset] == "max" else r"($\downarrow$)"
            )
            header_cells.append(r"\scriptsize " + dset + " " + arrow)
        latex_lines.append(" & ".join(header_cells) + r" \\")

        # sort domains to have consistent ordering
        all_domains = sorted(domain_groups.keys())
        if len(all_domains) == 3:
            all_domains = ["graph", "simplicial", "cell"]

        # For each domain, we do the "sandwiching" with midrules
        for dom in all_domains:
            dom_df = domain_groups[dom]
            dom_models = sorted(dom_df["model"].unique())
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
        latex_lines.append(r"\caption{" + caption_text + "}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)

    # Build the two separate tables
    latex_mantra = build_table(
        df_mantra,
        "Results for MANTRA datasets (MANTRA-N, MANTRA-O, MANTRA-BN).",
    )
    latex_others = build_table(df_other, "Results for all other datasets.")

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
        "MANTRA_orientation",
        "MANTRA_name",
        "MANTRA_betti_numbers"
    ]

    # Parse the dataframes
    df = parse_all_dfs(selected_datasets)

    # Generate the LaTeX table
    latex_table = generate_table(df, optimization_metrics)
    print(latex_table)
