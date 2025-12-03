"""Callback to compute and log data statistics for datasets."""

import lightning.pytorch as pl
import torch


class DataStatisticsCallback(pl.Callback):
    """Callback to log statistics of the lifted dataset.

    Parameters
    ----------
    domain : str
        Data domain. Only 'hypergraph' is currently implemented.
    """

    def __init__(self, domain: str):
        """Initialize the DataStatisticsCallback.

        Parameters
        ----------
        domain : str
            The domain of the data (must be 'hypergraph').
        """
        self.domain = domain

        if self.domain != "hypergraph":
            raise NotImplementedError(
                f"DataStatisticsCallback is not implemented for domain: {self.domain}"
            )

    def on_fit_start(self, trainer, pl_module):
        """Calculate and log statistics before the training loop begins.

        Parameters
        ----------
        trainer : pl.Trainer
            Lightning trainer.
        pl_module : pl.LightningModule
            Here for compatibility with lightining callbacks.
        """
        # Only calculate on the main process to avoid duplicate logging in DDP
        if not trainer.is_global_zero:
            return
        # 1. Access the dataset
        if trainer.datamodule is not None:
            # Try accessing the attribute directly, fallback to the loader
            if hasattr(trainer.datamodule, "train_dataset"):
                dataset = trainer.datamodule.train_dataset
            else:
                dataset = trainer.datamodule.train_dataloader().dataset
        else:
            dataset = trainer.train_dataloader.dataset

        print(
            f"[Statistics] Computing hypergraph statistics for {len(dataset)} samples..."
        )

        # Containers for aggregation
        all_he_sizes = []  # Cardinality of hyperedges
        all_node_degrees = []  # Degree of nodes
        total_nodes = 0
        total_hyperedges = 0

        # Iterate dataset
        for data in dataset:
            try:
                incidence_idx = data[1].index("incidence_hyperedges")
            except ValueError:
                continue

            H = data[0][incidence_idx]

            # 1. Basic Counts
            n_nodes, n_edges = H.shape
            total_nodes += n_nodes
            total_hyperedges += n_edges

            # 2. Extract Indices for efficient counting
            if H.is_sparse:
                indices = H.coalesce().indices()
                row_indices = indices[0]  # Nodes
                col_indices = indices[1]  # Hyperedges
            else:
                row_indices, col_indices = torch.nonzero(H, as_tuple=True)

            # 3. Calculate Hyperedge Sizes (Cardinality)
            he_sizes = torch.bincount(col_indices, minlength=n_edges).float()
            all_he_sizes.append(he_sizes)

            # 4. Calculate Node Degrees
            node_degrees = torch.bincount(
                row_indices, minlength=n_nodes
            ).float()
            all_node_degrees.append(node_degrees)

        # Concatenate all stats to compute global distributions
        if len(all_he_sizes) > 0:
            flat_he_sizes = torch.cat(all_he_sizes)
            flat_node_degrees = torch.cat(all_node_degrees)

            # --- Compute Statistics ---
            stats = {
                # Global Counts
                "stats/total_graphs": float(len(dataset)),
                "stats/total_nodes": float(total_nodes),
                "stats/total_hyperedges": float(total_hyperedges),
                # Hyperedge Cardinality (Size)
                "stats/he_size_avg": flat_he_sizes.mean().item(),
                "stats/he_size_std": flat_he_sizes.std().item(),
                "stats/he_size_median": flat_he_sizes.median().item(),
                "stats/he_size_max": flat_he_sizes.max().item(),
                # Node Degree (How many hyperedges a node belongs to)
                "stats/node_deg_avg": flat_node_degrees.mean().item(),
                "stats/node_deg_std": flat_node_degrees.std().item(),
                "stats/node_deg_median": flat_node_degrees.median().item(),
                # Global Density (Sparsity)
                # (Num_incidences) / (Num_Nodes * Num_Edges)
                "stats/global_sparsity": flat_he_sizes.sum().item()
                / (total_nodes * total_hyperedges)
                if total_nodes > 0
                else 0.0,
            }

            # --- Logging ---
            print("[Statistics] Logging the following metrics:")
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}")

            if trainer.logger is not None:
                # We pass step=0 so the logger knows where to place this point
                trainer.logger.log_metrics(stats, step=0)

        else:
            print(
                "[Statistics] Warning: No valid hypergraph data found to calculate statistics."
            )
