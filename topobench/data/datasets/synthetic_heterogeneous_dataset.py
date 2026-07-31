"""Deterministic native heterogeneous graph fixture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, InMemoryDataset


def _stratified_masks(
    labels: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic train, validation, and test masks by class.

    Parameters
    ----------
    labels : torch.Tensor
        One-dimensional integer class labels.
    generator : torch.Generator
        Local random generator used to shuffle each class independently.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Boolean train, validation, and test masks, respectively.

    Raises
    ------
    ValueError
        If a class has too few examples to populate all three splits.
    """
    masks = [torch.zeros(labels.numel(), dtype=torch.bool) for _ in range(3)]
    for class_id in labels.unique(sorted=True):
        indices = (labels == class_id).nonzero(as_tuple=False).view(-1)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_end = max(1, int(0.6 * indices.numel()))
        val_end = train_end + max(1, int(0.2 * indices.numel()))
        if val_end >= indices.numel():
            raise ValueError(
                "Each synthetic class needs train, validation, and test nodes"
            )
        masks[0][indices[:train_end]] = True
        masks[1][indices[train_end:val_end]] = True
        masks[2][indices[val_end:]] = True
    return masks[0], masks[1], masks[2]


def make_synthetic_heterogeneous_data(
    *,
    seed: int = 0,
    num_authors: int = 36,
    num_papers: int = 24,
    num_venues: int = 6,
) -> HeteroData:
    """Build a small deterministic heterogeneous node-classification graph.

    The graph contains labeled authors, featured papers, and featureless
    venues. Author labels are recoverable from both author features and the
    paper features reached through ``writes`` edges, making the graph suitable
    for correctness and overfitting diagnostics.

    Parameters
    ----------
    seed : int, default=0
        Seed for the fixture-local random generator.
    num_authors : int, default=36
        Even number of author nodes. Must be at least 12.
    num_papers : int, default=24
        Number of paper nodes. Must be divisible by four and no smaller than
        ``num_venues``.
    num_venues : int, default=6
        Number of featureless venue nodes. Must be at least two.

    Returns
    -------
    HeteroData
        Native PyG heterogeneous data with forward-only typed relations.

    Raises
    ------
    ValueError
        If the requested node counts violate the fixture size contract.
    """
    if num_authors < 12 or num_authors % 2:
        raise ValueError("num_authors must be even and at least 12")
    if num_papers < num_venues or num_venues < 2 or num_papers % 4:
        raise ValueError(
            "Require num_papers divisible by four and "
            "num_papers >= num_venues >= 2"
        )

    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_authors, dtype=torch.long) % 2
    train_mask, val_mask, test_mask = _stratified_masks(
        labels,
        generator=generator,
    )

    author_x = 0.1 * torch.randn(num_authors, 8, generator=generator)
    author_x[:, :2] += 2.0 * F.one_hot(labels, num_classes=2).float()
    paper_x = 0.1 * torch.randn(num_papers, 5, generator=generator)
    paper_signal = (torch.arange(num_papers) // 2) % 2
    paper_x[:, :2] += (
        1.5
        * F.one_hot(
            paper_signal,
            num_classes=2,
        ).float()
    )

    author_ids = torch.arange(num_authors).repeat_interleave(2)
    write_slot = torch.arange(2).repeat(num_authors)
    paper_ids = (2 * author_ids + write_slot) % num_papers

    covered_papers = torch.zeros(num_papers, dtype=torch.bool)
    covered_papers[paper_ids] = True
    uncovered_paper_ids = (~covered_papers).nonzero(as_tuple=False).view(-1)
    if uncovered_paper_ids.numel():
        coverage_author_ids = paper_signal[uncovered_paper_ids]
        author_ids = torch.cat([author_ids, coverage_author_ids])
        paper_ids = torch.cat([paper_ids, uncovered_paper_ids])

    data = HeteroData()
    data["author"].x = author_x
    data["author"].y = labels
    data["author"].train_mask = train_mask
    data["author"].val_mask = val_mask
    data["author"].test_mask = test_mask
    data["paper"].x = paper_x
    data["venue"].num_nodes = num_venues
    data["author", "writes", "paper"].edge_index = torch.stack(
        [author_ids, paper_ids]
    )
    data["paper", "published_in", "venue"].edge_index = torch.stack(
        [
            torch.arange(num_papers),
            torch.arange(num_papers) % num_venues,
        ]
    )
    data.validate(raise_on_error=True)
    return data


class SyntheticHeterogeneousDataset(InMemoryDataset):
    """Wrap one deterministic native heterogeneous graph as a PyG dataset.

    Parameters
    ----------
    **kwargs : int
        Keyword arguments forwarded to
        :func:`make_synthetic_heterogeneous_data`.
    """

    def __init__(self, **kwargs: int) -> None:
        super().__init__(root=None)
        data = make_synthetic_heterogeneous_data(**kwargs)
        self.data, self.slices = self.collate([data])
