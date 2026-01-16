"""Tests for the LinkPredictionReadOut layer."""

import pytest
import torch
import torch_geometric.data as tg_data

from topobench.nn.readouts.link_prediction_readout import LinkPredictionReadOut


class TestLinkPredictionReadOut:
    """Tests for the LinkPredictionReadOut layer."""

    @pytest.fixture
    def base_kwargs(self):
        """Fixture providing the required base parameters.

        Returns
        -------
        dict
            Base keyword arguments for LinkPredictionReadOut.
        """
        return {
            "hidden_dim": 16,
            "out_channels": 2,
            "task_level": "edge",
            "pooling_type": "sum",
            "logits_linear_layer": True,
        }

    @pytest.fixture
    def readout_layer(self, base_kwargs):
        """Fixture to create a LinkPredictionReadOut instance.

        Parameters
        ----------
        base_kwargs : dict
            Base keyword arguments for LinkPredictionReadOut.

        Returns
        -------
        LinkPredictionReadOut
            Instantiated readout layer.
        """
        return LinkPredictionReadOut(**base_kwargs)

    @pytest.fixture
    def sample_batch(self):
        """Fixture to create a sample batch of edge-level data.

        Returns
        -------
        torch_geometric.data.Data
            Batch with edge_label_index and edge_label.
        """
        # 5 nodes, simple edge set of 4 edges
        num_nodes = 5
        edge_label_index = torch.tensor([[0, 1, 2, 3],
                                         [1, 2, 3, 4]])
        # First two are positive, last two are negative
        edge_label = torch.tensor([1, 1, 0, 0])

        batch = tg_data.Data(
            num_nodes=num_nodes,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
        )
        return batch

    @pytest.fixture
    def sample_model_out(self, base_kwargs):
        """Fixture to create a sample model_out dict with node embeddings.

        Parameters
        ----------
        base_kwargs : dict
            Base keyword arguments containing hidden_dim.

        Returns
        -------
        dict
            Dictionary with key 'x_0' containing node embeddings.
        """
        num_nodes = 5
        hidden_dim = base_kwargs["hidden_dim"]
        x_0 = torch.randn(num_nodes, hidden_dim)
        return {"x_0": x_0}

    def test_initialization_valid(self, base_kwargs):
        """Test successful initialization with valid parameters.

        Parameters
        ----------
        base_kwargs : dict
            Base keyword arguments for LinkPredictionReadOut.
        """
        readout = LinkPredictionReadOut(**base_kwargs)
        assert isinstance(readout, LinkPredictionReadOut)
        assert readout.hidden_dim == base_kwargs["hidden_dim"]
        assert readout.out_channels == base_kwargs["out_channels"]
        assert readout.task_level == "edge"

    def test_invalid_task_level_raises(self, base_kwargs):
        """Test that invalid task_level raises ValueError.

        Parameters
        ----------
        base_kwargs : dict
            Base keyword arguments for LinkPredictionReadOut.
        """
        bad_kwargs = base_kwargs.copy()
        bad_kwargs["task_level"] = "node"
        with pytest.raises(ValueError, match="intended for 'edge' tasks"):
            LinkPredictionReadOut(**bad_kwargs)

    def test_repr(self, readout_layer):
        """Test string representation of the readout layer.

        Parameters
        ----------
        readout_layer : LinkPredictionReadOut
            Instantiated readout layer.
        """
        rep = repr(readout_layer)
        assert "LinkPredictionReadOut" in rep
        assert "hidden_dim=" in rep
        assert "out_channels=" in rep
        assert "task_level='edge'" in rep

    def test_forward_basic_shapes(
        self,
        readout_layer,
        sample_model_out,
        sample_batch,
    ):
        """Test that forward pass returns logits and labels with correct shapes.

        Parameters
        ----------
        readout_layer : LinkPredictionReadOut
            Readout layer under test.
        sample_model_out : dict
            Dictionary with node embeddings under 'x_0'.
        sample_batch : torch_geometric.data.Data
            Batch with edge_label_index and edge_label.
        """
        output = readout_layer(sample_model_out, sample_batch)

        logits = output["logits"]
        labels = output["labels"]

        # 4 edges, 2-class logits
        assert logits.shape == (4, 2)
        assert labels.shape == (4,)
        # Labels are integers 0/1
        assert labels.dtype == torch.long
        assert set(labels.tolist()) <= {0, 1}

    def test_forward_computes_correct_scores(self):
        """Test that the dot-product scores are embedded correctly into logits.

        Constructs a tiny example with deterministic embeddings.
        """
        hidden_dim = 3
        readout = LinkPredictionReadOut(
            hidden_dim=hidden_dim,
            out_channels=2,
            task_level="edge",
        )

        # 3 nodes in 3D
        x_0 = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # node 0
                [0.0, 1.0, 0.0],  # node 1
                [1.0, 1.0, 0.0],  # node 2
            ]
        )
        model_out = {"x_0": x_0}

        # Edges: (0,2) and (1,2)
        edge_label_index = torch.tensor([[0, 1],
                                         [2, 2]])
        edge_label = torch.tensor([1, 0])
        batch = tg_data.Data(
            num_nodes=3,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
        )

        output = readout(model_out, batch)
        logits = output["logits"]

        # Scores are dot products:
        # score(0,2) = [1,0,0]·[1,1,0] = 1
        # score(1,2) = [0,1,0]·[1,1,0] = 1
        expected_score = torch.tensor([1.0, 1.0])
        expected_logits = torch.stack([-expected_score, expected_score], dim=-1)

        assert torch.allclose(logits, expected_logits)

    def test_missing_x0_raises_keyerror(self, sample_batch):
        """Test that missing 'x_0' in model_out raises KeyError.

        Parameters
        ----------
        sample_batch : torch_geometric.data.Data
            Batch with edge-level labels.
        """
        readout = LinkPredictionReadOut(
            hidden_dim=8,
            out_channels=2,
            task_level="edge",
        )
        model_out = {}  # no 'x_0'

        with pytest.raises(KeyError, match="Expected node embeddings 'x_0'"):
            readout(model_out, sample_batch)

    def test_missing_edge_label_index_raises(self, sample_model_out):
        """Test that missing edge_label_index raises AttributeError."""
        readout = LinkPredictionReadOut(
            hidden_dim=8,
            out_channels=2,
            task_level="edge",
        )

        # Batch without edge_label_index
        batch = tg_data.Data(
            num_nodes=4,
            edge_label=torch.tensor([1, 0, 1]),
        )

        with pytest.raises(AttributeError, match="edge_label_index"):
            readout(sample_model_out, batch)

    def test_missing_edge_label_raises(self, sample_model_out):
        """Test that missing edge_label raises AttributeError."""
        readout = LinkPredictionReadOut(
            hidden_dim=8,
            out_channels=2,
            task_level="edge",
        )

        # Batch without edge_label
        edge_label_index = torch.tensor([[0, 1], [1, 2]])
        batch = tg_data.Data(
            num_nodes=3,
            edge_label_index=edge_label_index,
        )

        with pytest.raises(AttributeError, match="edge_label"):
            readout(sample_model_out, batch)