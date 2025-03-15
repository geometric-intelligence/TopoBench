""" Test the TBEvaluator class."""
import pytest
import torch
from topobench.evaluator import TBEvaluator

class TestTBEvaluator:
    """ Test the TBXEvaluator class."""

    def setup_method(self):
        """ Setup the test."""
        self.classification_metrics = ["accuracy", "precision", "recall", "auroc"]
        self.evaluator_classification = TBEvaluator(task="classification", num_classes=3, metrics=self.classification_metrics)
        #self.evaluator_multilabel = TBEvaluator(task="multilabel classification", num_classes=2, metrics=self.classification_metrics)
        self.regression_metrics = ["example", "mae"]
        self.evaluator_regression = TBEvaluator(task="regression", num_classes=1, metrics=self.regression_metrics)
       
        
    def test_repr(self):
        """Test the __repr__ method."""
        assert "TBEvaluator" in self.evaluator_classification.__repr__()
        #assert "TBEvaluator" in self.evaluator_multilabel.__repr__()
        assert "TBEvaluator" in self.evaluator_regression.__repr__()

    def test_update_and_compute(self):
        """Test the update and compute methods."""
        self.evaluator_classification.update({"logits": torch.randn(10, 3), "labels": torch.randint(0, 3, (10,))})
        out = self.evaluator_classification.compute()
        for metric in self.classification_metrics:
            assert metric in out
        #self.evaluator_multilabel.update({"logits": torch.randn(10, 2), "labels": torch.randint(0, 2, (10, 2))})
        #out = self.evaluator_multilabel.compute()
        for metric in self.classification_metrics:
            assert metric in out
        self.evaluator_regression.update({"logits": torch.randn(10, 1), "labels": torch.randn(10,)})
        out = self.evaluator_regression.compute()
        for metric in self.regression_metrics:
            assert metric in out
    
    def test_not_implemented(self):
        """Test not implemented methods."""
        with pytest.raises(ValueError):
            TBEvaluator(task="wrong", num_classes=2, metrics=self.classification_metrics)
        
        evaluator = TBEvaluator(task="multilabel classification", num_classes=3, metrics=self.classification_metrics)
        with pytest.raises(NotImplementedError):
            evaluator.update({"logits": torch.tensor([[1,0,0], [0,1,1]]), "labels": torch.tensor([[1,1,0]])})

        
    def test_reset(self):
        """Test the reset method."""
        #self.evaluator_multilabel.reset()
        self.evaluator_regression.reset()
