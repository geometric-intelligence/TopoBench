"""This module contains the Evaluator class that is responsible for computing the metrics."""

from torchmetrics import MetricCollection

from topobench.evaluator import METRICS, AbstractEvaluator


class TBEvaluator(AbstractEvaluator):
    r"""Evaluator class that is responsible for computing the metrics.

    Parameters
    ----------
    task : str
        The task type. It can be either "classification" or "regression".
    **kwargs : dict
        Additional arguments for the class. The arguments depend on the task.
        In "classification" scenario, the following arguments are expected:
        - num_classes (int): The number of classes.
        - metrics (list[str]): A list of classification metrics to be computed.
        In "regression" scenario, the following arguments are expected:
        - metrics (list[str]): A list of regression metrics to be computed.
    """

    def __init__(self, task, **kwargs):
        # Define the task
        self.task = task

        # Define the metrics depending on the task
        if kwargs["num_classes"] > 1 and self.task == "classification":
            # Note that even for binary classification, we use multiclass metrics
            # According to the torchmetrics documentation (https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#torchmetrics.classification.MulticlassAccuracy)
            # This setup should work correctly
            parameters = {"num_classes": kwargs["num_classes"]}
            parameters["task"] = "multiclass"
            metric_names = kwargs["metrics"]

        elif self.task == "multilabel classification":
            parameters = {"num_classes": kwargs["num_classes"]}
            parameters["task"] = "multilabel"
            parameters["num_labels"] = kwargs["num_classes"]
            metric_names = kwargs["metrics"]

        elif self.task == "regression":
            parameters = {}
            metric_names = kwargs["metrics"]

        else:
            raise ValueError(f"Invalid task {task}")

        metrics = {}
        for name in metric_names:
            if name in ["recall", "precision", "auroc", "f1", "f1_macro"]:
                metrics[name] = METRICS[name](average="macro", **parameters)
            elif name == "f1_weighted":
                metrics[name] = METRICS[name](average="weighted", **parameters)
            elif name == "confusion_matrix":
                metrics[name] = METRICS[name](**parameters)
            elif name == "rmse":
                # RMSE is MSE with squared=False
                metrics[name] = METRICS[name](squared=False, **parameters)
            else:
                metrics[name] = METRICS[name](**parameters)
        self.metrics = MetricCollection(metrics)

        self.best_metric = {}

        # Initialize comprehensive best metrics tracking
        self.best_metrics = {}
        self.metric_optimization_direction = {
            # Metrics to maximize (higher is better)
            "accuracy": "max",
            "precision": "max",
            "recall": "max",
            "auroc": "max",
            "f1": "max",
            "f1_macro": "max",
            "f1_weighted": "max",
            "r2": "max",
            # Metrics to minimize (lower is better)
            "mae": "min",
            "mse": "min",
            "rmse": "min",
            "denormalized_rmse": "min",
            "loss": "min",
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, metrics={self.metrics})"

    def update(self, model_out: dict):
        r"""Update the metrics with the model output.

        Parameters
        ----------
        model_out : dict
            The model output. It should contain the following keys:
            - logits : torch.Tensor
            The model predictions.
            - labels : torch.Tensor
            The ground truth labels.
            - batch : torch_geometric.data.Data (optional)
            The batch data containing target normalizer stats.

        Raises
        ------
        ValueError
            If the task is not valid.
        """
        preds = model_out["logits"].cpu()
        target = model_out["labels"].cpu()

        if self.task == "regression":
            try:
                self.metrics.update(preds, target.unsqueeze(1))
            except RuntimeError:
                self.metrics.update(preds.unsqueeze(1), target.unsqueeze(1))

        elif self.task == "classification":
            self.metrics.update(preds, target)

        elif self.task == "multilabel classification":
            # Raise not supported error
            raise NotImplementedError(
                "Multilabel classification is not supported yet"
            )

        else:
            raise ValueError(f"Invalid task {self.task}")

    def compute(self):
        r"""Compute the metrics.

        Returns
        -------
        dict
            Dictionary containing the computed metrics.
        """
        return self.metrics.compute()

    def reset(self):
        """Reset the metrics.

        This method should be called after each epoch.
        """
        self.metrics.reset()

    def update_best_metrics(self, metrics_dict: dict, mode: str) -> None:
        r"""Update best metrics tracking.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary containing the computed metrics.
        mode : str
            The mode of the model, either "train", "val", or "test".
        """
        for key, value in metrics_dict.items():
            metric_key = f"{mode}/{key}"

            # Convert tensor to float if needed
            if hasattr(value, "item"):
                value = value.item()
            elif hasattr(value, "cpu"):
                value = value.cpu().item()

            # Initialize best value if not exists
            if metric_key not in self.best_metrics:
                self.best_metrics[metric_key] = value

            # Determine optimization direction for this metric
            optimization_direction = self.metric_optimization_direction.get(
                key, "max"
            )

            # Update best value based on optimization direction
            if optimization_direction == "max":
                if value > self.best_metrics[metric_key]:
                    self.best_metrics[metric_key] = value
            else:  # min
                if value < self.best_metrics[metric_key]:
                    self.best_metrics[metric_key] = value

    def update_best_loss(self, loss_value: float, mode: str) -> None:
        r"""Update best loss tracking.

        Parameters
        ----------
        loss_value : float
            The current loss value.
        mode : str
            The mode of the model, either "train", "val", or "test".
        """
        metric_key = f"{mode}/loss"

        # Initialize best value if not exists
        if metric_key not in self.best_metrics:
            self.best_metrics[metric_key] = loss_value

        # Loss should be minimized
        if loss_value < self.best_metrics[metric_key]:
            self.best_metrics[metric_key] = loss_value

    def get_best_metrics(self) -> dict:
        r"""Get the best metrics achieved so far.

        Returns
        -------
        dict
            Dictionary containing the best metrics for each mode and metric.
        """
        return self.best_metrics.copy()

    def log_best_metrics_summary(self) -> None:
        r"""Log a summary of all best metrics achieved during training."""
        if self.best_metrics:
            print("\n" + "=" * 60)
            print("BEST METRICS ACHIEVED DURING TRAINING")
            print("=" * 60)

            # Group metrics by mode
            train_metrics = {
                k: v
                for k, v in self.best_metrics.items()
                if k.startswith("train/")
            }
            val_metrics = {
                k: v
                for k, v in self.best_metrics.items()
                if k.startswith("val/")
            }
            test_metrics = {
                k: v
                for k, v in self.best_metrics.items()
                if k.startswith("test/")
            }

            for mode, metrics in [
                ("TRAINING", train_metrics),
                ("VALIDATION", val_metrics),
                ("TEST", test_metrics),
            ]:
                if metrics:
                    print(f"\n{mode} METRICS:")
                    print("-" * 40)
                    for metric_key, best_value in sorted(metrics.items()):
                        metric_name = metric_key.split("/", 1)[1]
                        optimization = self.metric_optimization_direction.get(
                            metric_name, "max"
                        )
                        print(
                            f"  Best {metric_name}: {best_value:.6f} ({optimization})"
                        )

            print("=" * 60)
