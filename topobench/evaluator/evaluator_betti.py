"""This module contains the Evaluator class that is responsible for computing the metrics."""

from torchmetrics import MetricCollection

from topobench.evaluator import METRICS, AbstractEvaluator


class BettiEvaluator(AbstractEvaluator):
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

        # In the case of betti numbers we break the multilabel classification into multiple multiclass classifications with 2 classes (following the original paper's approach)
        parameters = {}  # "num_classes": 2 Every betti number either 0 or 1
        parameters["task"] = "binary"
        self.metric_names = kwargs["metrics"]

        metrics = {}
        for name_betti in self.metric_names:
            name, betti = name_betti.split("_")
            if name in ["recall", "precision", "auroc", "f1"]:
                metrics[name_betti] = METRICS[name](
                    average="macro", **parameters
                )

            else:
                metrics[name_betti] = METRICS[name](**parameters)
        self.metrics = MetricCollection(metrics)

        self.best_metric = {}

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

        Raises
        ------
        ValueError
            If the task is not valid.
        """
        preds = model_out["logits"].cpu()
        target = model_out["labels"].cpu()

        for metric_name in self.metric_names:
            _, betti = metric_name.split("_")
            self.metrics[metric_name].update(
                preds[:, int(betti)], target[:, int(betti)]
            )

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
