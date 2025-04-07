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
        self.metric_names = kwargs["metrics"]
        self.num_betti_numbers = kwargs["num_betti_numbers"]

        # In the case of betti numbers we break the multilabel classification into multiple multiclass classifications with 2 classes (following the original paper's approach)
        parameters_betti = {}
        for betti_num, betti_classes in enumerate(kwargs["num_betti_numbers"]):
            out_dict = {}

            if betti_classes == 2:
                out_dict["task"] = "binary"

            else:
                assert betti_classes > 2, (
                    f"Betti number {betti_num} has invalid number of classes: {betti_classes}. Must be >= 2."
                )
                out_dict["task"] = "multiclass"
                out_dict["num_classes"] = betti_classes

            parameters_betti[betti_num] = out_dict

        metrics = {}
        for name_betti in self.metric_names:
            name, betti = name_betti.split("_")
            parameters = parameters_betti[int(betti)]
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
        preds = model_out["logits"].cpu().detach().clone()
        target = model_out["labels"].cpu()
        # As we know that each betti number is bound, we should ensure we work with the right range, hence we clip the predictions

        # It is necessary to round the predictions to the nearest integer as we calculate classification tasks while having optimized regression tasks.
        preds = preds.round().long()

        for metric_name in self.metric_names:
            _, betti = metric_name.split("_")

            # Clamp the predictions for betti numbers to be in the range [0, max_betti_number]
            current_preds = preds[:, int(betti)].clamp(
                min=0, max=self.num_betti_numbers[int(betti)] - 1
            )

            self.metrics[metric_name].update(
                current_preds, target[:, int(betti)]
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
