"""Metrics for model evaluation."""

import torch


def modernity_score(prob_per_label: torch.Tensor):
    """Modernity score.

    Computes the modernity score as the weighted sum of the predicted probabilities per class and the class label.

    :param prob_per_label: Predicted probabilities per label (num_images x classes).
    """
    # modernity score = weighted sum of model_outputs times year_label
    # year labels start with 0, so make sure to +1 each label
    if prob_per_label.ndim != 2:
        raise ValueError(
            "Input must be a 2-dim tensor, but has {prob_per_label.ndim} dimensions."
        )

    num_year_classes = prob_per_label.shape[-1]
    year_labels = torch.arange(num_year_classes) + 1

    return torch.sum(year_labels * prob_per_label, dim=1)
