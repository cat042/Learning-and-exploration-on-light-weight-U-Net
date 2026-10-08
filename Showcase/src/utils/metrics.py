import torch
import torch.nn.functional as F


DEFAULT_FOCAL_ALPHA = 0.25
DEFAULT_FOCAL_GAMMA = 2.0


def dice_scores(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    denominator = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (2.0 * intersection + eps) / (denominator + eps)


def iou_scores(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return (intersection + eps) / (union + eps)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating_factor = (1.0 - p_t).pow(gamma)
    return (alpha_factor * modulating_factor * bce).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    focal_bce = focal_bce_loss(logits, targets)
    return focal_bce + dice_loss(logits, targets)


def batch_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (probabilities >= threshold).float()
    return dice_scores(predictions, targets), iou_scores(predictions, targets)


def batch_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(logits)
    return batch_metrics(probabilities, targets, threshold=threshold)