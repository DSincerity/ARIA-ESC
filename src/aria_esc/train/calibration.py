"""Temperature scaling fit on the validation split.
"""
import math


def fit_temperature(val_logits, val_labels) -> float:
    import torch
    from torch.nn import functional as F

    logits = torch.as_tensor(val_logits, dtype=torch.float64).detach().cpu()
    labels = torch.as_tensor(val_labels).detach().cpu()
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) == 0:
        raise ValueError("Expected nonempty [N, 2] validation logits")
    if labels.shape != (len(logits),) or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected N binary validation labels")
    if not torch.isfinite(logits).all():
        raise ValueError("Validation logits must be finite")
    labels = labels.long()
    # Optimize log-temperature so the temperature stays positive; objective is validation NLL.
    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=logits.dtype))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(1e-3, 1e3).item())
    if not math.isfinite(temperature) or F.cross_entropy(logits / temperature, labels) > F.cross_entropy(logits, labels):
        return 1.0
    return temperature


def apply_temperature(logits, temperature: float):
    import torch

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values = torch.as_tensor(logits, dtype=torch.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not torch.isfinite(values).all():
        raise ValueError("Expected finite [N, 2] logits")
    return torch.softmax(values / temperature, dim=-1)[:, 1]


def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
    """Binary ECE: positive rate vs. mean P(positive) over equal-width bins on [0, 1]."""
    import torch

    probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    bin_index = torch.clamp((probabilities * bins).long(), max=bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = bin_index == b
        if mask.any():
            ece += mask.double().mean().item() * abs(labels[mask].mean().item() - probabilities[mask].mean().item())
    return ece


def calibration_metrics(logits, labels, temperature: float) -> dict:
    import torch
    from torch.nn import functional as F

    values = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    probabilities = apply_temperature(values, temperature)
    return {"nll": F.cross_entropy(values / temperature, labels).item(),
            "brier": ((probabilities - labels) ** 2).mean().item(),
            "ece": expected_calibration_error(probabilities, labels),
            "accuracy": (values.argmax(-1) == labels).double().mean().item()}
