from __future__ import annotations
from typing import Any
def reconstructionloss(prediction: Any, target: Any, *, kind: str = "mse") -> Any:
    import torch.nn.functional as F
    if kind == "mse":
        return F.mse_loss(prediction, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    raise ValueError(f"unknown reconstruction loss: {kind}")
def distillationkl(student_logits: Any, teacher_logits: Any, *, temperature: float = 1.0) -> Any:
    import torch.nn.functional as F
    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature * temperature
def patternseparationloss(codes: Any, *, margin: float = 0.1) -> Any:
    import torch
    import torch.nn.functional as F
    normalized = F.normalize(codes, dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    eye = torch.eye(similarities.shape[-1], device=similarities.device, dtype=torch.bool)
    off_diagonal = similarities.masked_select(~eye)
    return F.relu(off_diagonal - margin).mean() if off_diagonal.numel() else similarities.sum() * 0
def sourcecontrastiveloss(query_codes: Any, positive_codes: Any, *, temperature: float = 0.07) -> Any:
    import torch
    import torch.nn.functional as F
    queries = F.normalize(query_codes, dim=-1)
    positives = F.normalize(positive_codes, dim=-1)
    logits = queries @ positives.transpose(-1, -2) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
