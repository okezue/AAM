from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
from aamemory.training.codec import buildmlpactivationcodec
from aamemory.training.objectives import patternseparationloss, reconstructionloss
@dataclass
class CodecTrainingConfig:
    input_npz: str
    input_key: str = "activations"
    outputdir: str = "runs/codec"
    bottleneck_dimension: int = 256
    hidden_dimension: int | None = None
    batch_size: int = 64
    epochs: int = 10
    learningrate: float = 1e-3
    weight_decay: float = 1e-4
    separation_weight: float = 0.01
    validation_fraction: float = 0.1
    seed: int = 0
    device: str = "cuda"
def trainactivationcodec(config: CodecTrainingConfig) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except ImportError as exc:
        raise ImportError("Codec training requires `pip install -e .[hf]`") from exc
    torch.manual_seed(config.seed)
    arrays = np.load(config.input_npz)
    activations = np.asarray(arrays[config.input_key], dtype=np.float32)
    if activations.ndim != 2:
        raise ValueError("activation dataset must be a 2-D [examples, features] array")
    tensor = torch.from_numpy(activations)
    dataset = TensorDataset(tensor)
    validation_size = int(len(dataset) * config.validation_fraction)
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size)
    model = buildmlpactivationcodec(
        input_dimension=activations.shape[1],
        bottleneck_dimension=config.bottleneck_dimension,
        hidden_dimension=config.hidden_dimension,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learningrate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    outputdir = Path(config.outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(config.device)
            reconstruction, code = model(batch)
            loss = reconstructionloss(reconstruction, batch) + config.separation_weight * patternseparationloss(code)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses: list[float] = []
        with torch.inference_mode():
            for (batch,) in validation_loader:
                batch = batch.to(config.device)
                reconstruction, code = model(batch)
                loss = reconstructionloss(reconstruction, batch) + config.separation_weight * patternseparationloss(code)
                validation_losses.append(float(loss.cpu()))
        record = {
            "epoch": float(epoch + 1),
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": float(np.mean(validation_losses)) if validation_losses else float("nan"),
        }
        history.append(record)
        torch.save(
            {"model": model.statedict(), "shape": asdict(model.shape), "config": asdict(config)},
            outputdir / "checkpoint_last.pt",
        )
    report = {"config": asdict(config), "history": history, "examples": len(dataset)}
    (outputdir / "training_report.json").write_text(json.dumps(report, indent=2))
    return report
