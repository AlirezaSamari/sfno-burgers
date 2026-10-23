import json
from pathlib import Path

import torch

from sfno.experiments import ABLATIONS
from sfno.model import make_model
from sfno.runtime import load_torch, sha256


def checkpoint_paths(directory, mapping=None):
    directory = Path(directory)
    overrides = json.loads(Path(mapping).read_text(encoding="utf-8")) if mapping else {}
    unknown = set(overrides) - {ablation.key for ablation in ABLATIONS}
    if unknown:
        raise ValueError(f"Unknown checkpoint mapping keys: {sorted(unknown)}")
    resolved = {}
    for ablation in ABLATIONS:
        if ablation.key in overrides:
            candidate = Path(overrides[ablation.key])
            candidates = [candidate if candidate.is_absolute() else directory / candidate]
        else:
            candidates = [directory / name for name in (ablation.filename, *ablation.aliases)]
            candidates = [path for path in candidates if path.is_file()]
        if not candidates or not candidates[0].is_file():
            raise FileNotFoundError(
                f"Missing {ablation.key} checkpoint in {directory}; expected {ablation.filename}."
            )
        if len({sha256(path) for path in candidates}) > 1:
            raise ValueError(
                f"Conflicting checkpoint aliases for {ablation.key}. Select one explicitly with --mapping."
            )
        resolved[ablation.key] = candidates[0].resolve()
    return resolved


def load_state(path, model):
    state = load_torch(path)
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError(f"Expected a plain model state_dict: {path}")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError(f"Nonfinite checkpoint parameters: {path}")
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_models(config, device, directory, mapping=None):
    paths = checkpoint_paths(directory, mapping)
    models = {}
    sources = {}
    for ablation in ABLATIONS:
        model = make_model(config, device)
        models[ablation.key] = load_state(paths[ablation.key], model)
        sources[ablation.key] = {
            "path": str(paths[ablation.key]),
            "sha256": sha256(paths[ablation.key]),
        }
    return models, sources


def load_evaluated_model(evaluation, config, device, checkpoint=None):
    source = evaluation["checkpoints"]["sfno"]
    path = Path(checkpoint) if checkpoint else Path(source["path"])
    if not path.is_file():
        raise FileNotFoundError(
            "S-FNO checkpoint not found. Supply its current path with --checkpoint."
        )
    if sha256(path) != source["sha256"]:
        raise ValueError("The S-FNO checkpoint differs from the one used in this evaluation.")
    return load_state(path, make_model(config, device))
