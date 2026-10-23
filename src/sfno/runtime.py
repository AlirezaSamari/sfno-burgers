import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from sfno import __version__


def resolve_device(value="auto"):
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Choose CPU or CUDA for this implementation.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "python": random.getstate(),
    }


def restore_rng(state, device, source_device):
    if torch.device(source_device).type != device.type:
        raise ValueError(
            f"The cached random stream was generated on {source_device}; requested {device}. "
            "Regenerate the dataset on the requested device to preserve the saved random sequence."
        )
    torch.set_rng_state(state["torch_cpu"].cpu())
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    random.setstate(state["python"])
    if device.type == "cuda":
        saved_index = torch.device(source_device).index or 0
        if saved_index >= len(state["torch_cuda"]):
            raise ValueError("The cache has no CUDA random state for its recorded device.")
        torch.cuda.set_rng_state(state["torch_cuda"][saved_index].cpu(), device=device)


def environment(device):
    import matplotlib
    import scipy

    return {
        "package": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_torch(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def prepare_output(path, resume=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {path}. Use a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(output, stage, config, device, inputs, artifacts, extra=None):
    output = Path(output)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "config_sha256": config.fingerprint(),
        "environment": environment(device),
        "inputs": inputs,
        "artifacts": [
            {"path": str(Path(path).relative_to(output)), "sha256": sha256(path)}
            for path in artifacts
        ],
    }
    if extra:
        manifest.update(extra)
    write_json(output / "manifest.json", manifest)
    return manifest
