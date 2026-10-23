import csv
import logging
from pathlib import Path

import torch

from sfno.checkpoints import load_models
from sfno.config import Config
from sfno.data import dataset_source, load_dataset
from sfno.experiments import ABLATIONS
from sfno.metrics import gaussian_uq_metrics, moment_errors, physical_errors, summarize_fields
from sfno.runtime import (
    capture_rng,
    load_torch,
    prepare_output,
    restore_rng,
    save_torch,
    sha256,
    to_cpu,
    write_json,
    write_manifest,
)

LOGGER = logging.getLogger(__name__)


def predict(model, initial, forcing, x_grid, t_grid, batch_size, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, initial.shape[0], batch_size):
            outputs.append(
                model(
                    x_grid,
                    t_grid,
                    initial[start : start + batch_size].to(device),
                    forcing[start : start + batch_size].to(device),
                )
            )
    predicted = torch.cat(outputs, dim=0)
    if not torch.isfinite(predicted).all().item():
        raise FloatingPointError("Nonfinite model predictions.")
    return predicted


def write_tables(output, metrics):
    groups = {
        "table1.csv": [
            "mean_error_percent",
            "variance_error_percent",
            "log_spectrum_error_percent",
            "gradient_pdf_error_percent",
            "gradient_energy_error_percent",
        ],
        "table2.csv": [
            "coverage_90_percent",
            "coverage_95_percent",
            "coverage_99_percent",
            "gaussian_crps",
        ],
    }
    for filename, columns in groups.items():
        with (Path(output) / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "name", *columns])
            writer.writeheader()
            for ablation in ABLATIONS:
                writer.writerow(
                    {
                        "model": ablation.key,
                        "name": ablation.name,
                        **{column: metrics[ablation.key][column] for column in columns},
                    }
                )


def evaluate_models(data_path, checkpoints, output, device, mapping=None):
    payload, config = load_dataset(data_path)
    output = prepare_output(output)
    restore_rng(payload["rng_after_data"], device, payload["device"])
    models, sources = load_models(config, device, checkpoints, mapping)
    rng_after_models = capture_rng()
    test = payload["test"]
    x_grid, t_grid = test["x"].to(device), test["t"].to(device)
    dx = config.physics.length / (config.data.nx - 1)
    predictions, metrics = {}, {}
    summaries = {"reference": summarize_fields(test["target"], config)}
    for ablation in ABLATIONS:
        key = ablation.key
        predicted = predict(
            models[key],
            test["u0"],
            test["xi"],
            x_grid,
            t_grid,
            config.diagnostics.inference_batch_size,
            device,
        )
        metrics[key] = {
            **moment_errors(predicted, test["target"]),
            **physical_errors(predicted, test["target"], dx, config.data.nt, config.diagnostics),
            **gaussian_uq_metrics(predicted, test["target"], config.diagnostics),
        }
        summaries[key] = summarize_fields(predicted, config)
        predictions[key] = predicted.cpu()
        LOGGER.info(
            "%s mean %.4f%% variance %.4f%% spectrum %.4f%%",
            key,
            metrics[key]["mean_error_percent"],
            metrics[key]["variance_error_percent"],
            metrics[key]["log_spectrum_error_percent"],
        )
    bundle = {
        "kind": "sfno_evaluation",
        "schema_version": 1,
        "config": config.to_dict(),
        "config_sha256": config.fingerprint(),
        "device": str(device),
        "dataset": dataset_source(data_path),
        "checkpoints": sources,
        "test": test,
        "predictions": predictions,
        "metrics": metrics,
        "summaries": summaries,
        "rng_after_models": rng_after_models,
    }
    save_torch(output / "evaluation.pt", to_cpu(bundle))
    write_json(output / "metrics.json", metrics)
    write_json(output / "config.json", config.to_dict())
    write_tables(output, metrics)
    artifacts = [
        output / name
        for name in ("evaluation.pt", "metrics.json", "config.json", "table1.csv", "table2.csv")
    ]
    write_manifest(
        output, "evaluate", config, device, [dataset_source(data_path), sources], artifacts
    )
    return output


def load_evaluation(path):
    path = Path(path)
    path = path / "evaluation.pt" if path.is_dir() else path
    bundle = load_torch(path)
    if bundle.get("kind") != "sfno_evaluation":
        raise ValueError("Expected an sfno evaluation bundle.")
    config = Config.from_dict(bundle["config"])
    if config.fingerprint() != bundle["config_sha256"]:
        raise ValueError("The evaluation configuration fingerprint is invalid.")
    return bundle, config


def evaluation_source(path):
    path = Path(path)
    path = path / "evaluation.pt" if path.is_dir() else path
    return {"path": str(path.resolve()), "sha256": sha256(path)}
