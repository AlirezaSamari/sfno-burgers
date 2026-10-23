import time

import numpy as np
import torch

from sfno.checkpoints import load_evaluated_model
from sfno.evaluate import evaluation_source, load_evaluation
from sfno.runtime import prepare_output, write_json, write_manifest
from sfno.solver import solve_stochastic_burgers_mc


def timed_call(function, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000 * (time.perf_counter() - start)


def benchmark_experiment(evaluation_path, output, device, checkpoint=None):
    evaluation, config = load_evaluation(evaluation_path)
    model = load_evaluated_model(evaluation, config, device, checkpoint)
    output = prepare_output(output)
    test = evaluation["test"]
    initial, forcing = test["u0"][:1].to(device), test["xi"][:1].to(device)
    coefficients = test["coefficients"][:1].to(device)
    x_grid, t_grid = test["x"].to(device), test["t"].to(device)

    def run_model():
        return model(x_grid, t_grid, initial, forcing)

    def run_solver():
        return solve_stochastic_burgers_mc(initial, coefficients, x_grid, t_grid, config.physics)

    with torch.no_grad():
        for _ in range(config.benchmark.warmup):
            run_model()
        run_solver()
        model_times = [timed_call(run_model, device) for _ in range(config.benchmark.model_repeats)]
        solver_times = [
            timed_call(run_solver, device) for _ in range(config.benchmark.solver_repeats)
        ]
    result = {
        "sample_index": 0,
        "batch_size": 1,
        "device": str(device),
        "model_ms": model_times,
        "solver_ms": solver_times,
        "model_mean_ms": float(np.mean(model_times)),
        "model_std_ms": float(np.std(model_times)),
        "solver_mean_ms": float(np.mean(solver_times)),
        "solver_std_ms": float(np.std(solver_times)),
        "measured_speedup": float(np.mean(solver_times) / np.mean(model_times)),
        "timing": "synchronized wall-clock; prepared inputs; actual model and solver",
    }
    write_json(output / "benchmark.json", result)
    write_manifest(
        output,
        "benchmark",
        config,
        device,
        [evaluation_source(evaluation_path), evaluation["checkpoints"]["sfno"]],
        [output / "benchmark.json"],
    )
    return result
