import torch
from torch.nn import functional as F

from sfno.checkpoints import load_evaluated_model
from sfno.evaluate import evaluation_source, load_evaluation
from sfno.forcing import compute_xi_grid, generate_random_ic
from sfno.runtime import prepare_output, restore_rng, save_torch, to_cpu, write_json, write_manifest
from sfno.solver import solve_stochastic_burgers_mc


def resolution_experiment(evaluation_path, output, device, checkpoint=None, include_unpaired=False):
    evaluation, config = load_evaluation(evaluation_path)
    model = load_evaluated_model(evaluation, config, device, checkpoint)
    output = prepare_output(output)
    restore_rng(evaluation["rng_after_models"], device, evaluation["device"])
    physics, resolution = config.physics, config.resolution
    x_grid = torch.linspace(0, physics.length, resolution.nx, device=device)
    t_grid = torch.linspace(0, physics.duration, resolution.nt, device=device)
    coefficients = torch.randn(1, physics.forcing_modes_x * physics.forcing_modes_t, device=device)
    initial = generate_random_ic(1, x_grid, resolution.length_scale, physics)
    forcing = compute_xi_grid(coefficients, x_grid, t_grid, physics)
    with torch.no_grad():
        reference = solve_stochastic_burgers_mc(initial, coefficients, x_grid, t_grid, physics)
        predicted = model(x_grid, t_grid, initial, forcing)
    if not torch.isfinite(reference).all() or not torch.isfinite(predicted).all():
        raise FloatingPointError("Nonfinite finer-grid fields.")
    bundle = {
        "kind": "sfno_resolution",
        "schema_version": 1,
        "config": config.to_dict(),
        "evaluation": evaluation_source(evaluation_path),
        "checkpoint": evaluation["checkpoints"]["sfno"],
        "x": x_grid,
        "t": t_grid,
        "coefficients": coefficients,
        "u0": initial,
        "xi": forcing,
        "reference": reference,
        "predicted": predicted,
    }
    if include_unpaired:
        coarse = evaluation["test"]["target"][0:1].unsqueeze(1)
        bundle["legacy_unpaired_diagnostic"] = {
            "paired": False,
            "coarse_test_index": 0,
            "interpolated": F.interpolate(
                coarse,
                size=(resolution.nx, resolution.nt),
                mode="bilinear",
                align_corners=True,
            ).squeeze(1),
        }
    save_torch(output / "resolution.pt", to_cpu(bundle))
    write_json(output / "config.json", config.to_dict())
    write_manifest(
        output,
        "resolution",
        config,
        device,
        [evaluation_source(evaluation_path), evaluation["checkpoints"]["sfno"]],
        [output / "resolution.pt", output / "config.json"],
        {"sample_count": 1, "legacy_unpaired_diagnostic": include_unpaired},
    )
    return output
