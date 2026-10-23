import logging

import torch

from sfno.checkpoints import load_evaluated_model
from sfno.evaluate import evaluation_source, load_evaluation
from sfno.forcing import compute_xi_grid
from sfno.runtime import prepare_output, save_torch, to_cpu, write_json, write_manifest

LOGGER = logging.getLogger(__name__)


def recover_coefficients(model, initial, observed, x_grid, t_grid, config):
    inverse, physics = config.inverse, config.physics
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(inverse.seed)
    coefficients = torch.randn(
        1,
        physics.forcing_modes_x * physics.forcing_modes_t,
        device=x_grid.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [coefficients],
        lr=inverse.learning_rate,
        betas=(inverse.adam_beta1, inverse.adam_beta2),
        eps=inverse.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=inverse.scheduler_step,
        gamma=inverse.scheduler_gamma,
    )
    history = []
    for iteration in range(inverse.iterations):
        optimizer.zero_grad()
        forcing = compute_xi_grid(coefficients, x_grid, t_grid, physics)
        predicted = model(x_grid, t_grid, initial, forcing)
        data_loss = torch.mean((predicted - observed) ** 2)
        penalty = inverse.regularization * torch.mean(coefficients**2)
        loss = data_loss + penalty
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite inverse loss at iteration {iteration}.")
        if iteration + 1 == inverse.iterations:
            last_coefficients = coefficients.detach().clone()
        history.append(
            {
                "iteration": iteration,
                "data_loss": data_loss.item(),
                "regularization_loss": penalty.item(),
                "objective": loss.item(),
            }
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % 200 == 0 or iteration + 1 == inverse.iterations:
            LOGGER.info("Inverse iteration %04d data loss %.6e", iteration, data_loss.item())
    result = {
        "last_evaluated": {
            "coefficients": last_coefficients,
            "xi": forcing.detach(),
            "predicted": predicted.detach(),
            "data_loss": data_loss.item(),
            "objective": loss.item(),
            "iteration": inverse.iterations - 1,
        },
        "history": history,
    }
    with torch.no_grad():
        final_forcing = compute_xi_grid(coefficients, x_grid, t_grid, physics)
        final_predicted = model(x_grid, t_grid, initial, final_forcing)
        final_data = torch.mean((final_predicted - observed) ** 2)
        final_objective = final_data + inverse.regularization * torch.mean(coefficients**2)
    result["after_final_update"] = {
        "coefficients": coefficients.detach(),
        "xi": final_forcing,
        "predicted": final_predicted,
        "data_loss": final_data.item(),
        "objective": final_objective.item(),
        "completed_updates": inverse.iterations,
    }
    return result


def inverse_experiment(evaluation_path, output, device, checkpoint=None):
    evaluation, config = load_evaluation(evaluation_path)
    model = load_evaluated_model(evaluation, config, device, checkpoint)
    output = prepare_output(output)
    test = evaluation["test"]
    index = config.inverse.observation_index
    initial = test["u0"][index : index + 1].to(device)
    observed = test["target"][index : index + 1].to(device)
    true_forcing = test["xi"][index : index + 1].to(device)
    x_grid, t_grid = test["x"].to(device), test["t"].to(device)
    recovered = recover_coefficients(model, initial, observed, x_grid, t_grid, config)
    with torch.no_grad():
        fluid_from_true = model(x_grid, t_grid, initial, true_forcing)
    bundle = {
        "kind": "sfno_inverse",
        "schema_version": 1,
        "config": config.to_dict(),
        "evaluation": evaluation_source(evaluation_path),
        "checkpoint": evaluation["checkpoints"]["sfno"],
        "observation_index": index,
        "x": x_grid,
        "t": t_grid,
        "u0": initial,
        "observed": observed,
        "true_forcing": true_forcing,
        "true_coefficients": test["coefficients"][index : index + 1],
        "fluid_from_true": fluid_from_true,
        **recovered,
    }
    summary = {
        key: {
            field: value
            for field, value in recovered[key].items()
            if not isinstance(value, torch.Tensor)
        }
        for key in ("last_evaluated", "after_final_update")
    }
    save_torch(output / "inverse.pt", to_cpu(bundle))
    write_json(output / "history.json", recovered["history"])
    write_json(output / "fit_summary.json", summary)
    write_json(output / "config.json", config.to_dict())
    artifacts = [
        output / name for name in ("inverse.pt", "history.json", "fit_summary.json", "config.json")
    ]
    write_manifest(
        output,
        "inverse",
        config,
        device,
        [evaluation_source(evaluation_path), evaluation["checkpoints"]["sfno"]],
        artifacts,
        {"plot_iterate": "last_evaluated"},
    )
    return output
