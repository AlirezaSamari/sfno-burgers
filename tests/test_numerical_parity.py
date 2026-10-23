from dataclasses import replace

import numpy as np
import pytest
import reference_implementation as reference
import torch

from sfno.config import Config
from sfno.data import build_dataset
from sfno.forcing import compute_xi_grid, generate_random_ic
from sfno.losses import composite_loss, loss_components, pde_residual_loss
from sfno.metrics import (
    gaussian_uq_metrics,
    moment_errors,
    physical_errors,
    spatial_gradients,
    structure_function,
)
from sfno.model import StochasticFNO, count_parameters
from sfno.runtime import seed_everything
from sfno.solver import integration_parameters, solve_stochastic_burgers_mc


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_paper_settings(paper):
    assert paper.to_dict() == Config().to_dict()
    assert (paper.data.train_samples, paper.data.test_samples) == (800, 500)
    assert (paper.training.epochs, paper.training.batch_size) == (1301, 16)
    assert (paper.inverse.iterations, paper.inverse.observation_index) == (2000, 10)
    assert paper.physics.viscosity == reference.nu
    assert paper.physics.minimum_substeps == 200
    assert paper.training.spectral_weight == 0.1
    assert paper.training.increment_weight == 0.05
    assert paper.training.residual_weight == 0.1


def test_forcing_and_initial_conditions(paper):
    x, t = torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)
    seed_everything(42)
    z = torch.randn(3, 256)
    actual = compute_xi_grid(z, x, t, paper.physics)
    exact(actual, reference.compute_xi_grid(z, x, t))
    seed_everything(78)
    initial = generate_random_ic(3, x, 0.1, paper.physics)
    seed_everything(78)
    exact(initial, reference.generate_random_ic(3, x, 0.1))
    torch.testing.assert_close(initial.abs().max(dim=1).values, torch.full((3,), 1.5))
    assert initial.mean(dim=1).abs().max() < 1e-6


def test_single_forcing_mode_has_declared_normalization(paper):
    x, t = torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)
    z = torch.zeros(1, 256)
    z[0, 0] = 1
    expected = 2 / 16 * torch.sin(torch.pi * x[:, None]) * torch.sin(torch.pi * t[None, :])
    torch.testing.assert_close(compute_xi_grid(z, x, t, paper.physics)[0], expected)


def test_solver_matches_reference_at_paper_resolution(paper):
    seed_everything(11)
    x, t = torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)
    z = torch.randn(2, 256)
    initial = generate_random_ic(2, x, 0.1, paper.physics)
    with torch.no_grad():
        actual = solve_stochastic_burgers_mc(initial, z, x, t, paper.physics)
        expected = reference.solve_stochastic_burgers_mc(initial, z, x, t)
    exact(actual, expected)
    exact(actual[:, :, 0], initial)
    assert integration_parameters(x, t, paper.physics)[2] == 200


def test_zero_solution_is_preserved(paper):
    x, t = torch.linspace(0, 1, 8), torch.linspace(0, 1, 4)
    with torch.no_grad():
        value = solve_stochastic_burgers_mc(
            torch.zeros(2, 8), torch.randn(2, 256), x, t, paper.physics
        )
    exact(value, torch.zeros_like(value))


def test_generation_draw_order_and_partial_batch(paper, monkeypatch):
    config = replace(paper, data=replace(paper.data, nx=8, nt=4, generation_batch_size=50))
    monkeypatch.setattr(reference, "nx_grid", 8)
    monkeypatch.setattr(reference, "nt_grid", 4)
    seed_everything(42)
    expected = reference.build_dataset(50, 0.2)
    expected_next = torch.randn(5)
    seed_everything(42)
    actual = build_dataset(50, 0.2, config, torch.device("cpu"))
    for key, value in zip(("u0", "xi", "target", "x", "t"), expected):
        exact(actual[key], value)
    exact(torch.randn(5), expected_next)
    partial = build_dataset(3, 0.2, config, torch.device("cpu"))
    assert partial["target"].shape[0] == 3


def test_architecture_initialization_forward_and_parameter_counts():
    torch.manual_seed(27)
    expected = reference.StochasticFNO()
    torch.manual_seed(27)
    actual = StochasticFNO()
    assert tuple(actual.state_dict()) == tuple(expected.state_dict())
    for key, tensor in actual.state_dict().items():
        exact(tensor, expected.state_dict()[key])
    counts = count_parameters(actual)
    assert counts["parameter_entries"] == 4_219_713
    assert counts["real_scalar_components"] == 8_414_017
    x, t = torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)
    u0, xi = torch.randn(1, 64), torch.randn(1, 64, 64)
    with torch.no_grad():
        exact(actual(x, t, u0, xi), expected(x, t, u0, xi))


@pytest.mark.parametrize("weights", [(0.0, 0.0), (0.1, 0.0), (0.0, 0.05), (0.1, 0.05)])
def test_composite_loss_value_and_gradient(weights):
    torch.manual_seed(13)
    actual_input = torch.randn(2, 64, 64, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    target = torch.randn_like(actual_input)
    actual = composite_loss(actual_input, target, *weights)
    expected = reference.compute_topological_spectral_loss(expected_input, target, *weights)
    exact(actual, expected)
    actual.backward()
    expected.backward()
    exact(actual_input.grad, expected_input.grad)


def test_increment_penalty_uses_magnitudes_without_grid_scaling():
    prediction = torch.tensor([[[1.0], [3.0], [0.0]]])
    target = -prediction
    _, _, increment = loss_components(prediction, target)
    assert increment.item() == 0
    shifted = torch.tensor([[[1.0], [4.0], [0.0]]])
    assert loss_components(prediction, shifted)[2].item() == 1


def test_pde_loss_and_gradient_match_reference(paper):
    torch.manual_seed(2)
    predicted = torch.randn(2, 64, 64, requires_grad=True)
    original = predicted.detach().clone().requires_grad_(True)
    xi, initial = torch.randn(2, 64, 64), torch.randn(2, 64)
    x, t = torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)
    actual = pde_residual_loss(
        predicted, xi, x, t, paper.physics.viscosity, paper.physics.noise_scale
    )
    expected = reference.compute_pinn_pde_loss(
        original, xi, initial, x, t, reference.nu, reference.sigma
    )
    exact(actual, expected)
    actual.backward()
    expected.backward()
    exact(predicted.grad, original.grad)


def test_physical_uq_and_structure_metrics(paper):
    torch.manual_seed(9)
    true = 0.1 * torch.randn(12, 64, 64)
    predicted = true + 0.01 * torch.randn_like(true)
    dx = 1 / 63
    actual = physical_errors(predicted, true, dx, 64, paper.diagnostics)
    expected = reference.compute_physical_errors(predicted, true, dx, 64)
    np.testing.assert_array_equal(list(actual.values()), expected)
    uq = gaussian_uq_metrics(predicted, true, paper.diagnostics)
    np.testing.assert_array_equal(list(uq.values()), reference.compute_uq_metrics(predicted, true))
    actual_r, actual_s2 = structure_function(true, 21, dx)
    expected_r, expected_s2 = reference.compute_structure_function(true, 21)
    np.testing.assert_array_equal(actual_r, expected_r)
    np.testing.assert_array_equal(actual_s2, expected_s2)
    assert moment_errors(true, true) == {"mean_error_percent": 0.0, "variance_error_percent": 0.0}


def test_centered_gradient_on_linear_field():
    x = torch.linspace(0, 1, 9)
    values = (3 * x).reshape(1, 9, 1).expand(2, -1, 4)
    exact(spatial_gradients(values, 1 / 8), torch.full((2, 7, 4), 3.0))
