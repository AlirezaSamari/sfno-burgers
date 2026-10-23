import math

import torch


def compute_xi_grid(z_batch, x_grid, t_grid, physics):
    batch_size = z_batch.shape[0]
    device = x_grid.device
    x_scaled = x_grid.unsqueeze(1) * torch.arange(
        1, physics.forcing_modes_x + 1, device=device
    ).unsqueeze(0)
    phi = torch.sqrt(torch.tensor(2.0 / physics.length, device=device)) * torch.sin(
        torch.pi * x_scaled / physics.length
    )
    t_scaled = t_grid.unsqueeze(1) * torch.arange(
        1, physics.forcing_modes_t + 1, device=device
    ).unsqueeze(0)
    psi = torch.sqrt(torch.tensor(2.0 / physics.duration, device=device)) * torch.sin(
        torch.pi * t_scaled / physics.duration
    )
    coefficients = z_batch.view(batch_size, physics.forcing_modes_x, physics.forcing_modes_t)
    xi = torch.einsum("bmn, xm, tn -> bxt", coefficients, phi, psi)
    return xi / math.sqrt(physics.forcing_modes_x * physics.forcing_modes_t)


def generate_random_ic(batch_size, x_grid, length_scale, physics):
    count = x_grid.shape[0]
    xx, yy = torch.meshgrid(x_grid, x_grid, indexing="ij")
    covariance = torch.exp(-0.5 * ((xx - yy) / length_scale) ** 2)
    factor = torch.linalg.cholesky(
        covariance + physics.covariance_jitter * torch.eye(count, device=x_grid.device)
    )
    gaussian = torch.randn(batch_size, count, device=x_grid.device)
    initial = torch.einsum("ij, bj -> bi", factor, gaussian)
    initial = initial - initial.mean(dim=1, keepdim=True)
    maximum = initial.abs().max(dim=1, keepdim=True)[0]
    return physics.initial_amplitude * (initial / (maximum + physics.initial_normalization_epsilon))
