import math

import torch

from sfno.forcing import compute_xi_grid


def integration_parameters(x_grid, t_grid, physics):
    dx = (x_grid[-1] - x_grid[0]) / (x_grid.shape[0] - 1)
    dt_save = (t_grid[-1] - t_grid[0]) / (t_grid.shape[0] - 1)
    cfl_diff = dx**2 / (physics.diffusion_cfl_divisor * physics.viscosity)
    cfl_adv = dx / physics.advection_cfl_speed
    dt_safe = min(cfl_diff, cfl_adv).item()
    substeps = max(int(math.ceil(dt_save.item() / dt_safe)), physics.minimum_substeps)
    return dx, dt_save / substeps, substeps


def solve_stochastic_burgers_mc(u0_batch, z_batch, x_grid, t_grid, physics):
    batch_size, nx, nt = u0_batch.shape[0], x_grid.shape[0], t_grid.shape[0]
    dx, dt, substeps = integration_parameters(x_grid, t_grid, physics)
    history = torch.zeros(batch_size, nx, nt, device=x_grid.device)
    history[:, :, 0] = u0_batch
    xi = compute_xi_grid(z_batch, x_grid, t_grid, physics)
    u = u0_batch.clone()
    for n in range(nt - 1):
        xi_start, xi_end = xi[:, :, n], xi[:, :, n + 1]
        for step in range(substeps):
            alpha = step / substeps
            xi_current = (1 - alpha) * xi_start + alpha * xi_end
            forward = (torch.roll(u, -1, dims=1) - u) / dx
            backward = (u - torch.roll(u, 1, dims=1)) / dx
            u_x = torch.where(u > 0, backward, forward)
            u_xx = (torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)) / dx**2
            u = u + dt * (
                -u * u_x + physics.viscosity * u_xx + physics.noise_scale * xi_current * u
            )
        history[:, :, n + 1] = u
    return history
