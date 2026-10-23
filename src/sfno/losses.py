import math

import torch


def loss_components(u_pred, u_true, spectral_floor=1e-8):
    spatial = torch.mean((u_pred - u_true) ** 2)
    predicted_fft = torch.fft.rfft(u_pred, dim=1, norm="forward")
    true_fft = torch.fft.rfft(u_true, dim=1, norm="forward")
    predicted_power, true_power = torch.abs(predicted_fft) ** 2, torch.abs(true_fft) ** 2
    spectral = torch.mean(
        (torch.log10(predicted_power + spectral_floor) - torch.log10(true_power + spectral_floor))
        ** 2
    )
    predicted_increments = torch.abs(u_pred[:, 1:, :] - u_pred[:, :-1, :])
    true_increments = torch.abs(u_true[:, 1:, :] - u_true[:, :-1, :])
    increment = torch.mean((predicted_increments - true_increments) ** 2)
    return spatial, spectral, increment


def composite_loss(u_pred, u_true, spectral_weight=0.1, increment_weight=0.05, spectral_floor=1e-8):
    spatial, spectral, increment = loss_components(u_pred, u_true, spectral_floor)
    return spatial + spectral_weight * spectral + increment_weight * increment


def pde_residual_loss(u_pred, xi, x_grid, t_grid, viscosity, noise_scale):
    nx, dt, dx = x_grid.shape[0], t_grid[1] - t_grid[0], x_grid[1] - x_grid[0]
    u_t = (u_pred[:, :, 1:] - u_pred[:, :, :-1]) / dt
    wavenumber = (
        (torch.fft.fftfreq(nx, d=dx.item()).to(u_pred.device) * 2 * math.pi)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    transformed = torch.fft.fft(u_pred[:, :, :-1], dim=1)
    u_x = torch.fft.ifft(1j * wavenumber * transformed, dim=1).real
    u_xx = torch.fft.ifft(-(wavenumber**2) * transformed, dim=1).real
    inner, forcing = u_pred[:, :, :-1], xi[:, :, :-1]
    residual = u_t + inner * u_x - viscosity * u_xx - noise_scale * forcing * inner
    return torch.mean(residual**2)
