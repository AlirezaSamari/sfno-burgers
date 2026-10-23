import math

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
L, T = (1.0, 1.0)
nu = 0.02 / np.pi
sigma = 0.5
Nx_modes, Nt_modes = (16, 16)
N_modes = Nx_modes * Nt_modes
nx_grid, nt_grid = (64, 64)
dx = L / (nx_grid - 1)


def compute_xi_grid(z_batch, x_grid, t_grid):
    batch_size = z_batch.shape[0]
    x_scaled = x_grid.unsqueeze(1) * torch.arange(1, Nx_modes + 1, device=device).unsqueeze(0)
    phi = torch.sqrt(torch.tensor(2.0 / L, device=device)) * torch.sin(torch.pi * x_scaled / L)
    t_scaled = t_grid.unsqueeze(1) * torch.arange(1, Nt_modes + 1, device=device).unsqueeze(0)
    psi = torch.sqrt(torch.tensor(2.0 / T, device=device)) * torch.sin(torch.pi * t_scaled / T)
    Z_3d = z_batch.view(batch_size, Nx_modes, Nt_modes)
    xi = torch.einsum("bmn, xm, tn -> bxt", Z_3d, phi, psi)
    return xi / math.sqrt(Nx_modes * Nt_modes)


def generate_random_ic(batch_size, x_grid, length_scale=0.2):
    N = x_grid.shape[0]
    X, Y = torch.meshgrid(x_grid, x_grid, indexing="ij")
    K = torch.exp(-0.5 * ((X - Y) / length_scale) ** 2)
    L_mat = torch.linalg.cholesky(K + 1e-05 * torch.eye(N, device=device))
    z = torch.randn(batch_size, N, device=device)
    u0 = torch.einsum("ij, bj -> bi", L_mat, z)
    u0 = u0 - u0.mean(dim=1, keepdim=True)
    max_vals = u0.abs().max(dim=1, keepdim=True)[0]
    return 1.5 * (u0 / (max_vals + 1e-08))


def solve_stochastic_burgers_mc(u0_batch, z_batch, x_grid, t_grid):
    batch_size, nx, nt = (u0_batch.shape[0], x_grid.shape[0], t_grid.shape[0])
    dx = (x_grid[-1] - x_grid[0]) / (nx - 1)
    dt_save = (t_grid[-1] - t_grid[0]) / (nt - 1)
    cfl_diff = dx**2 / (2.5 * nu)
    cfl_adv = dx / 4.0
    dt_safe = min(cfl_diff, cfl_adv).item()
    n_substeps = max(int(math.ceil(dt_save.item() / dt_safe)), 200)
    dt = dt_save / n_substeps
    u_history = torch.zeros(batch_size, nx, nt, device=device)
    u_history[:, :, 0] = u0_batch
    xi = compute_xi_grid(z_batch, x_grid, t_grid)
    u = u0_batch.clone()
    for n in range(0, nt - 1):
        xi_start, xi_end = (xi[:, :, n], xi[:, :, n + 1])
        for s in range(n_substeps):
            alpha = s / n_substeps
            xi_current = (1 - alpha) * xi_start + alpha * xi_end
            u_fwd = (torch.roll(u, -1, dims=1) - u) / dx
            u_bwd = (u - torch.roll(u, 1, dims=1)) / dx
            u_x = torch.where(u > 0, u_bwd, u_fwd)
            u_xx = (torch.roll(u, -1, dims=1) - 2 * u + torch.roll(u, 1, dims=1)) / dx**2
            u = u + dt * (-u * u_x + nu * u_xx + sigma * xi_current * u)
        u_history[:, :, n + 1] = u
    return u_history


def build_dataset(num_samples, length_scale):
    print(f"Generating {num_samples} Ground Truth Trajectories (l={length_scale})...")
    x_grid = torch.linspace(0, L, nx_grid, device=device)
    t_grid = torch.linspace(0, T, nt_grid, device=device)
    u0_all, xi_all, u_true_all = ([], [], [])
    batch_size = 50
    with torch.no_grad():
        for _ in range(0, num_samples, batch_size):
            z_batch = torch.randn(batch_size, N_modes, device=device)
            u0_batch = generate_random_ic(batch_size, x_grid, length_scale=length_scale)
            xi_batch = compute_xi_grid(z_batch, x_grid, t_grid)
            u_true = solve_stochastic_burgers_mc(u0_batch, z_batch, x_grid, t_grid)
            u0_all.append(u0_batch.cpu())
            xi_all.append(xi_batch.cpu())
            u_true_all.append(u_true.cpu())
    return (torch.cat(u0_all), torch.cat(xi_all), torch.cat(u_true_all), x_grid, t_grid)


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes, self.width = (modes, width)
        scale = 1 / (width * width)
        self.weights = nn.Parameter(
            scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat)
        )
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        batchsize = x.shape[0]
        bypass = self.w(x)
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.width,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes, : self.modes] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes, : self.modes], self.weights
        )
        x_spatial = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return F.gelu(x_spatial + bypass)


class StochasticFNO(nn.Module):
    def __init__(self, modes=16, width=64, num_layers=4):
        super().__init__()
        self.lifting = nn.Linear(4, width)
        self.fourier_layers = nn.ModuleList(
            [SpectralConv2d(width, modes) for _ in range(num_layers)]
        )
        self.projection = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x_grid, t_grid, u0, xi):
        B, Nx, Nt = xi.shape
        xx, tt = torch.meshgrid(x_grid, t_grid, indexing="ij")
        grid_4d = torch.stack(
            [
                xx.unsqueeze(0).expand(B, -1, -1),
                tt.unsqueeze(0).expand(B, -1, -1),
                u0.unsqueeze(-1).expand(B, Nx, Nt),
                xi,
            ],
            dim=-1,
        )
        x = self.lifting(grid_4d).permute(0, 3, 1, 2)
        for layer in self.fourier_layers:
            x = layer(x)
        return self.projection(x.permute(0, 2, 3, 1)).squeeze(-1)


def compute_topological_spectral_loss(u_pred, u_true, lambda_spec=0.1, lambda_topo=0.05):
    loss_spatial = torch.mean((u_pred - u_true) ** 2)
    u_pred_fft = torch.fft.rfft(u_pred, dim=1, norm="forward")
    u_true_fft = torch.fft.rfft(u_true, dim=1, norm="forward")
    pred_power = torch.abs(u_pred_fft) ** 2
    true_power = torch.abs(u_true_fft) ** 2
    loss_spectral = torch.mean(
        (torch.log10(pred_power + 1e-08) - torch.log10(true_power + 1e-08)) ** 2
    )
    grad_x_pred = torch.abs(u_pred[:, 1:, :] - u_pred[:, :-1, :])
    grad_x_true = torch.abs(u_true[:, 1:, :] - u_true[:, :-1, :])
    loss_topo = torch.mean((grad_x_pred - grad_x_true) ** 2)
    return loss_spatial + lambda_spec * loss_spectral + lambda_topo * loss_topo


def compute_pinn_pde_loss(u_pred, xi, u0, x_grid, t_grid, nu, sigma):
    Nx, dt, dx = (x_grid.shape[0], t_grid[1] - t_grid[0], x_grid[1] - x_grid[0])
    u_t = (u_pred[:, :, 1:] - u_pred[:, :, :-1]) / dt
    k_x = (torch.fft.fftfreq(Nx, d=dx).to(device) * 2 * math.pi).unsqueeze(0).unsqueeze(-1)
    u_pred_ft = torch.fft.fft(u_pred[:, :, :-1], dim=1)
    u_x = torch.fft.ifft(1j * k_x * u_pred_ft, dim=1).real
    u_xx = torch.fft.ifft(-(k_x**2) * u_pred_ft, dim=1).real
    u_inner, xi_inner = (u_pred[:, :, :-1], xi[:, :, :-1])
    residual = u_t + u_inner * u_x - nu * u_xx - sigma * xi_inner * u_inner
    loss_phys = torch.mean(residual**2)
    return loss_phys


def compute_structure_function(u_tensor, max_shift):
    u_late = u_tensor[:, :, -1]
    S2, r_vals = (np.zeros(max_shift), np.zeros(max_shift))
    for shift in range(1, max_shift + 1):
        diff = u_late[:, shift:] - u_late[:, :-shift]
        S2[shift - 1] = torch.mean(diff**2).item()
        r_vals[shift - 1] = shift * dx
    return (r_vals, S2)


def compute_physical_errors(preds, true, dx, nt_grid):
    pred_fft = torch.fft.rfft(preds[:, :, -1], dim=1, norm="forward")
    true_fft = torch.fft.rfft(true[:, :, -1], dim=1, norm="forward")
    pred_power = torch.mean(torch.abs(pred_fft) ** 2, dim=0).cpu().numpy()
    true_power = torch.mean(torch.abs(true_fft) ** 2, dim=0).cpu().numpy()
    err_spec = np.linalg.norm(
        np.log10(pred_power[1:] + 1e-08) - np.log10(true_power[1:] + 1e-08)
    ) / np.linalg.norm(np.log10(true_power[1:] + 1e-08))

    def get_grad_pdf(tensor):
        grads = (tensor[:, 2:, int(0.5 * nt_grid) :] - tensor[:, :-2, int(0.5 * nt_grid) :]) / (
            2 * dx
        )
        hist, _ = np.histogram(
            grads.flatten().cpu().numpy(), bins=100, range=(-30, 15), density=True
        )
        return hist

    true_pdf = get_grad_pdf(true)
    pred_pdf = get_grad_pdf(preds)
    err_grad = np.linalg.norm(pred_pdf - true_pdf) / np.linalg.norm(true_pdf)
    grad_pred = (preds[:, 2:, int(0.5 * nt_grid) :] - preds[:, :-2, int(0.5 * nt_grid) :]) / (
        2 * dx
    )
    grad_true = (true[:, 2:, int(0.5 * nt_grid) :] - true[:, :-2, int(0.5 * nt_grid) :]) / (2 * dx)
    ens_pred_mean = (0.5 * torch.sum(grad_pred**2, dim=1) * dx).mean(dim=0).cpu().numpy()
    ens_true_mean = (0.5 * torch.sum(grad_true**2, dim=1) * dx).mean(dim=0).cpu().numpy()
    err_ens = np.linalg.norm(ens_pred_mean - ens_true_mean) / np.linalg.norm(ens_true_mean)
    return (err_spec * 100, err_grad * 100, err_ens * 100)


def compute_uq_metrics(preds, true):
    pred_mean = preds.mean(dim=0).cpu().numpy()
    pred_std = preds.std(dim=0).cpu().numpy()
    true_np = true.cpu().numpy()

    def get_coverage(z_score):
        lower = pred_mean - z_score * pred_std
        upper = pred_mean + z_score * pred_std
        in_bound = (true_np >= lower) & (true_np <= upper)
        return np.mean(in_bound) * 100

    cov_90 = get_coverage(1.645)
    cov_95 = get_coverage(1.96)
    cov_99 = get_coverage(2.576)
    x_norm = (true_np - pred_mean) / (pred_std + 1e-08)
    crps = pred_std * (
        x_norm * (2 * stats.norm.cdf(x_norm) - 1) + 2 * stats.norm.pdf(x_norm) - 1 / np.sqrt(np.pi)
    )
    mean_crps = np.mean(crps)
    return (cov_90, cov_95, cov_99, mean_crps)
