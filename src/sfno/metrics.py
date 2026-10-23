import numpy as np
import torch
from scipy import stats


def relative_norm_percent(predicted, reference):
    denominator = np.linalg.norm(reference)
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("A relative diagnostic has an invalid zero or nonfinite reference norm.")
    value = np.linalg.norm(predicted - reference) / denominator * 100
    if not np.isfinite(value):
        raise ValueError("A relative diagnostic is nonfinite.")
    return float(value)


def moment_errors(predicted, reference):
    true_mean = reference.mean(dim=0).cpu().numpy()
    true_variance = reference.var(dim=0).cpu().numpy()
    pred_mean = predicted.mean(dim=0).cpu().numpy()
    pred_variance = predicted.var(dim=0).cpu().numpy()
    return {
        "mean_error_percent": relative_norm_percent(pred_mean, true_mean),
        "variance_error_percent": relative_norm_percent(pred_variance, true_variance),
    }


def terminal_spectrum(tensor):
    transformed = torch.fft.rfft(tensor[:, :, -1], dim=1, norm="forward")
    return torch.mean(torch.abs(transformed) ** 2, dim=0).cpu().numpy()


def spatial_gradients(tensor, dx, start=0):
    return (tensor[:, 2:, start:] - tensor[:, :-2, start:]) / (2 * dx)


def gradient_energy(gradient, dx):
    return 0.5 * torch.sum(gradient**2, dim=1) * dx


def histogram_density(values, bins, value_range=None):
    histogram, edges = np.histogram(values, bins=bins, range=value_range, density=True)
    if not np.isfinite(histogram).all():
        raise ValueError("The gradient histogram has no finite mass inside the configured range.")
    return histogram, edges


def physical_errors(predicted, reference, dx, nt, diagnostics):
    pred_power, true_power = terminal_spectrum(predicted), terminal_spectrum(reference)
    spectral = relative_norm_percent(
        np.log10(pred_power[1:] + diagnostics.spectral_floor),
        np.log10(true_power[1:] + diagnostics.spectral_floor),
    )
    start = int(diagnostics.late_time_fraction * nt)
    grad_pred = spatial_gradients(predicted, dx, start)
    grad_true = spatial_gradients(reference, dx, start)
    histogram_range = (diagnostics.gradient_min, diagnostics.gradient_max)
    true_hist, _ = histogram_density(
        grad_true.flatten().cpu().numpy(),
        diagnostics.gradient_histogram_bins,
        histogram_range,
    )
    pred_hist, _ = histogram_density(
        grad_pred.flatten().cpu().numpy(),
        diagnostics.gradient_histogram_bins,
        histogram_range,
    )
    pred_energy = gradient_energy(grad_pred, dx).mean(dim=0).cpu().numpy()
    true_energy = gradient_energy(grad_true, dx).mean(dim=0).cpu().numpy()
    return {
        "log_spectrum_error_percent": spectral,
        "gradient_pdf_error_percent": relative_norm_percent(pred_hist, true_hist),
        "gradient_energy_error_percent": relative_norm_percent(pred_energy, true_energy),
    }


def gaussian_uq_metrics(predicted, reference, diagnostics):
    mean = predicted.mean(dim=0).cpu().numpy()
    std = predicted.std(dim=0).cpu().numpy()
    observations = reference.cpu().numpy()
    coverage = []
    for z_score in diagnostics.coverage_z:
        lower, upper = mean - z_score * std, mean + z_score * std
        coverage.append(float(np.mean((observations >= lower) & (observations <= upper)) * 100))
    normalized = (observations - mean) / (std + diagnostics.gaussian_std_epsilon)
    crps = std * (
        normalized * (2 * stats.norm.cdf(normalized) - 1)
        + 2 * stats.norm.pdf(normalized)
        - 1 / np.sqrt(np.pi)
    )
    return {
        "coverage_90_percent": coverage[0],
        "coverage_95_percent": coverage[1],
        "coverage_99_percent": coverage[2],
        "gaussian_crps": float(np.mean(crps)),
    }


def structure_function(tensor, max_shift, dx):
    terminal = tensor[:, :, -1]
    values, separation = np.zeros(max_shift), np.zeros(max_shift)
    for shift in range(1, max_shift + 1):
        difference = terminal[:, shift:] - terminal[:, :-shift]
        values[shift - 1] = torch.mean(difference**2).item()
        separation[shift - 1] = shift * dx
    return separation, values


def summarize_fields(tensor, config):
    diagnostics = config.diagnostics
    dx = config.physics.length / (tensor.shape[1] - 1)
    start = int(diagnostics.late_time_fraction * tensor.shape[2])
    gradient = spatial_gradients(tensor, dx, start)
    energy = gradient_energy(gradient, dx)
    histogram, edges = histogram_density(
        gradient.flatten().cpu().numpy(),
        np.linspace(
            diagnostics.gradient_min, diagnostics.gradient_max, diagnostics.gradient_plot_edges
        ),
    )
    separation, s2 = structure_function(
        tensor,
        tensor.shape[1] // diagnostics.structure_max_shift_divisor,
        dx,
    )
    return {
        "mean": tensor.mean(dim=0).cpu(),
        "variance": tensor.var(dim=0).cpu(),
        "std": tensor.std(dim=0).cpu(),
        "spectrum": torch.from_numpy(terminal_spectrum(tensor)),
        "gradient_histogram": torch.from_numpy(histogram),
        "gradient_edges": torch.from_numpy(edges),
        "gradient_energy_mean": energy.mean(dim=0).cpu(),
        "gradient_energy_std": energy.std(dim=0).cpu(),
        "separation": torch.from_numpy(separation),
        "structure_function": torch.from_numpy(s2),
    }
