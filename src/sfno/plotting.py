import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sfno.config import Config
from sfno.evaluate import evaluation_source, load_evaluation
from sfno.experiments import ABLATIONS
from sfno.runtime import load_torch, prepare_output, resolve_device, sha256, write_manifest


@contextmanager
def plot_style():
    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.family": "serif",
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        yield


def array(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)


def finish_line(fig, ax, output, name, legends, **legend_options):
    if legends:
        ax.legend(frameon=False, **legend_options)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = Path(output) / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def contour(x, t, values, output, name, plotting, cmap="viridis", limits=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    values = array(values)
    options = {"levels": plotting.contour_levels}
    if limits:
        low, high = limits
        if low == high:
            high = low + 1e-8
        options = {"levels": np.linspace(low, high, plotting.contour_levels), "extend": "both"}
    image = ax.contourf(array(x), array(t), values.T, cmap=cmap, **options)
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    fig.tight_layout()
    path = Path(output) / f"{name}.png"
    fig.savefig(path, dpi=plotting.png_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_evaluation_contours(bundle, config, output):
    test, summaries = bundle["test"], bundle["summaries"]
    x, t = test["x"], test["t"]
    ref_var = array(summaries["reference"]["variance"])
    limits = float(ref_var.min()), float(ref_var.max())
    paths = [
        contour(x, t, ref_var, output, "variance_true_mcfdm", config.plotting, "magma", limits)
    ]
    for key, suffix in [("standard", "std_fno"), ("sfno", "s_fno")]:
        variance = array(summaries[key]["variance"])
        paths.append(
            contour(
                x, t, variance, output, f"variance_pred_{suffix}", config.plotting, "magma", limits
            )
        )
        paths.append(
            contour(
                x,
                t,
                np.abs(ref_var - variance),
                output,
                f"variance_error_{suffix}",
                config.plotting,
                "magma",
            )
        )
    index = config.diagnostics.realization_index
    true_field = array(test["target"][index])
    limits = float(true_field.min()), float(true_field.max())
    paths.append(
        contour(
            x,
            t,
            test["xi"][index],
            output,
            "single_realization_noise_field",
            config.plotting,
            "RdBu_r",
        )
    )
    paths.append(
        contour(x, t, true_field, output, "single_realization_true", config.plotting, limits=limits)
    )
    for key, suffix in [("standard", "std"), ("sfno", "s_fno")]:
        field = array(bundle["predictions"][key][index])
        paths.append(
            contour(
                x,
                t,
                field,
                output,
                f"single_realization_pred_{suffix}",
                config.plotting,
                limits=limits,
            )
        )
        paths.append(
            contour(
                x,
                t,
                np.abs(true_field - field),
                output,
                f"single_realization_error_{suffix}",
                config.plotting,
            )
        )
    return paths


def plot_cross_sections(bundle, config, output, legends):
    x, t = array(bundle["test"]["x"]), array(bundle["test"]["t"])
    summaries = bundle["summaries"]
    paths = []
    for fraction, name in zip(config.diagnostics.cross_section_fractions, ("t025", "t075")):
        index = int(fraction * len(t))
        fig, ax = plt.subplots(figsize=(8, 5))
        mean = array(summaries["reference"]["mean"])[:, index]
        std = array(summaries["reference"]["std"])[:, index]
        spread = config.diagnostics.cross_section_spread
        ax.plot(x, mean, color="#222222", lw=2.5, label="Reference mean")
        ax.fill_between(
            x,
            mean - spread * std,
            mean + spread * std,
            color="#BDBDBD",
            alpha=0.35,
            label=rf"Reference $\pm {spread:g}\sigma$",
        )
        for key, color, linestyle, label in [
            ("standard", "purple", "-.", "Standard FNO"),
            ("sfno", "#0072B2", "--", "S-FNO"),
        ]:
            mean = array(summaries[key]["mean"])[:, index]
            std = array(summaries[key]["std"])[:, index]
            ax.plot(x, mean, color=color, ls=linestyle, lw=2.0, label=f"{label} mean")
            ax.plot(
                x,
                mean + spread * std,
                color=color,
                ls=":",
                lw=1.5,
                label=rf"{label} $\pm {spread:g}\sigma$",
            )
            ax.plot(x, mean - spread * std, color=color, ls=":", lw=1.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel(f"$u(x,t={t[index]:.2f})$")
        ax.grid(alpha=0.2)
        paths.append(
            finish_line(
                fig,
                ax,
                output,
                f"uq_cross_section_{name}",
                legends,
                loc="upper right",
                fontsize=10,
                ncol=2,
            )
        )
    return paths


def plot_spectrum(bundle, config, output, legends):
    summaries = bundle["summaries"]
    reference = array(summaries["reference"]["spectrum"])
    modes = np.arange(len(reference))
    wavenumber = 2 * np.pi * modes / config.physics.length
    valid = wavenumber[1:]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.plot(valid, reference[1:], color="#111111", lw=3.2, label="Reference", zorder=10)
    for ablation in ABLATIONS:
        ax.plot(
            valid,
            array(summaries[ablation.key]["spectrum"])[1:],
            color=ablation.color,
            ls=ablation.linestyle,
            lw=2.5 if ablation.key == "sfno" else 1.5,
            label=ablation.name,
            zorder=8 if ablation.key == "sfno" else 3,
        )
    anchor = min(1, len(valid) - 1)
    dense = np.geomspace(valid[anchor], valid[-1], 500)
    ax.plot(
        dense,
        reference[1:][anchor] * (dense / valid[anchor]) ** -2,
        color="#AAAAAA",
        ls=":",
        lw=2,
        label=r"$k^{-2}$ reference",
    )
    cutoff = 2 * np.pi * config.architecture.modes / config.physics.length
    if cutoff <= valid[-1]:
        ax.axvspan(cutoff, valid[-1], color="gray", alpha=0.045)
        ax.axvline(cutoff, color="gray", ls="--", lw=1.8, alpha=0.75)
        ax.annotate(
            "Retained block edge",
            xy=(cutoff, 0.65),
            xycoords=("data", "axes fraction"),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10,
            rotation=90,
            va="center",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(valid[0], valid[-1])
    ax.set_xlabel(r"Nominal wavenumber $2\pi k/L$")
    ax.set_ylabel(r"Spatial power $E(k)$")
    ax.grid(True, which="major", alpha=0.18)
    ax.grid(True, which="minor", ls=":", alpha=0.08)
    return finish_line(
        fig,
        ax,
        output,
        "spatial_power_spectrum_all_models",
        legends,
        fontsize=10,
        ncol=2,
        loc="lower left",
    )


def plot_gradient_diagnostics(bundle, config, output, legends):
    summaries = bundle["summaries"]
    selected = [
        ("reference", "#222222", "-", "Reference"),
        ("standard", "purple", ":", "Standard FNO"),
        ("sfno", "#0072B2", "--", "S-FNO"),
    ]
    paths = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, color, style, label in selected:
        ax.stairs(
            array(summaries[key]["gradient_histogram"]),
            array(summaries[key]["gradient_edges"]),
            color=color,
            ls=style,
            lw=2.2,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Spatial gradient $\partial u/\partial x$")
    ax.set_ylabel("Probability density")
    ax.grid(True, which="both", ls="--", alpha=0.2)
    paths.append(finish_line(fig, ax, output, "gradient_pdf_heavy_tails", legends))
    start = int(config.diagnostics.late_time_fraction * config.data.nt)
    t = array(bundle["test"]["t"])[start:]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color, style, label in selected:
        ax.plot(
            t,
            array(summaries[key]["gradient_energy_mean"]),
            color=color,
            ls=style,
            lw=2.2,
            label=label,
        )
    mean = array(summaries["reference"]["gradient_energy_mean"])
    spread = config.diagnostics.gradient_energy_spread * array(
        summaries["reference"]["gradient_energy_std"]
    )
    ax.fill_between(
        t,
        mean - spread,
        mean + spread,
        color="#BDBDBD",
        alpha=0.35,
        label=rf"Reference $\pm {config.diagnostics.gradient_energy_spread:g}\sigma$",
    )
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Mean gradient energy $\Omega(t)$")
    ax.grid(True, ls="--", alpha=0.2)
    paths.append(
        finish_line(fig, ax, output, "ensemble_enstrophy_evolution", legends, loc="upper right")
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for (key, color, style, label), marker in zip(selected, ("o", "^", "s")):
        ax.plot(
            array(summaries[key]["separation"]),
            array(summaries[key]["structure_function"]),
            color=color,
            ls=style,
            marker=marker,
            lw=2.2,
            markersize=4,
            label=label,
        )
    separation = array(summaries["reference"]["separation"])
    values = array(summaries["reference"]["structure_function"])
    anchor = min(2, len(values) - 1)
    ax.plot(
        separation,
        values[anchor] * (separation / separation[anchor]),
        color="#7A7A7A",
        ls="--",
        lw=1.8,
        label=r"$S_2\propto r$ reference",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Separation $r$")
    ax.set_ylabel(r"Structure function $S_2(r)$")
    ax.grid(True, which="both", ls="--", alpha=0.2)
    paths.append(finish_line(fig, ax, output, "structure_function_s2", legends))
    return paths


def plot_resolution(bundle, config, output):
    x, t = bundle["x"], bundle["t"]
    reference, predicted = array(bundle["reference"])[0], array(bundle["predicted"])[0]
    limits = float(reference.min()), float(reference.max())
    suffix = f"{config.resolution.nx}x{config.resolution.nt}"
    paths = [
        contour(
            x,
            t,
            reference,
            output,
            f"super_resolution_true_{suffix}",
            config.plotting,
            limits=limits,
        ),
        contour(
            x,
            t,
            predicted,
            output,
            f"super_resolution_pred_sfno_{suffix}",
            config.plotting,
            limits=limits,
        ),
        contour(
            x,
            t,
            np.abs(reference - predicted),
            output,
            f"super_resolution_error_sfno_{suffix}",
            config.plotting,
        ),
    ]
    if "legacy_unpaired_diagnostic" in bundle:
        folder = Path(output) / "legacy_unpaired"
        folder.mkdir(exist_ok=True)
        interpolated = array(bundle["legacy_unpaired_diagnostic"]["interpolated"])[0]
        paths.append(
            contour(
                x, t, interpolated, folder, "unpaired_interpolation", config.plotting, limits=limits
            )
        )
        paths.append(
            contour(
                x,
                t,
                np.abs(reference - interpolated),
                folder,
                "unpaired_difference",
                config.plotting,
            )
        )
    return paths


def plot_inverse(bundle, config, output, legends):
    x, t = bundle["x"], bundle["t"]
    true_forcing = array(bundle["true_forcing"])[0]
    recovered = array(bundle["last_evaluated"]["xi"])[0]
    paths = []
    for values, name, cmap in [
        (true_forcing, "inverse_1a_true_forcing", "RdBu_r"),
        (recovered, "inverse_1b_recovered_forcing", "RdBu_r"),
        (array(bundle["fluid_from_true"])[0], "inverse_2a_fluid_from_true", "viridis"),
        (
            array(bundle["last_evaluated"]["predicted"])[0],
            "inverse_2b_fluid_from_recovered",
            "viridis",
        ),
    ]:
        paths.append(contour(x, t, values, output, name, config.plotting, cmap))
    true_fft = torch.fft.rfft(bundle["true_forcing"][0], dim=0)
    recovered_fft = torch.fft.rfft(bundle["last_evaluated"]["xi"][0], dim=0)
    true_power = array(torch.mean(torch.abs(true_fft) ** 2, dim=1))
    recovered_power = array(torch.mean(torch.abs(recovered_fft) ** 2, dim=1))
    modes = np.arange(len(true_power))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(modes, true_power, color="#222222", lw=2.5, label="True forcing")
    ax.plot(modes, recovered_power, color="#0072B2", ls="--", lw=2.5, label="Recovered forcing")
    ax.axvline(
        config.architecture.modes, color="#7A7A7A", ls=":", lw=2, label="Retained block edge"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Spatial Fourier index")
    ax.set_ylabel(r"Time-averaged forcing power $E_\xi(k)$")
    ax.grid(True, ls="--", alpha=0.2)
    paths.append(finish_line(fig, ax, output, "forcing_power_spectrum", legends))
    return paths


def render_figures(
    evaluation_path,
    output,
    resolution_path=None,
    inverse_path=None,
    training_path=None,
    no_legends=False,
):
    bundle, config = load_evaluation(evaluation_path)
    output = prepare_output(output)
    legends = config.plotting.legends and not no_legends
    sources = [evaluation_source(evaluation_path)]
    with plot_style():
        paths = plot_evaluation_contours(bundle, config, output)
        paths += plot_cross_sections(bundle, config, output, legends)
        paths.append(plot_spectrum(bundle, config, output, legends))
        paths += plot_gradient_diagnostics(bundle, config, output, legends)
        for value, filename, kind in [
            (resolution_path, "resolution.pt", "resolution"),
            (inverse_path, "inverse.pt", "inverse"),
        ]:
            if value is None:
                continue
            path = Path(value)
            path = path / filename if path.is_dir() else path
            supplemental = load_torch(path)
            if supplemental.get("kind") != f"sfno_{kind}":
                raise ValueError(f"Expected an {kind} bundle.")
            if supplemental["evaluation"]["sha256"] != sources[0]["sha256"]:
                raise ValueError(f"The {kind} bundle belongs to a different evaluation.")
            if Config.from_dict(supplemental["config"]).fingerprint() != config.fingerprint():
                raise ValueError(f"The {kind} configuration differs from the evaluation.")
            sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
            paths += (
                plot_resolution(supplemental, config, output)
                if kind == "resolution"
                else plot_inverse(supplemental, config, output, legends)
            )
        if training_path:
            training_path = Path(training_path)
            stored_config = Config.from_dict(
                json.loads((training_path / "config.json").read_text())
            )
            if stored_config.fingerprint() != config.fingerprint():
                raise ValueError("The training configuration differs from the evaluation.")
            path = training_path / "histories.json"
            histories = json.loads(path.read_text())
            fig, ax = plt.subplots(figsize=(8, 6))
            for ablation in ABLATIONS:
                ax.plot(
                    histories[ablation.key],
                    color=ablation.color,
                    ls=ablation.linestyle,
                    lw=2,
                    label=ablation.name,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Total training objective")
            ax.set_yscale("log")
            ax.grid(True, ls="--", alpha=0.5)
            paths.append(
                finish_line(
                    fig, ax, output, "training_convergence_comparison", legends, fontsize=11
                )
            )
            sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    write_manifest(
        output,
        "plot",
        config,
        resolve_device("cpu"),
        sources,
        paths,
        {
            "line_format": "pdf",
            "contour_format": "png",
            "legends": legends,
            "inverse_plot_iterate": "last_evaluated" if inverse_path else None,
        },
    )
    return paths
