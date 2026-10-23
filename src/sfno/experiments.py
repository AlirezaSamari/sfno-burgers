from dataclasses import dataclass


@dataclass(frozen=True)
class Ablation:
    key: str
    name: str
    filename: str
    color: str
    linestyle: str
    spectral: bool = False
    increment: bool = False
    residual: bool = False
    aliases: tuple = ()


ABLATIONS = (
    Ablation("standard", "Standard FNO", "model_standard_fno.pt", "purple", ":"),
    Ablation(
        "spectral", "FNO + Spectral", "model_fno_plus_spectral.pt", "red", "--", spectral=True
    ),
    Ablation(
        "gradient", "FNO + Gradient", "model_fno_plus_gradient.pt", "#E69F00", "--", increment=True
    ),
    Ablation(
        "residual",
        "Hybrid PINO",
        "model_hybrid_s-pino.pt",
        "green",
        "-.",
        residual=True,
        aliases=("model_hybrid_pino.pt",),
    ),
    Ablation("sfno", "S-FNO", "model_s-fno_ours.pt", "#0072B2", "-", spectral=True, increment=True),
)


def objective_weights(ablation, training):
    return (
        training.spectral_weight if ablation.spectral else 0.0,
        training.increment_weight if ablation.increment else 0.0,
        training.residual_weight if ablation.residual else 0.0,
    )
