import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class Physics:
    length: float = 1.0
    duration: float = 1.0
    viscosity: float = 0.02 / math.pi
    noise_scale: float = 0.5
    forcing_modes_x: int = 16
    forcing_modes_t: int = 16
    initial_amplitude: float = 1.5
    covariance_jitter: float = 1e-5
    initial_normalization_epsilon: float = 1e-8
    minimum_substeps: int = 200
    diffusion_cfl_divisor: float = 2.5
    advection_cfl_speed: float = 4.0


@dataclass(frozen=True)
class Data:
    nx: int = 64
    nt: int = 64
    train_samples: int = 800
    test_samples: int = 500
    generation_batch_size: int = 50
    train_length_scale: float = 0.2
    test_length_scale: float = 0.1


@dataclass(frozen=True)
class Architecture:
    modes: int = 16
    width: int = 64
    num_layers: int = 4
    projection_width: int = 128


@dataclass(frozen=True)
class Training:
    epochs: int = 1301
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    gradient_clip: float = 1.0
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    spectral_weight: float = 0.1
    increment_weight: float = 0.05
    residual_weight: float = 0.1
    spectral_floor: float = 1e-8
    checkpoint_interval: int = 100


@dataclass(frozen=True)
class Diagnostics:
    inference_batch_size: int = 50
    spectral_floor: float = 1e-8
    late_time_fraction: float = 0.5
    gradient_histogram_bins: int = 100
    gradient_plot_edges: int = 100
    gradient_min: float = -30.0
    gradient_max: float = 15.0
    structure_max_shift_divisor: int = 3
    realization_index: int = 20
    cross_section_fractions: tuple = (0.25, 0.75)
    coverage_z: tuple = (1.645, 1.960, 2.576)
    gaussian_std_epsilon: float = 1e-8
    cross_section_spread: float = 2.0
    gradient_energy_spread: float = 1.96


@dataclass(frozen=True)
class Resolution:
    nx: int = 128
    nt: int = 128
    length_scale: float = 0.1


@dataclass(frozen=True)
class Inverse:
    observation_index: int = 10
    seed: int = 42
    iterations: int = 2000
    learning_rate: float = 0.05
    regularization: float = 5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    scheduler_step: int = 200
    scheduler_gamma: float = 0.5


@dataclass(frozen=True)
class Plotting:
    png_dpi: int = 600
    contour_levels: int = 100
    legends: bool = True


@dataclass(frozen=True)
class Benchmark:
    warmup: int = 20
    model_repeats: int = 100
    solver_repeats: int = 10


@dataclass(frozen=True)
class Config:
    profile: str = "paper"
    seed: int = 42
    physics: Physics = field(default_factory=Physics)
    data: Data = field(default_factory=Data)
    architecture: Architecture = field(default_factory=Architecture)
    training: Training = field(default_factory=Training)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    resolution: Resolution = field(default_factory=Resolution)
    inverse: Inverse = field(default_factory=Inverse)
    plotting: Plotting = field(default_factory=Plotting)
    benchmark: Benchmark = field(default_factory=Benchmark)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def validate(self):
        positive = (
            self.physics.length,
            self.physics.duration,
            self.physics.viscosity,
            self.physics.forcing_modes_x,
            self.physics.forcing_modes_t,
            self.physics.initial_amplitude,
            self.physics.covariance_jitter,
            self.physics.initial_normalization_epsilon,
            self.physics.minimum_substeps,
            self.physics.diffusion_cfl_divisor,
            self.physics.advection_cfl_speed,
            self.data.generation_batch_size,
            self.data.train_length_scale,
            self.data.test_length_scale,
            self.architecture.modes,
            self.architecture.width,
            self.architecture.num_layers,
            self.architecture.projection_width,
            self.training.epochs,
            self.training.batch_size,
            self.training.learning_rate,
            self.training.gradient_clip,
            self.training.scheduler_step,
            self.training.scheduler_gamma,
            self.training.spectral_floor,
            self.training.checkpoint_interval,
            self.diagnostics.inference_batch_size,
            self.diagnostics.spectral_floor,
            self.diagnostics.gaussian_std_epsilon,
            self.diagnostics.structure_max_shift_divisor,
            self.inverse.iterations,
            self.inverse.learning_rate,
            self.inverse.scheduler_step,
            self.inverse.scheduler_gamma,
            self.resolution.length_scale,
            self.plotting.png_dpi,
            self.benchmark.model_repeats,
            self.benchmark.solver_repeats,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError(
                "All scale, size and iteration parameters must be positive and finite."
            )
        if min(self.data.nx, self.data.nt, self.resolution.nx, self.resolution.nt) < 4:
            raise ValueError("Grid dimensions must be at least four.")
        if min(self.data.train_samples, self.data.test_samples) < 2:
            raise ValueError("At least two samples are required per split.")
        for nx, nt in [(self.data.nx, self.data.nt), (self.resolution.nx, self.resolution.nt)]:
            if self.architecture.modes > min(nx, nt // 2 + 1):
                raise ValueError("The retained modes exceed the available Fourier grid.")
        if not 0 <= self.diagnostics.late_time_fraction < 1:
            raise ValueError("late_time_fraction must lie in [0, 1).")
        if not 0 <= self.diagnostics.realization_index < self.data.test_samples:
            raise ValueError("realization_index is outside the test split.")
        if not 0 <= self.inverse.observation_index < self.data.test_samples:
            raise ValueError("observation_index is outside the test split.")
        if len(self.diagnostics.cross_section_fractions) != 2 or any(
            not 0 <= value < 1 for value in self.diagnostics.cross_section_fractions
        ):
            raise ValueError("Two cross-section fractions in [0, 1) are required.")
        if len(self.diagnostics.coverage_z) != 3:
            raise ValueError("Three Gaussian coverage z scores are required.")
        if self.diagnostics.gradient_min >= self.diagnostics.gradient_max:
            raise ValueError("The histogram range must have positive width.")
        if min(self.diagnostics.gradient_histogram_bins, self.diagnostics.gradient_plot_edges) < 2:
            raise ValueError("At least two histogram bins or edges are required.")
        if self.plotting.contour_levels < 2 or self.benchmark.warmup < 0:
            raise ValueError("Invalid plotting levels or benchmark warmup.")
        if self.data.nx // self.diagnostics.structure_max_shift_divisor < 1:
            raise ValueError("The structure function needs at least one spatial shift.")
        for section in (self.training, self.inverse):
            if not 0 <= section.adam_beta1 < 1 or not 0 <= section.adam_beta2 < 1:
                raise ValueError("Adam betas must lie in [0, 1).")
            if section.adam_epsilon <= 0:
                raise ValueError("Adam epsilon must be positive.")
        if (
            min(
                self.training.weight_decay,
                self.training.spectral_weight,
                self.training.increment_weight,
                self.training.residual_weight,
                self.inverse.regularization,
            )
            < 0
        ):
            raise ValueError("Regularization weights cannot be negative.")
        return self

    @classmethod
    def from_dict(cls, value):
        sections = {
            "physics": Physics,
            "data": Data,
            "architecture": Architecture,
            "training": Training,
            "diagnostics": Diagnostics,
            "resolution": Resolution,
            "inverse": Inverse,
            "plotting": Plotting,
            "benchmark": Benchmark,
        }
        unknown = set(value) - {"profile", "seed", *sections}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        converted = dict(value)
        for key, constructor in sections.items():
            if key in value:
                converted[key] = constructor(**value[key])
        return cls(**converted).validate()


def load_config(path=None, profile="paper"):
    source = Path(path) if path else files("sfno").joinpath(f"configs/{profile}.json")
    return Config.from_dict(json.loads(source.read_text(encoding="utf-8")))
