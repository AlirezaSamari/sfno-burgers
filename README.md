# Spectrally logarithmic regularied Fourier NEural Operators (S-FNO) for randomly forced Burgers dynamics

Modular PyTorch implementation of the five-objective Fourier neural operator study, including data generation, training, ensemble diagnostics, finer-grid inference, inverse forcing recovery, and figure export.

The `paper` configuration specifies the physical parameters, data generation, model architecture, training objectives and evaluation settings used in this study. Separate commands run each stage with saved configuration and random-state information.

## Start here

1. Inspect [the paper configuration](src/sfno/configs/paper.json).
2. Read [the implemented methods](docs/METHODS.md) and [the validation record](docs/VALIDATION.md).
3. Run the numerical equivalence tests and the small integration workflow below.
4. Evaluate the study checkpoints, or regenerate data and train all five objectives.

**Availability:** pretrained checkpoints are not included in this repository. Recorded study metrics are provided as reference values, separately from generated outputs. Reproducing those values requires the corresponding checkpoints, evaluation data and a compatible numerical environment. Dependency versions used for the original reported experiments have not been independently verified.

## Installation

The checked environment uses Python 3.12 and the dependencies in `requirements-tested.txt`. For a CPU installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt
python -m pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\activate`. For GPU training, install the appropriate PyTorch 2.6.0 CUDA build using the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/), then install the remaining requirements and this package. CUDA is recommended for the full training schedule.

Both `sfno` and `python -m sfno` expose the same commands. The examples use the latter.

## Reviewer check

```bash
python -m sfno inspect
python -m pytest -q
python -m sfno smoke --output runs/smoke
python -m sfno verify --run runs/smoke
```

Numerical regression tests compare the forcing, initial conditions, reference solver, model initialization and forward pass, losses and their derivatives, and reported metric formulas against a reference implementation. They also check checkpoint aliases, random-state serialization, and interrupted training against uninterrupted training.

The `smoke` profile uses small grids, small networks, two epochs per model, and four inverse iterations. Its outputs check execution only; they are not manuscript results. The full `paper` profile is never shortened automatically. [Validation details](docs/VALIDATION.md) distinguish tested behavior from unexecuted full-scale experiments.

## Evaluate existing study checkpoints

Place the five plain PyTorch state dictionaries in `checkpoints/`:

| Objective | Filename |
| --- | --- |
| Spatial MSE | `model_standard_fno.pt` |
| MSE + logarithmic spectral penalty | `model_fno_plus_spectral.pt` |
| MSE + increment-magnitude penalty | `model_fno_plus_gradient.pt` |
| MSE + PDE residual penalty | `model_hybrid_s-pino.pt` |
| MSE + both supervised penalties | `model_s-fno_ours.pt` |

`model_hybrid_pino.pt` is also accepted for the residual model. If both residual filenames exist with different contents, supply an explicit mapping; neither file is chosen silently. `checkpoints/mapping.example.json` shows the format. Checkpoints must be plain PyTorch state dictionaries with keys and tensor dimensions matching the model.

```bash
python -m sfno checkpoints --directory checkpoints
python -m sfno generate --output runs/paper-data --device cuda:0
python -m sfno evaluate --data runs/paper-data/dataset.pt --checkpoints checkpoints --output runs/evaluation --device cuda:0
python -m sfno resolution --evaluation runs/evaluation --output runs/resolution --device cuda:0
python -m sfno inverse --evaluation runs/evaluation --output runs/inverse --device cuda:0
python -m sfno plot --evaluation runs/evaluation --resolution runs/resolution --inverse runs/inverse --output runs/figures
python -m sfno compare --evaluation runs/evaluation --reference docs/reported_metrics.json
```

Use `--mapping checkpoints/mapping.example.json` on `checkpoints` or `evaluate` when a mapping is needed. Paths inside the mapping are relative to the checkpoint directory unless absolute. If a checkpoint is moved after evaluation, pass its new path with `--checkpoint` to `resolution`, `inverse`, or `benchmark`; its hash must match the evaluated file.

Data generation deliberately draws the 800 training samples before the 500 test samples, including for evaluation. Skipping the first draws would change the test ensemble. Dataset generation and random-state replay must use the same device type. Replace `cuda:0` with `cpu` throughout for a CPU run; this does not imply numerical identity with the original GPU run.

## Train from scratch

```bash
python -m sfno reproduce --output runs/paper --device cuda:0
```

This runs the entire workflow using the paper settings. Alternatively, reuse an existing dataset cache:

```bash
python -m sfno train --data runs/paper-data/dataset.pt --output runs/training --device cuda:0
python -m sfno evaluate --data runs/paper-data/dataset.pt --checkpoints runs/training/checkpoints --output runs/evaluation-trained --device cuda:0
```

Training creates all five independently initialized models before training them sequentially in the configured order: standard, spectral, gradient, residual and S-FNO. All 800 training samples are used with the same shuffled batch sampler and one sequential random stream. The exported weights are the final weights after 1,301 epochs; no best-epoch selection is performed.

The latest resumable state contains all five model states, optimizer states, scheduler states, histories, and random states. It is saved every 100 completed epochs and at each model boundary. This file is substantially larger than an individual inference checkpoint.

```bash
python -m sfno train --data runs/paper-data/dataset.pt --output runs/training --device cuda:0 --resume
```

`--stop-after 10` stops after ten additional model epochs at a resumable boundary. `--resume` restores the saved configuration and dataset identity. Other commands require a new or empty output directory so previous results are not overwritten accidentally.

## Settings

| Component | Paper setting |
| --- | --- |
| Seed | 42 |
| Domain | Space and time intervals `[0, 1]` |
| Viscosity / noise scale | `0.02 / pi` / `0.5` |
| Forcing basis | 16 spatial × 16 temporal sine modes |
| Training / test grid | 64 × 64, endpoint-inclusive |
| Training / test samples | 800 / 500 |
| Initial-field covariance lengths | 0.2 / 0.1 |
| Initial-field maximum amplitude | 1.5 |
| Generation / training / inference batches | 50 / 16 / 50 |
| Solver substeps | At least 200 per saved interval |
| FNO | Width 64, four blocks, one 16 × 16 Fourier corner |
| Projection | 64 → 128 → 1, GELU |
| Epochs | 1,301, indexed 0–1,300 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Scheduler / clipping | StepLR, 100 epochs, factor 0.5 / norm 1.0 |
| Spectral / increment / residual weights | 0.1 / 0.05 / 0.1, according to objective |
| Finer-grid test | One fresh 128 × 128 realization |
| Inverse fit | Test index 10, 2,000 Adam updates, initial learning rate 0.05 |
| Inverse regularization / scheduler | 0.0005 × mean squared coefficients / 200 updates, factor 0.5 |

All numerical settings, including floors, histogram definitions and Gaussian quantiles, appear in the JSON configuration. Print them with `python -m sfno settings`. Custom JSON files can be passed to `generate`, `reproduce`, `inspect`, and `checkpoints` with `--config`. Downstream stages inherit the saved configuration instead of silently substituting new settings.

## Outputs

| Stage | Main outputs |
| --- | --- |
| `generate` | `dataset.pt`, configuration, provenance manifest |
| `train` | Five plain state dictionaries, resume state, histories, manifest |
| `evaluate` | `table1.csv`, `table2.csv`, `metrics.json`, prediction/diagnostic bundle, manifest |
| `resolution` | Inputs, coefficients, reference field, S-FNO field, manifest |
| `inverse` | True inputs, both final-iteration states, loss history, fit summary, manifest |
| `plot` | PDF line plots, PNG contours, figure manifest |
| `benchmark` | Raw synchronized timing samples and environment metadata |

Each manifest records configuration values, input hashes, output hashes, package versions, device information, and numerical backend flags. `verify` checks listed artifact hashes; it does not establish historical checkpoint provenance or scientific validity. `compare` reports differences from recorded study results without treating the reference values as newly reproduced results.

## Figures

Line plots are native PDFs with embedded TrueType fonts and editable text. Contours are PNGs at 600 DPI in the paper profile. To export line figures without legends:

```bash
python -m sfno plot --evaluation runs/evaluation --resolution runs/resolution --inverse runs/inverse --output runs/figures-no-legends --no-legends
```

Figures use cached predictions and the diagnostic definitions in [the implemented methods](docs/METHODS.md). Labels distinguish ensemble spread from confidence intervals, nominal Fourier indices from the stencil period, and increment magnitudes from topology. Reference/prediction contours use shared levels when shared limits are specified. The manuscript workflow diagram is not recreated or modified.

## Additional diagnostics

```bash
python -m sfno benchmark --evaluation runs/evaluation --output runs/benchmark --device cuda:0
```

This measures the evaluated S-FNO and reference solver on the same input and device, with synchronization and raw timing samples.

The unpaired bilinear diagnostic is available only with `resolution --include-unpaired-diagnostic`. It interpolates coarse test index 0, a different realization from the finer-grid input. Its output is marked unpaired and its figures are isolated under `legacy_unpaired/`. It is not included in error tables or the default workflow.

## Development

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
python -m build
```

The GitHub Actions workflow runs linting, numerical tests and the CPU integration workflow. The source is organized by scientific responsibility under `src/sfno/`. Scientific definitions and implementation notes are in `docs/`.

