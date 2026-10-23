# Checkpoints

Store the five study state dictionaries here, or pass another directory to `sfno evaluate --checkpoints`.

| Key | Accepted filename |
| --- | --- |
| `standard` | `model_standard_fno.pt` |
| `spectral` | `model_fno_plus_spectral.pt` |
| `gradient` | `model_fno_plus_gradient.pt` |
| `residual` | `model_hybrid_s-pino.pt` or `model_hybrid_pino.pt` |
| `sfno` | `model_s-fno_ours.pt` |

Checkpoint keys and tensor dimensions must match the model's state dictionary. Loading is strict and uses PyTorch's `weights_only=True` path. Archives containing an entire pickled model object are not accepted.

Use `mapping.example.json` with `--mapping` to select alternate filenames or resolve two differing residual checkpoints. Hashes are recorded in evaluation outputs. Checkpoint identity and dimensional compatibility do not by themselves establish that a file produced a historical manuscript figure.

Pretrained checkpoints are not included in this repository. A new training run exports plain state dictionaries under its `checkpoints/` subdirectory. `.pt` and `.pth` files are excluded from Git by default; the repository source archive does not contain trained model weights.

