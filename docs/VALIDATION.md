# Validation record

## Environment

Validation was performed on CPU with Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8, pytest 9.1.1 and Ruff 0.17.2. Numerical tests use one PyTorch CPU thread. Dependency versions used for the original reported experiments have not been independently verified.

## Completed checks

| Check | Outcome |
| --- | --- |
| Numerical and workflow suite | 22 tests passed |
| Forcing and initial-condition functions | Exact tensor equality with the reference implementation on CPU |
| Reference solver at the 64 × 64 paper grid | Exact tensor equality with the reference implementation for two realizations |
| Model initialization and forward evaluation at paper dimensions | Exact state-dictionary and output equality |
| Composite-loss values and input derivatives | Exact equality for all four supervised weight combinations |
| Residual-loss value and input derivatives | Exact equality |
| Physical, Gaussian-marginal and structure-function formulas | Exact equality with the reference implementation |
| Five-model sequential training order | Exact parameter and loss-history equality for a reduced one-epoch comparison with the reference training loop |
| Interrupted versus uninterrupted training | Exact final parameter and loss-history equality for all five reduced models |
| Random-state serialization | Round trip through restricted tensor/primitive loading |
| Checkpoint aliases | Identical aliases accepted; differing aliases require an explicit mapping |
| Dataset/configuration mismatch and checkpoint mismatch | Rejected |
| Artifact corruption | Detected by manifest verification |
| Reduced complete workflow | Generation, all five training objectives, evaluation, finer-grid inference, inverse fitting and plotting completed |
| Figure exports | 8 native PDF line plots and 18 PNG contours in the reduced workflow |
| PDF inspection | Embedded TrueType fonts, extractable text and no raster image objects in all eight line figures |
| Optional command paths | Benchmark, unpaired diagnostic, legend-free plotting, metric comparison and artifact verification completed |
| Missing-checkpoint command | Returned nonzero exit status |
| Packaging | Source distribution and wheel built; wheel installed into a separate directory and CLI/configuration resources loaded successfully |
| Style | Ruff lint and formatting checks passed |

The integration outputs use the explicitly reduced `smoke` profile. Their metric values are not scientific results and are not included in `reported_metrics.json`. The optional timing command was exercised only on this reduced profile; no manuscript speedup claim was evaluated.

## Scope

- Full 1,301-epoch retraining was not performed during packaging.
- Full 800/500-trajectory regeneration was not performed during packaging.
- Pretrained checkpoints are not included in this repository; reported table values are reference records and were not independently recomputed during packaging.
- CUDA execution was not tested in the validation environment.
- The GitHub Actions workflow is supplied for repository use; a remote GitHub Actions run was not performed.
- Original manuscript figure assets were not altered or asserted to match newly generated figures.

The numerical regression checks compare implemented operators, objectives and diagnostics with a reference implementation. They do not establish continuum-solver convergence, independent uncertainty calibration, generalization beyond the reported problem, or the provenance of unavailable historical checkpoints.

