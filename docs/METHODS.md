# Implemented methods

This document describes the code being executed. Array order is `(realization, space, saved time)`. Field calculations use single-precision real tensors; Fourier weights use complex single precision. The package does not enable mixed precision, compilation, or additional deterministic-backend flags.

## Reference dynamics

The nominal equation is

$$
\partial_t u+u\partial_xu=\nu\partial_{xx}u+\sigma\xi u,
\qquad \nu=0.02/\pi,\quad\sigma=0.5.
$$

For independently sampled standard normal coefficients,

$$
\xi(x,t)=\frac{1}{\sqrt{M_xM_t}}
\sum_{m=1}^{M_x}\sum_{n=1}^{M_t}
z_{mn}\sqrt{\frac{2}{L}}\sin\frac{m\pi x}{L}
\sqrt{\frac{2}{T}}\sin\frac{n\pi t}{T},
\qquad M_x=M_t=16.
$$

The initial vector is sampled with covariance

$$
K_{ij}=\exp\left[-\frac{(x_i-x_j)^2}{2\ell^2}\right],
\quad v=\operatorname{chol}(K+10^{-5}I)g,
\quad u_0=1.5\frac{v-\overline v}{\max_i|v_i-\overline v|+10^{-8}}.
$$

Here `g` has independent standard normal entries. The normalization makes the resulting initial vector non-Gaussian. The code samples forcing coefficients first and initial-condition Gaussian vectors second in each generation batch.

Both grids include both endpoints. With $h=L/(N_x-1)$ and saved-time step $\Delta t=T/(N_t-1)$,

$$
\delta t_{\mathrm{safe}}=
\min\left(\frac{h^2}{2.5\nu},\frac{h}{4}\right),
\qquad
n_s=\max\left(\left\lceil\frac{\Delta t}{\delta t_{\mathrm{safe}}}\right\rceil,200\right),
\qquad \delta t=\Delta t/n_s.
$$

At substep $s=0,\ldots,n_s-1$, forcing is linearly interpolated between its two saved-time values with weight $s/n_s$. Each explicit update is

$$
u_i^{r+1}=u_i^r+\delta t\left[-u_i^r D_{\mathrm{up}}u_i^r+
\nu\frac{u_{i+1}^r-2u_i^r+u_{i-1}^r}{h^2}+\sigma\xi_i^r u_i^r\right],
$$

where $D_{\mathrm{up}}u_i=(u_i-u_{i-1})/h$ for $u_i>0$, and $(u_{i+1}-u_i)/h$ otherwise. Neighbors are implemented with `torch.roll`, including across the array seam. Consequently, this is a circular-index discrete target on an endpoint-inclusive coordinate grid. It is not a zero-Dirichlet solver or the conventional endpoint-excluded periodic discretization; the stencil period is $N_xh$. This fixed step-size rule does not establish adaptive stability or solver convergence.

## Fourier operator

The four input channels are `(x, t, u0(x), xi(x,t))`. A linear layer lifts them to width 64. Each of four blocks applies

$$
v_{j+1}=\operatorname{GELU}\left[
\mathcal F^{-1}\!\left(R_j\mathcal F(v_j)\right)+W_jv_j+b_j\right].
$$

The transform is `torch.fft.rfft2` with its default normalization. Only `[:16, :16]` is parameterized in the transformed space-time array. There is no independently learned negative-spatial-frequency block. A pointwise convolution provides the local branch. Complex weights are initialized as `(1 / width**2) * torch.randn(..., dtype=torch.cfloat)` before moving the model to the execution device. Projection uses linear layers `64 → 128 → 1` with GELU between them.

The model has 4,219,713 parameter entries when complex entries count once, or 8,414,017 real scalar components. Pointwise branches and nonlinearities can produce output frequencies outside the retained Fourier block.

## Training objectives

Let $\widehat u_{bkn}$ denote the spatial `rfft(..., dim=1, norm="forward")`, and $P_{bkn}=|\widehat u_{bkn}|^2$. All indicated averages include every realization and saved time in the minibatch.

$$
\mathcal L_{\mathrm{data}}=\operatorname{mean}(u_{\mathrm{pred}}-u_{\mathrm{ref}})^2,
$$

$$
\mathcal L_{\mathrm{spec}}=\operatorname{mean}
\left[\log_{10}(P_{\mathrm{pred}}+10^{-8})-
\log_{10}(P_{\mathrm{ref}}+10^{-8})\right]^2,
$$

$$
\mathcal L_{\mathrm{inc}}=\operatorname{mean}
\left[|u^{\mathrm{pred}}_{i+1}-u^{\mathrm{pred}}_i|-
|u^{\mathrm{ref}}_{i+1}-u^{\mathrm{ref}}_i|\right]^2.
$$

The spectral loss includes DC and every one-sided spatial mode. The increment penalty uses adjacent raw differences, omits the circular seam, and does not divide by grid spacing. It measures increment magnitudes, not signed derivatives or a topological invariant. If written in terms of absolute finite-difference derivatives, its weight includes a factor $h^2$.

| Objective | Data | Spectral | Increments | Residual |
| --- | ---: | ---: | ---: | ---: |
| Standard FNO | 1 | 0 | 0 | 0 |
| FNO + Spectral | 1 | 0.1 | 0 | 0 |
| FNO + Gradient | 1 | 0 | 0.05 | 0 |
| Hybrid PINO | 1 | 0 | 0 | 0.1 |
| S-FNO | 1 | 0.1 | 0.05 | 0 |

For the residual objective, time derivatives use forward differences. Spatial derivatives use a full complex FFT and `fftfreq(Nx, d=h)`:

$$
\mathcal L_{\mathrm{res}}=\operatorname{mean}
\left[D_tu+uD_x^{\mathcal F}u-\nu D_{xx}^{\mathcal F}u-\sigma\xi u\right]^2.
$$

The residual is evaluated at all saved times except the last, uses the left-time forcing, and includes all spatial indices. It is not the reference solver's upwind residual. The initial field enters through the operator inputs; the residual objective adds no initial-condition or boundary penalty.

All 800 samples are training data. There is no held-out validation subset. AdamW uses learning rate $10^{-3}$, weight decay $10^{-4}$, betas $(0.9,0.999)$, and epsilon $10^{-8}$. Each update clips the global gradient norm to 1.0. StepLR is advanced after each epoch, with step size 100 and factor 0.5. Epochs run from 0 through 1,300 inclusive. The loss history is the unweighted mean of minibatch losses.

## Random sequence and execution order

1. Seed PyTorch and NumPy with 42.
2. Generate 800 training trajectories in batches of 50 at $\ell=0.2$.
3. Generate 500 test trajectories in batches of 50 at $\ell=0.1$.
4. Construct all five models in the order listed above, without reseeding between them.
5. Train models sequentially with the same shuffled `DataLoader`, batch size 16, zero worker processes, and no dropped batch.

Dataset caches store the random state after step 3. The training command restores it before model construction. The evaluation command separately restores it, constructs/loads the five models in the same order, and stores the state after model construction for the finer-grid experiment. Evaluating a model does not consume random numbers in this architecture. The finer-grid sample is generated from the stored evaluation random state, so executing stages as separate commands preserves the configured draw order.

PyTorch does not guarantee identical results across releases, devices, or platforms. The package therefore records environment versions and backend settings and does not claim that seed 42 alone establishes bitwise historical reproduction. See the [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Table 1

Mean and variance errors are relative Euclidean norms over the space-time arrays, expressed as percentages. Variance and standard deviation use PyTorch's sample estimates with correction 1.

The terminal spectrum is an ensemble mean of power, followed by a logarithm. Its reported diagnostic is

$$
100\frac{\left\|\log_{10}(\overline P_{\mathrm{pred}}+10^{-8})-
\log_{10}(\overline P_{\mathrm{ref}}+10^{-8})\right\|_2}
{\left\|\log_{10}(\overline P_{\mathrm{ref}}+10^{-8})\right\|_2},
\qquad k=1,\ldots,N_x/2.
$$

DC is omitted here. This differs from the per-realization, all-time training penalty and is not a relative raw-power error.

The centered interior gradient is

$$
g_{bin}=\frac{u_{b,i+1,n}-u_{b,i-1,n}}{2h}.
$$

Gradient diagnostics use saved-time indices 32–63 and interior spatial indices 1–62. The table uses a density histogram with **100 bins** over `[-30, 15]`; values outside the range do not contribute to the density normalization. The gradient-PDF error is a relative Euclidean norm of those histogram values. The figure uses **100 edges**, hence 99 bins.

The gradient-energy proxy is

$$
\Omega_b(t_n)=\frac{h}{2}\sum_{i=1}^{N_x-2}g_{bin}^2.
$$

Its table error compares the ensemble-mean late-time curves in relative Euclidean norm. The figure band is the reference mean plus/minus 1.96 ensemble standard deviations, not a confidence interval for the mean.

## Table 2 and other ensemble plots

Pointwise Gaussian marginals are fitted using the predicted ensemble mean $\mu$ and sample standard deviation $s$. Coverage is the proportion of reference values inside $[\mu-zs,\mu+zs]$, averaged over realizations, space and time. The nominal 90%, 95%, and 99% diagnostics use $z=1.645,1.960,2.576$ respectively.

Gaussian marginal CRPS is evaluated as:

$$
q=\frac{y-\mu}{s+10^{-8}},\qquad
\operatorname{CRPS}=s\left[q(2\Phi(q)-1)+2\phi(q)-\pi^{-1/2}\right].
$$

The final value averages over all reference observations. The same paired ensemble supplies the fitted predictive moments and reference verification values; this is not an independent calibration split or a measure of training-induced epistemic uncertainty.

Cross sections use indices `int(0.25 * Nt)` and `int(0.75 * Nt)`, which are 16 and 48 on the 64-point endpoint-inclusive grid. The plotted times are approximately 0.254 and 0.762. The bands are plus/minus two ensemble standard deviations. The displayed realization is test index 20.

The second-order structure function uses terminal fields and nonwrapping increments:

$$
S_2(sh)=\operatorname{mean}_{b,i}(u_{b,i+s,-1}-u_{b,i,-1})^2,
\qquad s=1,\ldots,\lfloor N_x/3\rfloor.
$$

The linear reference and inverse-square spectral reference are visual comparisons, not fitted scaling-law estimates. The spectral plot uses the nominal axis $2\pi k/L$, whose period differs from the solver's $N_xh$ stencil period.

## Finer-grid experiment

One new coefficient vector and one initial condition at $\ell=0.1$ are sampled on a 128 × 128 grid after restoring the evaluation random state. The reference solver and S-FNO receive the same initial condition and forcing. No network retraining is performed. This is a single-realization diagnostic, not an ensemble convergence estimate.

The optional bilinear interpolation diagnostic uses coarse test index 0, which is a different realization from the finer-grid input. It is available only as an explicitly requested unpaired diagnostic and is excluded from the default comparisons and metric tables.

## Inverse experiment

Test index 10 supplies the known initial condition and the full space-time reference observation. The S-FNO parameters are frozen. With the same 256-coefficient forcing expansion, the objective is

$$
\mathcal J(z)=\operatorname{mean}
\left[\mathcal G_\theta(u_0,\xi(z))-u_{\mathrm{obs}}\right]^2
+5\times10^{-4}\frac{\|z\|_2^2}{256}.
$$

The initial coefficient vector is drawn after `torch.manual_seed(42)`. Adam uses learning rate 0.05, betas $(0.9,0.999)$ and epsilon $10^{-8}$ for 2,000 updates. StepLR advances after each update, with step size 200 and factor 0.5.

The inverse figures use fields evaluated just before the final optimizer update, stored as `last_evaluated`. The package also stores `after_final_update`, including separately reevaluated fields and losses, so coefficients, fields and objective values remain associated with their respective iterates. The two states are labeled separately in the saved fit summary.

The routine performs a regularized optimization. It makes no uniqueness, identifiability, physical-null-space, or probabilistic MAP guarantee.

## Timing

The optional benchmark uses the actual evaluated S-FNO and reference solver, the same test realization, and the same execution device. Inputs are transferred before timing. Model inference consumes the precomputed forcing field; the solver regenerates forcing from its coefficients, as its function normally does. Each wall-clock interval is bounded by CUDA synchronization when applicable. Model warmup is 20 calls, with 100 model measurements; the solver has one warmup and 10 measurements. The timings exclude data generation, checkpoint loading and host/device transfers and include complete output-field computation.

