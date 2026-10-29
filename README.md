# Neural SDEs for Financial Market Modeling

This repository contains the code and supporting material for my thesis project on **Neural Stochastic Differential Equations (Neural SDEs)** for financial time-series generation.

The project investigates whether a **Neural SDE-based GAN architecture (SDE-GAN)** can learn the underlying dynamics of financial markets and generate synthetic trajectories that are statistically realistic and temporally coherent.

The model is trained on **daily S&P 500 data** and combines:

- a **Neural SDE generator** in the **Itô** formulation,
- a **Neural CDE-based discriminator / critic**,
- and a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** training framework.

The goal is not point forecasting. Instead, the model learns a flexible stochastic process intended to reproduce the **distribution of entire financial trajectories** and relevant properties of the observed series.

---

## Project overview

Traditional stochastic models in finance are powerful, but often rely on explicit assumptions about the structure of the underlying data-generating process. This project explores a more flexible alternative by parameterizing the drift and diffusion of a stochastic differential equation with neural networks.

The generator maps random noise into latent continuous-time dynamics through a Neural SDE. The critic evaluates complete trajectories using a Neural Controlled Differential Equation (Neural CDE). Training is adversarial and uses the Wasserstein objective with a gradient penalty for stabilization.

The resulting framework models the **path-space distribution** of the financial time series rather than only individual observations or one-step transitions.

---

## Main objectives

The project is built around the following questions:

- Can an **SDE-GAN** learn the data-generating dynamics of a major financial index?
- Can generated trajectories reproduce important **stylized properties** of financial returns?
- Can the model capture distributional and temporal characteristics rather than simply memorizing the training data?
- Does the learned process generalize to unseen observations?
- What practical difficulties arise when Neural SDEs are trained in an adversarial framework?

---

## Dataset and preprocessing

The model is trained on **daily closing prices of the S&P 500 index**, covering the period from **January 1, 2014 to December 31, 2019**.

The data source used in the thesis is **Investing.com**, and only trading days are considered.

The financial series is transformed before training and processed chronologically to preserve its temporal structure. The implementation includes:

- chronological train/test splitting without shuffling,
- robust scaling with `RobustScaler`,
- fixed-length sliding windows for sequence modeling,
- conversion of trajectories to continuous-time interpolation coefficients for the CDE critic.

The utility code uses a default sequence length of **50 observations** for the sliding-window dataset.

---

## Model architecture

### Neural SDE generator

The generator is implemented as an **Itô Neural SDE**.

Its main components are:

- an initial MLP mapping random initial noise to a latent initial state,
- a neural drift function,
- a neural diffusion function,
- learnable scaling parameters for drift and diffusion,
- a linear readout mapping the latent process back to the observed data space.

Both drift and diffusion networks use **LipSwish** activations. The SDE is solved with:

```text
torchsde.sdeint_adjoint
```

using the **Euler method**.

The generator therefore defines a stochastic continuous-time latent process of the form

```text
dX_t = f_θ(t, X_t) dt + g_θ(t, X_t) dW_t
```

where both the drift `f_θ` and diffusion `g_θ` are learned neural networks.

### Neural CDE critic

The discriminator is implemented as a **Neural CDE-based critic**.

Real and generated sequences are converted into continuous paths through linear interpolation. The critic then processes those paths with a controlled differential equation.

In the implementation:

- the CDE is solved with **RK4**,
- the hidden trajectory is summarized using its mean and standard deviation,
- the resulting representation is mapped to a scalar critic score.

This allows the discriminator to evaluate complete trajectories instead of treating observations independently.

### WGAN-GP training

Training follows the **Wasserstein GAN with Gradient Penalty** framework.

The critic learns to assign higher scores to real trajectories than to generated ones, while the generator is trained to increase the critic score of synthetic paths.

A gradient penalty is computed on interpolations between real and generated trajectories to encourage the Lipschitz constraint required by the Wasserstein objective.

The implementation also monitors gradient behavior during training and uses **Stochastic Weight Averaging (SWA)** in the later stages of optimization.

---

## Training and implementation details

Some of the main implementation choices are:

| Component | Choice |
| --- | --- |
| SDE formulation | Itô |
| Generator solver | Euler |
| CDE critic solver | RK4 |
| Generator library | `torchsde` |
| Critic library | `torchcde` |
| Adversarial objective | WGAN-GP |
| Activation | LipSwish |
| Preprocessing | RobustScaler |
| Sequence construction | Sliding windows |
| Model stabilization | Gradient penalty + SWA |

The implementation separates the reusable model components from the experimental notebook so that the Neural SDE generator, CDE critic, preprocessing pipeline and gradient-penalty logic can be inspected independently.

---

## Evaluation

The SDE-GAN is evaluated by comparing real and generated financial trajectories at both **distributional** and **temporal** levels rather than relying on a single forecasting metric.

The analysis includes:

- Maximum Mean Discrepancy (**MMD²**),
- Kolmogorov-Smirnov (**KS**) statistics,
- Wasserstein distance,
- empirical moments and volatility,
- kernel density estimates,
- autocorrelation functions (**ACF**) and ACF mean-squared error,
- qualitative comparison of generated and real paths.

Evaluation is performed separately on training and unseen test data to assess both fidelity and generalization.

---

## Results

The experiments show that the Neural SDE-GAN is able to generate **diverse, non-collapsed financial trajectories** whose return distributions remain close to those observed in the real S&P 500 series.

### Quantitative summary

| Metric | Training data | Test data |
| --- | ---: | ---: |
| MMD² | ≈ **0.0020** | ≈ **0.0067** |
| Mean KS statistic | ≈ **0.051** | — |
| Mean Wasserstein distance | ≈ **0.12** | — |
| Generated volatility (σ) | close to empirical scale | ≈ **0.70–1.01** |
| Real volatility (σ) | — | ≈ **1.34–1.36** |

The low training-set MMD², KS and Wasserstein values indicate good agreement between the empirical and generated return distributions. On unseen data, the MMD² remains small, showing that the learned stochastic process retains meaningful distributional structure beyond the training sample.

### Temporal behavior

The generated series also achieve **low ACF mean-squared error**, indicating that short-range temporal dependence is reproduced reasonably well. Qualitative trajectory inspection did not reveal mode or trajectory collapse: the generator produces visibly different stochastic paths rather than repeatedly reproducing a small set of patterns.

### Generalization and limitations

The main difference on the test period is a tendency to **underestimate volatility**, particularly during more turbulent market conditions. Generated test trajectories show standard deviations of roughly **0.70–1.01**, compared with about **1.34–1.36** for the corresponding real data.

Overall, the model captures a substantial part of the **distributional shape, temporal structure and path variability** of the financial returns while remaining less accurate for extreme volatility regimes. This is consistent with the goal of the project: learning a realistic stochastic trajectory distribution rather than producing deterministic point forecasts.

The notebook contains the full experimental workflow, plots and diagnostics, while the thesis provides the complete interpretation and discussion of these results.

---

## Repository structure

```text
.
├── SDE_GAN_ito copy.ipynb
│   └── Main experimental notebook: financial-data loading and preprocessing,
│       chronological train/test preparation, SDE-GAN training,
│       WGAN-GP optimization, SWA, synthetic trajectory generation,
│       diagnostics, plots and evaluation of real vs generated series.
│
├── SDE_GAN_utils_copy.py
│   └── Core reusable components: LipSwish activation, MLP blocks,
│       Itô Neural SDE generator, Neural CDE critic, RobustScaler-based
│       preprocessing, sliding-window dataset construction and
│       WGAN gradient-penalty implementation.
│
├── swa_generator.pt
│   └── Saved generator checkpoint obtained using Stochastic Weight Averaging.
│
├── Thesis_PaolaPinna_284961-2 2.pdf
│   └── Full thesis describing the theoretical background, methodology,
│       implementation, experiments, results and discussion.
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/paolapinna03/neural-sde-gan-finance.git
cd neural-sde-gan-finance
```

### 2. Install the main dependencies

The project requires Python and the main scientific / deep-learning libraries used by the notebook and utility module:

```bash
pip install numpy pandas torch scikit-learn torchsde torchcde torchdiffeq matplotlib
```

Depending on the environment used for the notebook, additional visualization or notebook packages may also be required.

### 3. Open the main notebook

Run:

```text
SDE_GAN_ito copy.ipynb
```

The notebook contains the end-to-end experimental pipeline. Dataset paths may need to be adapted to the local environment before execution.

### 4. Reuse the model components

The reusable implementations of the generator, critic and preprocessing utilities are available in:

```text
SDE_GAN_utils_copy.py
```

---

## Trained checkpoint

The repository includes:

```text
swa_generator.pt
```

This file stores the **Stochastic Weight Averaged generator checkpoint** produced during training and can be used to restore the trained generator without retraining the complete adversarial model from scratch.

---

## Main technologies

- Python
- PyTorch
- `torchsde`
- `torchcde`
- `torchdiffeq`
- NumPy
- pandas
- scikit-learn
- Neural Stochastic Differential Equations
- Neural Controlled Differential Equations
- Wasserstein GAN with Gradient Penalty
- Stochastic Weight Averaging

---

## Thesis

The complete theoretical background, methodology, experiments and discussion are available in:

[`Thesis_PaolaPinna_284961-2 2.pdf`](./Thesis_PaolaPinna_284961-2%202.pdf)

---

## License

See [`LICENSE`](./LICENSE).
