# Neural SDEs for Financial Market Modeling

This repository contains the code and supporting material for my thesis project on **Neural Stochastic Differential Equations (Neural SDEs)** for financial time series generation.  
The project investigates whether a **Neural SDE-based GAN architecture (SDE-GAN)** can learn the underlying dynamics of financial markets and generate synthetic trajectories that are both statistically realistic and temporally coherent.

The model is trained on **daily S&P 500 data** and combines:
- a **Neural SDE generator** in the **Itô** formulation,
- a **CDE-based discriminator (critic)**,
- and a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** training framework.

The broader goal is not point forecasting, but learning a flexible stochastic process capable of reproducing key stylized properties of financial time series.

---

## Project overview

Traditional stochastic models in finance are powerful but often rely on restrictive assumptions about the data-generating process. This project explores a more flexible alternative by parameterizing the drift and diffusion of a stochastic differential equation with neural networks.

More specifically, the generator maps random noise into latent continuous-time dynamics through a Neural SDE, while the discriminator evaluates entire generated paths using a neural controlled differential equation. The framework is designed to model the **path-space distribution** of financial time series rather than only one-step transitions.

---

## Main objectives

The project is built around the following research questions:

- Can an **SDE-GAN** effectively learn the data-generating process of a major financial index?
- Can synthetic trajectories reproduce relevant **stylized facts** of returns, such as realistic marginal distributions, volatility patterns, and temporal dependence?
- Does the model generalize to unseen data rather than simply overfitting the training set?
- What practical challenges arise when training Neural SDEs in an adversarial setting?

---

## Dataset

The model is trained on **daily closing prices of the S&P 500 index**, covering the period from **January 1, 2014 to December 31, 2019**.  
The data source is **Investing.com**, and only trading days are included.

The raw closing prices are transformed into **log-returns**, which are used as the main training signal. In the implementation, the series is split into train and test sets without shuffling, scaled with a **RobustScaler**, and converted into fixed-length windows for sequence modeling.

---

## Model architecture

### Generator
The generator is implemented as an **Itô Neural SDE**. It consists of:
- an initial MLP mapping random initial noise to the latent initial state,
- a drift network,
- a diffusion network,
- a linear readout from latent space to data space.

The SDE is solved with `torchsde.sdeint_adjoint` using the **Euler** method. The implementation includes learnable scaling factors for drift and diffusion and uses **LipSwish** activations to encourage stable behavior.

### Discriminator / Critic
The discriminator is implemented as a **Neural CDE-based critic**.  
Generated and real trajectories are represented through interpolation coefficients, then processed by a controlled differential equation. The final score is computed from summary statistics of the hidden trajectory, specifically the hidden-state mean and standard deviation.

### Training
Training follows a **Wasserstein GAN with Gradient Penalty** setup:
- the critic is trained to assign higher scores to real paths than to fake ones,
- the generator is trained to maximize the critic’s score on generated paths,
- a gradient penalty is used to stabilize adversarial learning and enforce smoothness.

The training setup also uses monitoring of gradient norms and includes stochastic weight averaging in later training stages.

---

## Repository structure

```text
.
├── notebooks/
│   └── SDE_GAN_ito copy.ipynb
├── src/
│   └── SDE_GAN_utils_copy.py
├── thesis/
│   └── Thesis_PaolaPinna_284961-2 2.pdf
├── results/
├── README.md
└── requirements.txt
