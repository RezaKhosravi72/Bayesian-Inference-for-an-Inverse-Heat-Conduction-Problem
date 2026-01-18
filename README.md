# Bayesian Inference for an Inverse Heat Conduction Problem

This repository contains the submission for the *Bayesian Inference Challenge* on inverse heat conduction problems.

The objective is to infer an unknown Dirichlet boundary condition of a Laplace equation on the unit square using noisy interior temperature measurements, within a Bayesian framework.

---

## Repository Structure

```

challenge_submission/
├── inverse_heat_bayesian.pdf        # One-page PDF describing the modeling and results
├── inverse_heat_bayesian.ipynb      # Main Jupyter notebook (code, figures, inference)
├── inverse_heat_bayesian.pdf        # PDF export of the notebook
├── src/
│   └── helpers.py                   # Forward solver and utility functions
├── data/
│   └── processed_data.npz           # Processed measurement and grid data
├── prompts/
│   └── llm_prompt_history.txt       # Prompt history documenting LLM usage
└── README.md

```

---

## Method Overview

- **Forward model**  
  Laplace equation solved via finite differences with a spatially varying
  Dirichlet boundary condition parameterized using Gaussian radial basis functions.

- **Inverse problem formulation**  
  The unknown boundary coefficients are inferred from noisy interior measurements.

- **Bayesian framework**  
  - Gaussian likelihood with known noise level  
  - Gaussian prior enforcing smoothness and regularization  
  - Closed-form Gaussian posterior (linear inverse problem)

- **Inference output**  
  - Posterior mean boundary reconstruction  
  - Pointwise credible intervals  
  - Posterior covariance analysis for identifiability

---

## Reproducibility

The notebook is self-contained assuming the provided `measurements.npz` file is available.
All random sampling uses fixed seeds for reproducibility.

---

## LLM Usage Statement

Large Language Models were used as an auxiliary tool for:
- clarifying Bayesian modeling concepts,
- structuring the code and notebook layout,
- resolving implementation and numerical issues.

All modeling decisions, code execution, and result interpretation were performed and verified by the author.

---

## Author

*Reza Khosravi*

---