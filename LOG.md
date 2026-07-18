## System Architecture

The controlled block-LTI system:

```
x(t+1) = A x(t) + B u(t) + w(t)      w ~ N(0, Q)   process noise
y(t)   = C x(t)        + v(t)        v ~ N(0, R)   measurement noise
```

`A`, `B` are block matrices over the `N` clients. Diagonal blocks (`A_mm`, `B_mm`) and the
block-diagonal output map `C_mm` are known locally; off-diagonal blocks (`A_mn`, `B_mn`) are
the unknown inter-component coupling, estimated federally.

Three blocks run in sequence:

- **Block 1 — Data.** A synthetic generator produces block system matrices (stable `A`,
  persistently-exciting controls `u`) and observation time series, partitions them by client,
  and splits into train/test.
- **Block 2 — Federated Training.** Each client runs a local Kalman filter (DKF) that ignores
  cross-coupling, then a Virtual Feedback Learning (VFL) correction with an observation gain
  `θ_m` and a state-bias `φ_m`. A global learner estimates `A_mn`, `B_mn` by gradient descent
  and returns state gradients to the clients.
- **Block 3 — Counterfactual Reasoning.** Given a factual trajectory: **abduction** infers the
  client latent states, an **intervention** perturbs a chosen client's control (`do(u += δ)`),
  and a **rollout** propagates the learned model forward on the *same* noise realization; the
  predicted counterfactual is scored against the true counterfactual and baselines.

---

## Project Structure

Current state (grows as modules are added):

```
counterfactual_new_exp/
├── CLAUDE.md            # coding standards (local only, gitignored)
├── LOG.md              # this file: architecture + implementation log
├── requirements.txt    # python dependencies
├── generate_data.py    # entry point: data generation
├── configs/
│   └── data_linear.yaml       # 6-client controlled-LTI generation config
├── core/
│   └── KalmanFilter.py        # controlled discrete-time linear Kalman filter (DKF)
├── data/
│   ├── synthetic_lti.py       # controlled-LTI generator (+ per-client DKF precompute)
│   └── datasets/              # generated data (gitignored)
├── tests/
│   ├── test_data_generation.py   # data-generation validation (12 tests)
│   └── filter_comparison.py      # CKF vs DKF one-step-prediction comparison + plot
└── legacy/             # previous codebase — reference only (gitignored)
```

---

## Implementation Log

### Session 1 — 2026-07-15
- Reviewed the legacy counterfactual pipeline in `legacy/synthetic_send/` (authoritative
  training path: `main1.py` → `global_learner1.py` + `local_learner1.py` +
  `data_retriever.py`; counterfactual experiment: `counterfactual_exp.py`; control-theoretic
  analysis: `transfer_function.py`, `b_block_error.py`).
- Reviewed the sibling `FedGC-uncert-prop` project as the structural/style template, and
  identified what this project adds: controlled dynamics (`B`, `u`), per-client state-bias
  `φ`, federated estimation of `B_mn`, and the counterfactual reasoning + analysis layers.
- Wrote foundational docs: `CLAUDE.md` (coding standards with control/CF notation), `LOG.md`
  (this file), `requirements.txt`.
- Agreed build plan: mirror the uncert file structure (YAML configs, dataclasses, integer
  tuple keys `(m,n)`); build **data first** (6-client controlled LTI, like the uncert N6
  system), validate the data, then training, then counterfactual and analysis layers —
  testing each step.

### Session 2 — 2026-07-16/17 (Block 1: data generation + filter comparison)
- Added `core/KalmanFilter.py`: controlled discrete-time linear Kalman filter (predict with
  B·u using the previous control u_{t-1}; measurement update; `run()` over a column-major
  trajectory; x0=zeros/P0=I init). Used as the per-client DKF at data-generation time and
  (later) at training time.
- Added `data/synthetic_lti.py`: generates the 6-client controlled block-LTI system — stable
  A (spectral radius scaled to target), block B with configurable off-diagonal coupling
  (dense/shared/custom/none), block-diagonal C/Q/R; drives it with per-split controls;
  simulates train+test splits (shared true matrices, independent controls/noise/x0);
  precomputes each client's DKF; saves global + per-component CSVs + dataset_params.json.
  Includes `check_persistence_of_excitation`.
- Per-split `controls` config with two types: `gaussian` (u ~ N(mu, sigma²) every step, PE)
  and `mean_shift` (u=0 outside windows; inside each window the targeted components get
  u ~ N(mu, sigma²); writes shift_labels.csv). Train uses gaussian; test uses mean_shift
  (single-source / multi-source / global windows).
- Added `configs/data_linear.yaml`: N=6, P=2/D=8/S=2 per client, 19 A-edges, ρ_target=0.6,
  B coupling=dense, Q=R=0.001.
- Added `generate_data.py` entry point.
- Added `tests/test_data_generation.py`: 12 checks (shapes, A stability+structure, B dense,
  C/Q/R structure, PE on both splits, per-component↔global, DKF-precompute reproducibility,
  dynamics residual ~ N(0,Q), shared matrices) — all passing.
- Added `tests/filter_comparison.py`: scores CKF (oracle, full A/B/C) vs DKF (local, diagonal
  blocks only) one-step predictions vs X_complete; prints RMSE tables (incl. in/out-window
  residuals when shift_labels present) and saves a shaded trajectory/error plot. DKF is
  ~28–39× worse than CKF on the PE train regime; on the mean-shift test the DKF residual
  jumps ~10–20× inside windows (blind to the dense B_mn control coupling) while the CKF stays
  flat — the gap the VFL correction must close.
- CLAUDE.md rule 4 clarified: on-disk trajectory CSVs are row-major (T, dim); in-memory
  algorithm arrays are column-major (dim, T).
