# bayesian-finance-pymc

Bayesian financial modeling and time-series analysis with [PyMC](https://www.pymc.io/),
focused on asset returns, volatility estimation, and predictive inference.

The project walks through a complete workflow:
a frequentist baseline (OLS), a single-level Bayesian model, and a
hierarchical Bayesian model with sector-level priors, including diagnostics
and posterior predictive checks.

## Project overview

The notebook compares classical and Bayesian approaches for modeling daily
log-returns of a small, sector-diversified equity portfolio. The hypothesis
under test is that **volatility is structured by sector**, so a hierarchical
model with sector-level priors should describe returns better than a flat,
pooled model.

Approaches covered:

1. **Frequentist baseline** — OLS regression on average and per-stock prices,
   with residual diagnostics (Durbin-Watson, Jarque-Bera, etc.).
2. **Single-level Bayesian model** — Student-t likelihood with weakly-informative
   priors (`mu ~ Normal`, `sigma ~ HalfNormal`, `nu ~ Gamma`).
3. **Hierarchical Bayesian model** — sector-level priors on volatility, plus
   stock-specific intercept and time-slope drawn from group-level distributions.
4. **Sampler diagnostics** — trace plots, autocorrelation, ESS evolution, pair
   plots, and a discussion of MCMC vs. HMC/NUTS behaviour.
5. **Posterior predictive** — variance and volatility per predicted return,
   simulated price paths, and 2-sigma bands.

## Data

Daily closing prices are pulled live from Yahoo Finance via
[`yfinance`](https://pypi.org/project/yfinance/) inside the notebook. No
data is bundled in the repo.

| Ticker | Sector       |
|--------|--------------|
| GOOG   | Technology   |
| META   | Technology   |
| CAT    | Producers    |
| PEP    | Consumer     |
| KO     | Consumer     |

Default date range: `2024-01-01` to `2025-07-09`.

Returns are computed as `log(1 + pct_change)` and scaled by 100.

## Repository layout

```
.
├── financial_analysis_with_pymc.ipynb   # main notebook
├── archive/                             # snapshot of the original notebook
├── requirements.txt                     # pinned environment
├── LICENSE                              # MIT
└── README.md
```

## Requirements

- Python 3.11+ recommended
- The pinned environment is in `requirements.txt`. Key packages:
  `pymc`, `arviz`, `numpy`, `pandas`, `scipy`, `statsmodels`,
  `matplotlib`, `seaborn`, `xarray`, `yfinance`, `jupyter`.

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/bayesian-finance-pymc.git
cd bayesian-finance-pymc

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Graphviz is required for `pm.model_to_graphviz`. On macOS:

```bash
brew install graphviz
```

On Debian/Ubuntu:

```bash
sudo apt-get install graphviz
```

## Running the notebook

```bash
jupyter notebook financial_analysis_with_pymc.ipynb
```

Or open it in VS Code / JupyterLab and run cells top-to-bottom.

The first data-loading cell calls `yf.download(...)`, which needs network
access. If Yahoo Finance is unreachable, swap in any locally cached
DataFrame with the same shape.

Sampling time on a laptop ranges from a few seconds for the simple model to
a couple of minutes for the hierarchical model, depending on hardware.

## Notes

- `archive/` holds an unmodified snapshot of the notebook prior to the most
  recent cleanup. It is not required to run the analysis.
- All sampler settings, priors, and model structure in the working notebook
  are unchanged from the original analysis — only narrative text and
  comments were updated for clarity and consistency.

## License

Released under the [MIT License](LICENSE).
