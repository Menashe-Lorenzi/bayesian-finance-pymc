# bayesian-finance-pymc

Bayesian financial modeling and time-series analysis with [PyMC](https://www.pymc.io/),
focused on asset returns, volatility estimation, and predictive inference.

The project walks through a focused, end-to-end workflow: an OLS baseline,
three Bayesian models of increasing structure, sampler-geometry diagnostics
(centered → non-centered reparameterization), and a posterior-predictive
price-path simulation.

## Project overview

The notebook compares classical and Bayesian approaches for modeling daily
log-returns of a small, sector-diversified equity portfolio. The hypothesis
under test is that **volatility is structured by sector**, so a hierarchical
model with sector-level priors should describe returns better than a flat,
pooled model.

Sections in the notebook:

1. **Setup & Data** — load tickers, fetch prices, compute log-returns.
2. **Frequentist baseline** — OLS regression on average and per-stock prices,
   with residual diagnostics (Durbin–Watson, Jarque–Bera, etc.).
3. **Bayesian Model — Sector-Grouped Volatility** — Student-t likelihood
   with sector-level `HalfCauchy` priors on `sigma`.
4. **Bayesian Regression in Time** — single-level Student-t regression of
   the average return on time; sets up the posterior-predictive machinery.
5. **Hierarchical Bayesian Model** — per-stock intercept and time-slope
   drawn from group-level priors. Fit first in centered form, then
   reparameterized to non-centered to fix the funnel-shaped posterior
   geometry that NUTS struggles with.
6. **Posterior Predictive — Price Simulation** — variance and volatility
   per predicted return, reconstructed price paths, and 2σ bands.

Long, repetitive code blocks (data loading, EDA, OLS plotting, regression
intervals, price simulation) live in `utils.py` so the notebook stays
narrative-first.

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
├── utils.py                             # extracted helpers used by the notebook
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
