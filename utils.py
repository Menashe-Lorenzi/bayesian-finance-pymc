"""Helper functions for the financial analysis notebook.

Each function here is a verbatim extraction of the corresponding logic from
the original notebook - no math, priors, or numerical operations have been
changed. The goal is purely to keep the notebook readable.
"""
from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf


def load_prices_and_returns(tickers, start_date, end_date):
    """Download closing prices and compute log-returns scaled by 100.

    Returns: prices, returns, log_returns, scaled_returns, y_data
    where y_data is the dropna'd numpy array used as the observed variable.
    """
    prices = yf.download(tickers, start=start_date, end=end_date)["Close"]
    prices = prices[tickers]
    returns = prices.pct_change()
    log_returns = np.log(returns + 1)
    scaled_returns = log_returns * 100
    y_data = scaled_returns.dropna().values
    return prices, returns, log_returns, scaled_returns, y_data


def describe_volatility_and_correlations(prices, log_returns):
    """Print volatility, average prices, and plot the correlation heatmap."""
    print("--- Daily Volatility (Standard Deviation %) ---")
    volatility = (log_returns * 100).std()
    print(volatility.sort_values(ascending=False).to_string())
    print(f"\nAverage Portfolio Volatility: {volatility.mean():.4f}%")

    print("\n--- Average Stock Prices ($) ---")
    print(prices.mean().sort_values(ascending=False).to_string())

    print("\n--- Correlation Matrix ---")
    corr_matrix = (log_returns * 100).corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Stock Return Correlations")
    plt.show()
    return corr_matrix


def fit_ols_with_ci(
    time,
    y,
    label,
    line_color="black",
    fill_color="gray",
    figsize=(10, 5),
):
    """Fit an OLS regression of y on time and plot the line with a 95% CI band.

    Returns the fitted statsmodels OLS results.
    """
    X = sm.add_constant(time)
    model = sm.OLS(y, X).fit()

    pred = model.get_prediction(X).summary_frame(alpha=0.05)
    mean = pred["mean"]
    lower = pred["mean_ci_lower"]
    upper = pred["mean_ci_upper"]

    plt.figure(figsize=figsize)
    plt.plot(time, y, label=label, alpha=0.6)
    plt.plot(time, mean, label="Linear Fit", color=line_color, linestyle="--")
    plt.fill_between(time, lower, upper, color=fill_color, alpha=0.3, label="95% CI")
    plt.title(f"Regression of {label} Over Time")
    plt.xlabel("Time (days)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Regression for {label}:")
    print(model.summary())
    return model


def plot_regression_with_intervals(trace, ppc, data, n_grid=200):
    """Plot the Bayesian regression line with the 95% HDI of the mean and the
    95% predictive interval (extracted from the posterior predictive draws).
    """
    x_pred_grid = np.linspace(data["x"].min(), data["x"].max(), n_grid)

    y_draws = (
        ppc.posterior_predictive["y"]
        .stack(sample=("chain", "draw"))
        .values
    )

    alpha_samples = trace.posterior["intcpt"].stack(samples=("chain", "draw")).values
    beta_samples = trace.posterior["slope"].stack(samples=("chain", "draw")).values
    mu_pred_on_grid = np.array(
        [a + b * x_pred_grid for a, b in zip(alpha_samples, beta_samples)]
    )

    mean_line = mu_pred_on_grid.mean(axis=0)
    ci_low, ci_high = az.hdi(mu_pred_on_grid, hdi_prob=0.95).T
    pi_low, pi_high = az.hdi(y_draws.T, hdi_prob=0.95).T

    plt.figure(figsize=(10, 7))
    plt.fill_between(
        data["x"], pi_low, pi_high,
        alpha=0.20, color="skyblue", label="95% Predictive Interval",
    )
    plt.fill_between(
        x_pred_grid, ci_low, ci_high,
        alpha=0.45, color="dodgerblue", label="95% HDI of the Mean",
    )
    plt.plot(x_pred_grid, mean_line, lw=3, color="darkblue", label="Posterior Mean Line")
    plt.scatter(data["x"], data["y"], s=30, color="black", alpha=0.7, label="Data")
    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Average Return", fontsize=12)
    plt.title("Bayesian Regression: Mean, Confidence, and Predictive Intervals", fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def compute_predicted_volatility(ppc, num_days):
    """From a posterior predictive object, compute per-Y variance/volatility
    and a random subsample of `num_days` returns drawn (without replacement)
    from the flattened posterior predictive of `y`.

    Returns: random_samples, variances_per_y, volatility_per_y, results_df
    """
    posterior_y = ppc.posterior_predictive["y"]
    variances_per_y = posterior_y.var(dim=("chain", "draw"))

    all_samples = posterior_y.values.flatten()
    random_samples = np.random.choice(all_samples, size=num_days, replace=False)

    volatility_per_y = np.sqrt(variances_per_y.values)

    results_df = pd.DataFrame(
        {
            "postrior_y": random_samples,
            "predicted_variance": variances_per_y.values,
            "predicted_volatility": volatility_per_y,
        }
    )
    return random_samples, variances_per_y, volatility_per_y, results_df


def simulate_price_paths(p0, samples_pct, volatility_per_y):
    """Reconstruct an expected price path from percent returns, plus 2-sigma bands."""
    returns = 1 + (samples_pct / 100)
    cumulative_returns = np.cumprod(returns)
    price_paths = p0 * cumulative_returns

    upper_band = price_paths * ((100 + 2 * volatility_per_y) / 100)
    lower_band = price_paths * ((100 - 2 * volatility_per_y) / 100)
    return price_paths, upper_band, lower_band


def plot_simulated_paths(price_paths, lower_band, upper_band, real_prices):
    """Overlay the reconstructed price path and 2-sigma band on the actual mean price."""
    t = np.arange(len(price_paths))
    plt.figure(figsize=(14, 6))
    plt.plot(t, price_paths, label="Expected Price", color="blue")
    plt.fill_between(t, lower_band, upper_band, alpha=0.3, color="red", label="±2σ band")
    plt.plot(np.arange(len(real_prices)), real_prices,
             label="Actual Price", color="black", linestyle="--")
    plt.title("Expected Price Reconstructed from Predicted Returns")
    plt.xlabel("Time (days)")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
