# portfolio_var_calculator.py
# Portfolio Value-at-Risk (VaR) Calculator
# Implements historical and parametric VaR at configurable confidence levels.

import numpy as np
from scipy.stats import norm


def historical_var(returns, confidence_level=0.95):
    """
    Compute Historical VaR at a given confidence level.
    returns: 1D array-like of portfolio returns (e.g., daily returns)
    confidence_level: e.g., 0.95 for 95% VaR
    Returns VaR as a positive number representing loss.
    """
    returns = np.asarray(returns)
    # VaR is the loss such that (1 - confidence) of returns are worse
    percentile = (1 - confidence_level) * 100
    var_return = -np.percentile(returns, percentile)
    return var_return


def parametric_var(returns, confidence_level=0.95):
    """
    Compute Parametric (Gaussian) VaR at a given confidence level.
    Assumes returns are normally distributed.
    """
    returns = np.asarray(returns)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)  # sample std
    z = norm.ppf(1 - confidence_level)  # negative for left tail
    # VaR in return terms: -(mu + z * sigma)
    var_return = -(mu + z * sigma)
    return var_return


def compute_portfolio_var(returns, portfolio_value, confidence_levels=(0.95, 0.99)):
    """
    Compute historical and parametric VaR for multiple confidence levels.
    returns: 1D array-like of portfolio returns
    portfolio_value: current portfolio value in dollars
    confidence_levels: tuple of confidence levels to evaluate
    """
    results = []
    for conf in confidence_levels:
        hist_var_ret = historical_var(returns, conf)
        param_var_ret = parametric_var(returns, conf)

        hist_var_dollar = hist_var_ret * portfolio_value
        param_var_dollar = param_var_ret * portfolio_value

        results.append({
            "confidence": conf,
            "historical_var_return": hist_var_ret,
            "parametric_var_return": param_var_ret,
            "historical_var_dollar": hist_var_dollar,
            "parametric_var_dollar": param_var_dollar,
        })
    return results


if __name__ == "__main__":
    # Example: simulate daily returns for a simple multi-asset portfolio
    np.random.seed(42)
    n_days = 500

    # Simulate returns for 3 assets
    asset1 = np.random.normal(loc=0.0005, scale=0.015, size=n_days)
    asset2 = np.random.normal(loc=0.0003, scale=0.02, size=n_days)
    asset3 = np.random.normal(loc=0.0004, scale=0.012, size=n_days)

    # Equal-weight portfolio
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    portfolio_returns = (
            weights[0] * asset1 +
            weights[1] * asset2 +
            weights[2] * asset3
    )

    portfolio_value = 1_000_000  # $1M portfolio

    results = compute_portfolio_var(
        portfolio_returns,
        portfolio_value,
        confidence_levels=(0.95, 0.99)
    )

    print("Portfolio Value-at-Risk (VaR) Results")
    print(f"Portfolio value: ${portfolio_value:,.0f}")
    print("-" * 50)

    for r in results:
        conf = r["confidence"]
        print(f"Confidence level: {int(conf * 100)}%")
        print(
            f"  Historical VaR (return):  {r['historical_var_return']:.4f}  ({r['historical_var_return'] * 100:.2f}%)")
        print(
            f"  Parametric VaR (return):  {r['parametric_var_return']:.4f}  ({r['parametric_var_return'] * 100:.2f}%)")
        print(f"  Historical VaR (dollar):  ${r['historical_var_dollar']:,.2f}")
        print(f"  Parametric VaR (dollar):  ${r['parametric_var_dollar']:,.2f}")
        print("-" * 50)