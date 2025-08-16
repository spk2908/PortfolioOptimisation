import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize

# Load & prep
sbin = pd.read_csv("data/SBIN.NS2000-01-012024-01-01.csv", parse_dates=["Date"])
hdfc = pd.read_csv("data/HDFCBANK.NS2000-01-012024-01-01.csv", parse_dates=["Date"])
df = (sbin[["Date","Close"]].rename(columns={"Close":"SBIN"})
      .merge(hdfc[["Date","Close"]].rename(columns={"Close":"HDFCBANK"}), on="Date")
      .set_index("Date"))
rets = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)   # uses prices internally
S  = risk_models.sample_cov(df)

# helpers
def portfolio_var(w, returns_df, alpha=0.05):
    """Calculates the portfolio VaR"""
    pr = returns_df @ w
    return -np.percentile(pr, alpha*100)

def portfolio_return(w, mu):
    """Calculates the portfolio's expected return"""
    return w @ mu

def as_series(w, index):
    """Converts a numpy array of weights to a pandas Series"""
    return pd.Series(w, index=index).sort_index()

# target return grid (stay inside feasible range)
grid = np.linspace(mu.min()+1e-6, mu.max()-1e-6, 100)

mv_pts, cvar_pts, var_pts = [], [], []
common_pts = []   # intersections in weight space
w_tol = 1e-2
num_assets = len(mu)
bounds = tuple((0, 1) for asset in range(num_assets))
initial_guess = np.array(num_assets * [1. / num_assets,])

for r in grid:
    # Mean–Variance (volatility)
    try:
        ef_mv = EfficientFrontier(mu, S)
        ef_mv.efficient_return(r)
        w_mv = as_series(ef_mv.clean_weights(), mu.index)
        ret_mv, vol_mv, _ = ef_mv.portfolio_performance()
    except Exception:
        continue

    # CVaR
    try:
        ef_c = EfficientCVaR(mu, rets)
        ef_c.efficient_return(r)
        w_c = as_series(ef_c.clean_weights(), mu.index)
        ret_c, cvar_risk = ef_c.portfolio_performance()[:2]
    except Exception:
        continue

    # VaR Optimization (independent calculation)
    try:
        # Constraints:
        # 1. The portfolio's expected return should be equal to the target return `r`.
        # 2. The sum of weights must be 1.
        constraints = (
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - r},
            {'type': 'eq', 'fun': 'np.sum'}
        )
        
        # Optimization
        result = minimize(
            portfolio_var,
            initial_guess,
            args=(rets,),
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, mu) - r}
            ]
        )
        
        if result.success:
            w_var = as_series(result.x, mu.index)
            var_risk = portfolio_var(w_var, rets)
            ret_var = portfolio_return(w_var, mu)
            var_pts.append((var_risk, ret_var, w_var))
        
    except Exception:
        continue

    mv_pts.append((vol_mv, ret_mv, w_mv))
    cvar_pts.append((cvar_risk, ret_c, w_c))

    # intersection in weight space (same portfolio under all criteria)
    if 'w_var' in locals() and np.allclose(w_mv.values, w_c.reindex(w_mv.index).values, atol=w_tol) and np.allclose(w_mv.values, w_var.reindex(w_mv.index).values, atol=w_tol):
        common_pts.append({
            "ret": float(ret_mv),
            "w": w_mv,
            "mv_risk": float(vol_mv),
            "cvar_risk": float(cvar_risk),
            "var_risk": float(portfolio_var(w_mv, rets)), # Re-calculate VaR for the common weight
        })

# report
if common_pts:
    for i, p in enumerate(common_pts, 1):
        print(f"\nCommon-optimal portfolio {i}")
        print(f"  Expected return: {p['ret']:.6f}")
        print(f"  Weights:\n{p['w']}")
        print(f"  Volatility: {p['mv_risk']:.6f} | CVaR: {p['cvar_risk']:.6f} | VaR: {p['var_risk']:.6f}")
else:
    print("No common-optimal portfolios found within tolerance.")

# plot
plt.figure(figsize=(10,6))
# curves
plt.plot(*zip(*[(x,y) for x,y,_ in mv_pts]), label="Mean-Variance (Volatility)")
plt.plot(*zip(*[(x,y) for x,y,_ in cvar_pts]), label="CVaR")
plt.plot(*zip(*[(x,y) for x,y,_ in var_pts]), label="Mean-VaR")

# mark common portfolios on EACH curve at their own risk values
if common_pts:
    plt.scatter([p["mv_risk"] for p in common_pts],  [p["ret"] for p in common_pts],  marker="x", s=80, label="Common weight on MV")
    plt.scatter([p["cvar_risk"] for p in common_pts],[p["ret"] for p in common_pts],  marker="o", s=40, label="Common weight on CVaR")
    plt.scatter([p["var_risk"] for p in common_pts], [p["ret"] for p in common_pts],  marker="s", s=40, label="Common weight on VaR")

plt.xlabel("Risk (metric depends on curve)")
plt.ylabel("Expected Return")
plt.title("Frontiers with common-weight portfolios marked")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
