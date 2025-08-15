import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns

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
def portfolio_var(returns_df, w, alpha=0.05):
    pr = returns_df @ w
    return -np.percentile(pr, alpha*100)

def as_series(w):
    return pd.Series(w).sort_index()

# target return grid (stay inside feasible range)
grid = np.linspace(mu.min()+1e-6, mu.max()-1e-6, 60)

mv_pts, cvar_pts, var_pts = [], [], []
common_pts = []   # intersections in weight space
w_tol = 1e-2

for r in grid:
    # Mean–Variance (volatility)
    try:
        ef_mv = EfficientFrontier(mu, S)
        ef_mv.efficient_return(r)
        w_mv = as_series(ef_mv.clean_weights())
        ret_mv, vol_mv, _ = ef_mv.portfolio_performance()
    except Exception:
        continue

    # CVaR
    try:
        ef_c = EfficientCVaR(mu, rets)
        ef_c.efficient_return(r)
        w_c = as_series(ef_c.clean_weights())
        ret_c, cvar_risk = ef_c.portfolio_performance()[:2]
    except Exception:
        continue

    # VaR (note: here we *evaluate* VaR for the MV-optimal weights;
    # true VaR-optimized frontier would require a custom optimizer)
    var_risk = portfolio_var(rets, w_mv, alpha=0.05)

    mv_pts.append((vol_mv, ret_mv, w_mv))
    cvar_pts.append((cvar_risk, ret_c, w_c))
    var_pts.append((var_risk, ret_mv, w_mv))

    # intersection in weight space (same portfolio under all criteria)
    if np.allclose(w_mv.values, w_c.reindex(w_mv.index).values, atol=w_tol):
        common_pts.append({
            "ret": float(ret_mv),
            "w": w_mv,
            "mv_risk": float(vol_mv),
            "cvar_risk": float(cvar_risk),
            "var_risk": float(var_risk),
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
plt.plot(*zip(*[(x,y) for x,y,_ in var_pts]), label="VaR (evaluated on MV weights)")

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
