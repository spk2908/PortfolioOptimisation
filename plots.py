import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize

# Load & prep
cipla = pd.read_csv("data/CIPLA.NS2000-01-012024-01-01.csv", parse_dates=["Date"])
lnt = pd.read_csv("data/LNT2000-01-012024-01-01.csv", parse_dates=["Date"])
df = (cipla[["Date","Close"]].rename(columns={"Close":"CIPLA"})
      .merge(lnt[["Date","Close"]].rename(columns={"Close":"LNT"}), on="Date")
      .set_index("Date"))
rets = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)   # uses prices internally
S  = risk_models.sample_cov(df)

# helpers
def portfolio_var(w, returns_df, alpha=0.05):
    """Calculates the portfolio VaR"""
    pr = returns_df @ w
    # The VaR is the negative of the percentile, representing a loss
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
        constraints = (
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - r},
            {'type': 'eq', 'fun': 'np.sum'}
        )
        
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
            var_risk_val = portfolio_var(w_var, rets)
            ret_var = portfolio_return(w_var, mu)
            var_pts.append((var_risk_val, ret_var, w_var))
        
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
            "var_risk": float(portfolio_var(w_mv, rets)),
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

# plot frontiers
plt.figure(figsize=(10,6))
plt.plot(*zip(*[(x,y) for x,y,_ in mv_pts]), label="Mean-Variance (Volatility)")
plt.plot(*zip(*[(x,y) for x,y,_ in cvar_pts]), label="CVaR")
plt.plot(*zip(*[(x,y) for x,y,_ in var_pts]), label="Mean-VaR")

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


#---------------------------------------------------------------------#
# NEW: PLOT PORTFOLIO VALUE EVOLUTION FOR EXTREME RETURN PORTFOLIOS   #
#---------------------------------------------------------------------#

if mv_pts:
    # Sort the calculated portfolios by their expected return
    sorted_portfolios = sorted(mv_pts, key=lambda p: p[1]) # p[1] is the return

    # Select the two with the lowest and two with the highest returns
    portfolios_to_plot = [
        ("Lowest Return #1", sorted_portfolios[0]),
        ("Lowest Return #2", sorted_portfolios[1]),
        ("Highest Return #2", sorted_portfolios[-2]),
        ("Highest Return #1", sorted_portfolios[-1]),
    ]

    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Portfolio Value Evolution Over Time", fontsize=16)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, (title, portfolio_data) in enumerate(portfolios_to_plot):
        ax = axes[i]
        vol, ret, w = portfolio_data

        # Calculate the daily returns of this specific portfolio
        portfolio_daily_returns = rets.dot(w)

        # Calculate the cumulative growth of a $1 investment
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()

        # Plot the evolution
        cumulative_returns.plot(ax=ax)
        
        # Format the plot
        ax.set_title(f"{title}\nExp. Return: {ret:.4f} | Volatility: {vol:.4f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (Initial $1)")
        ax.grid(True)
        
        # Display weights on the chart
        weights_str = "\n".join([f"{idx}: {val:.2%}" for idx, val in w.items()])
        ax.text(0.05, 0.95, weights_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

else:
    print("No portfolios were calculated, cannot plot evolution graphs.")