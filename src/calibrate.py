# calibrate.py
# Fits two models to the real Deribit option prices for one expiry:
#   1. Black-Scholes: ONE volatility for every strike
#   2. Merton jump-diffusion: everyday volatility sigma, plus jumps
#      (lambda jumps a year on average, each of mean size m and spread delta)
# and asks which one matches the market better.
#
# NOTE: Deribit gives me the forward price F for the expiry, so I price on the forward with
# no interest rate (the "Black-76" version of Black-Scholes). Over a few weeks the effect of
# interest is tiny, and using F avoids having to guess a crypto interest rate.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

df = pd.read_csv("data/deribit_expiry.csv")
F = df["forward"].median()
T = df["T"].iloc[0]
K = df["strike"].values
is_put = (df["type"] == "P").values
market = df["price_usd"].values     # the out-of-the-money option's own price (put or call)


def bs_call(F, K, sigma, T, r=0.0):
    # Black-Scholes call on a forward; r lets Merton's formula reuse it
    d1 = (np.log(F / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return F * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def merton_call(F, K, sigma, lam, m, delta, T):
    # same formula as my C++ merton_call: Black-Scholes prices for n = 0, 1, 2, ... jumps,
    # weighted by the Poisson probability of n jumps
    k = np.exp(m + 0.5 * delta**2) - 1
    lam2 = lam * (1 + k)
    price = np.zeros_like(K, dtype=float)
    weight = np.exp(-lam2 * T)
    for n in range(60):
        sigma_n = np.sqrt(sigma**2 + n * delta**2 / T)
        r_n = -lam * k + n * np.log(1 + k) / T
        price += weight * bs_call(F, K, sigma_n, T, r_n)
        weight *= lam2 * T / (n + 1)
    return price


def otm_price(call_price):
    # turn call prices into the out-of-the-money option's price: puts below F, calls above
    # NOTE: put-call parity on a forward with no interest:  P = C - F + K
    return np.where(is_put, call_price - F + K, call_price)


def implied_vol(price, F, K, T):
    # bisection on every strike at once: the volatility that makes Black-Scholes match price
    # NOTE: I invert the OUT-OF-THE-MONEY price. A deep in-the-money call is almost all
    # intrinsic value (F - K), so its price barely depends on volatility and inverting it
    # would turn tiny price errors into huge volatility errors.
    lo, hi = np.full_like(K, 0.01, dtype=float), np.full_like(K, 5.0, dtype=float)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        too_low = otm_price(bs_call(F, K, mid, T)) < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


market_iv = implied_vol(market, F, K, T)

# ---------- fit 1: Black-Scholes, one volatility ----------
# I fit in implied-volatility terms, so cheap far-out options count as much as expensive ones
bs_fit = minimize(lambda p: np.mean((implied_vol(otm_price(bs_call(F, K, p[0], T)), F, K, T) - market_iv) ** 2),
                  x0=[0.5], bounds=[(0.05, 3.0)])
sigma_bs = bs_fit.x[0]

# ---------- fit 2: Merton, four parameters ----------
# NOTE: fits like this can get stuck, so I try a few starting points and keep the best
def merton_loss(p):
    return np.mean((implied_vol(otm_price(merton_call(F, K, *p, T)), F, K, T) - market_iv) ** 2)

bounds = [(0.05, 2.0), (0.0, 30.0), (-0.5, 0.5), (0.01, 1.0)]
best = None
for x0 in [[0.4, 2, -0.05, 0.1], [0.3, 10, 0.0, 0.05], [0.5, 1, -0.2, 0.2], [0.35, 5, 0.05, 0.1]]:
    fit = minimize(merton_loss, x0=x0, bounds=bounds, method="L-BFGS-B")
    if best is None or fit.fun < best.fun:
        best = fit
sigma, lam, m, delta = best.x

bs_prices = otm_price(bs_call(F, K, sigma_bs, T))
mj_prices = otm_price(merton_call(F, K, sigma, lam, m, delta, T))
bs_iv = implied_vol(bs_prices, F, K, T)
mj_iv = implied_vol(mj_prices, F, K, T)

results = pd.DataFrame({"strike": K, "market_usd": market, "bs_usd": bs_prices, "merton_usd": mj_prices,
                        "market_iv": market_iv, "bs_iv": bs_iv, "merton_iv": mj_iv,
                        "deribit_iv": df["deribit_iv"].values / 100})
results.to_csv("data/calibration.csv", index=False)
pd.Series({"F": F, "T": T, "sigma": sigma, "lambda": lam, "m": m, "delta": delta,
           "sigma_bs": sigma_bs}).to_csv("data/merton_params.csv", header=False)

# ---------- how good is each fit? ----------
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

print(f"expiry in {T * 365:.0f} days, forward ${F:,.0f}, {len(K)} strikes\n")
print("check: my implied vols vs Deribit's own (should be close):"
      f" average gap {np.mean(np.abs(market_iv - df['deribit_iv'].values / 100)) * 100:.2f} vol points\n")
print(f"Black-Scholes: sigma = {sigma_bs:.1%}")
print(f"Merton:        sigma = {sigma:.1%}, {lam:.1f} jumps a year, "
      f"average jump {np.exp(m) - 1:+.1%}, jump spread {delta:.1%}\n")
print("                       Black-Scholes    Merton")
print(f"implied vol error       {rmse(bs_iv, market_iv) * 100:>8.2f}     {rmse(mj_iv, market_iv) * 100:>8.2f}   (vol points, RMSE)")
print(f"price error             {rmse(bs_prices, market):>8.0f}     {rmse(mj_prices, market):>8.0f}   (US dollars, RMSE)")
print(f"worst price miss        {np.max(np.abs(bs_prices - market)):>8.0f}     {np.max(np.abs(mj_prices - market)):>8.0f}   (US dollars)")

# ---------- the picture ----------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(K / F, market_iv * 100, "o", color="black", label="Deribit market")
ax.plot(K / F, mj_iv * 100, "-", label="Merton jump-diffusion (fitted)")
ax.plot(K / F, bs_iv * 100, "--", label="Black-Scholes (one volatility)")
ax.axvline(1.0, color="grey", linewidth=0.5)
ax.set_xlabel("Strike / forward price")
ax.set_ylabel("Implied volatility (%)")
ax.set_title(f"Bitcoin option smile, {T * 365:.0f} days to expiry", loc="left")
ax.legend()
fig.tight_layout()
fig.savefig("figures/deribit_smile.png", dpi=150)
