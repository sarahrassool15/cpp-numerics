# plot_smile.py
# Plots the implied volatility at each strike for my two kinds of jumps.
# Run ./build/mc_jumps first so build/smile.csv exists.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("build/smile.csv")

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, marker in [("up and down jumps", "o"), ("crash jumps", "s")]:
    d = df[df["case"] == name]
    ax.plot(d["strike"], d["implied_vol"], marker + "-", label=name)

# without jumps, Black-Scholes is right and the line would be flat at the everyday volatility
ax.axhline(0.50, color="grey", linestyle="--", label="no jumps (Black-Scholes)")
ax.axvline(100, color="grey", linewidth=0.5)   # today's price

ax.set_xlabel("Strike (today's price = 100)")
ax.set_ylabel("Implied volatility")
ax.set_title("Jumps bend the flat Black-Scholes line into a smile or a skew", loc="left")
ax.legend()
fig.tight_layout()
fig.savefig("figures/smile.png", dpi=150)
