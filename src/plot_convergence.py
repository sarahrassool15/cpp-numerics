# plot_convergence.py
# Plots the actual error of each method against the number of paths, on log-log axes.
# Run ./build/mc_convergence first so build/convergence.csv exists.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("build/convergence.csv")

fig, ax = plt.subplots(figsize=(7, 5))
for method, marker in [("plain", "o"), ("antithetic", "s"), ("control variate", "^")]:
    d = df[df["method"] == method]
    # slope of log(error) against log(N): theory says -1/2
    slope = np.polyfit(np.log(d["n"]), np.log(d["rmse"]), 1)[0]
    ax.loglog(d["n"], d["rmse"], marker + "-", label=f"{method} (slope {slope:.2f})")
    # dashed: the standard error my pricer reported, to check it was honest
    ax.loglog(d["n"], d["avg_std_error"], "k--", lw=0.8, alpha=0.5)

ax.set_xlabel("Number of paths N")
ax.set_ylabel("Actual error (RMSE over 500 runs)")
ax.set_title("Call price error: every method falls like 1/√N", loc="left")
ax.legend()
fig.tight_layout()
fig.savefig("figures/convergence.png", dpi=150)

# print the slopes and how many times better each method is at the largest N
big = df[df["n"] == df["n"].max()].set_index("method")["rmse"]
for method in ["antithetic", "control variate"]:
    print(f"{method}: worth {(big['plain'] / big[method])**2:.1f}x plain paths")
