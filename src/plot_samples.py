# plot_samples.py
# Plots the Box-Muller numbers my C++ program saved, against the real normal curve.
# Run the C++ program first so build/samples.csv exists.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

z = pd.read_csv("build/samples.csv")["box_muller"]
x = np.linspace(-5, 5, 400)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# histogram of my numbers with the N(0, 1) curve on top
axes[0].hist(z, bins=100, range=(-5, 5), density=True, alpha=0.6, label="my Box-Muller (C++)")
axes[0].plot(x, stats.norm.pdf(x), label="N(0, 1)")
axes[0].set_xlabel("z")
axes[0].legend()

# QQ plot: if my numbers are normal, the dots sit on a straight line
# NOTE: compare with the Bitcoin QQ plot from my first project, which bent away at both ends
stats.probplot(z, dist="norm", plot=axes[1])
axes[1].set_title("QQ plot against N(0, 1)")

fig.tight_layout()
fig.savefig("figures/box_muller.png", dpi=150)


