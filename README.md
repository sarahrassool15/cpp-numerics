# Numerical methods in C++

My C++ work for computational finance andn I am starting from the bottom: generating the random numbers every simulation depends on.

## Making normal random numbers from scratch

A computer can only really produce uniform random numbers, but almost every model in finance needs normal ones. The normal CDF can't be inverted with a formula, so the usual trick for converting uniforms doesn't work directly.

The Box-Muller method gets around this by making two normals at once. Two independent normals form a round cloud around the origin, so any point in it can be described by a direction and a distance. The direction is uniform, so it's just 2πU₂. The distance has a CDF that can be inverted, which gives R = √(−2 ln U₁). Converting back with cos and sin gives two independent standard normals.

I implemented it in C++ and tested ten million draws against C++'s built-in normal generator and against theory:

| | Mean | Variance | Skewness | Kurtosis | P(\|Z\|>2) | P(\|Z\|>3) |
|---|---|---|---|---|---|---|
| My Box-Muller | 0.0003 | 1.0000 | 0.0013 | 2.9996 | 0.0455 | 0.0027 |
| C++ built-in | 0.0000 | 1.0003 | 0.0011 | 2.9990 | 0.0455 | 0.0027 |
| Theory | 0 | 1 | 0 | 3 | 0.0455 | 0.0027 |

Getting the middle of a distribution right is easy; the tails are the real test, and they match to four decimal places. The kurtosis is 3, compared with 12.9 for hourly Bitcoin returns in my earlier project.

![Histogram and QQ plot](figures/box_muller.png)

Box-Muller makes its numbers in pairs, so I also checked that the two in each pair are uncorrelated. With seed 42 the correlation was −0.0009, about two standard errors from zero, but other seeds gave values scattered on both sides of zero, so it's chance rather than a flaw.

Both generators produce ten million normals in about a tenth of a second.

One detail that matters: the uniform generator can return exactly 0, and ln 0 is −∞, so I use 1 − U, which can never be 0.

## Running it

```bash
mkdir -p build
clang++ -std=c++17 -O2 -Wall src/box_muller.cpp -o build/box_muller
./build/box_muller
python src/plot_samples.py
```