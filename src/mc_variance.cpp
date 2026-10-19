// mc_variance.cpp
// L5 step 3: variance reduction.
// Same option, same number of payoffs, three ways of averaging:
//   1. plain Monte Carlo
//   2. antithetic variates: use every normal twice, as Z and -Z
//   3. control variate: correct my answer using the stock price, whose true mean I know
// The estimators themselves now live in estimators.h, so I can reuse them in step 4.

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include "normals.h"
#include "black_scholes.h"
#include "payoffs.h"
#include "estimators.h"

// prints one method's result, and how many plain paths it is "worth"
// NOTE: variance shrinks like 1/n, so a standard error k times smaller is worth k^2 times the paths
void report(std::string name, Result res, Result base, double exact) {
    double ratio = (base.std_error / res.std_error) * (base.std_error / res.std_error);
    std::cout << "  " << name
              << "  price " << res.price
              << "   std error " << res.std_error
              << "   error/se " << (res.price - exact) / res.std_error
              << "   worth " << ratio << "x plain paths\n";
}

int main() {
    double S0 = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    int n = 1000000;

    std::mt19937_64 rng(42);
    std::vector<double> z;
    box_muller(rng, z, n);

    Call call(K);
    Put put(K);

    std::cout.precision(4);
    std::cout << std::fixed;

    std::cout << "Call (exact " << bs_call(S0, K, r, sigma, T) << ")\n";
    Result base = plain(call, S0, r, sigma, T, z);
    report("plain      ", base, base, bs_call(S0, K, r, sigma, T));
    report("antithetic ", antithetic(call, S0, r, sigma, T, z), base, bs_call(S0, K, r, sigma, T));
    report("control var", control_variate(call, S0, r, sigma, T, z), base, bs_call(S0, K, r, sigma, T));

    std::cout << "\nPut (exact " << bs_put(S0, K, r, sigma, T) << ")\n";
    base = plain(put, S0, r, sigma, T, z);
    report("plain      ", base, base, bs_put(S0, K, r, sigma, T));
    report("antithetic ", antithetic(put, S0, r, sigma, T, z), base, bs_put(S0, K, r, sigma, T));
    report("control var", control_variate(put, S0, r, sigma, T, z), base, bs_put(S0, K, r, sigma, T));
    return 0;
}
