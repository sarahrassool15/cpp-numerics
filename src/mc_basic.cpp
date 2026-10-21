// mc_basic.cpp
// step 1: pricing a European call by Monte Carlo, and checking it against Black-Scholes.
//
// The recipe:
//   1. simulate the stock price at expiry under the RISK NEUTRAL model (drift r, not mu):
//          S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)
//      (this is the exact GBM solution from my stochastic calculus project)
//   2. work out the payoff on each path:  max(S_T - K, 0)
//   3. average the payoffs and discount back to today:  price = e^{-rT} * average
// The Monte Carlo price is itself random so I also need to report its standard error

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>          // std::max
#include "normals.h"          // my Box Muller from last S6 which generates standard normals 
#include "black_scholes.h"    // the exact answer, to check against

int main() {
    // a standard textbook example: at-the-money call, one year
    double S0 = 100.0;     // price today
    double K = 100.0;      // strike
    double r = 0.05;       // risk free rate
    double sigma = 0.20;   // volatility
    double T = 1.0;        // one year to expiry

    int n = 1000000;                 // one million simulated paths
    std::mt19937_64 rng(42);
    std::vector<double> z;
    box_muller(rng, z, n);           // one normal per path

    double drift = (r - 0.5 * sigma * sigma) * T;   // NOTE: the Ito correction again
    double vol = sigma * std::sqrt(T);

    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol * z[i]);   // price at expiry on this path
        double payoff = std::max(ST - K, 0.0);           // call payoff
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    double mean = sum / n;
    double var = sum_sq / n - mean * mean;            // variance of the payoff
    double discount = std::exp(-r * T);

    double price = discount * mean;
    double std_error = discount * std::sqrt(var / n); // NOTE: shrinks like 1/sqrt(n)
    double exact = bs_call(S0, K, r, sigma, T);

    std::cout.precision(4);
    std::cout << std::fixed;
    std::cout << "Monte Carlo price:  " << price << "\n";
    std::cout << "Standard error:     " << std_error << "\n";
    std::cout << "95% interval:       [" << price - 1.96 * std_error << ", "
              << price + 1.96 * std_error << "]\n";
    std::cout << "Black-Scholes:      " << exact << "\n";
    std::cout << "Error:              " << price - exact
              << "  (" << (price - exact) / std_error << " standard errors)\n";
    return 0;
}


