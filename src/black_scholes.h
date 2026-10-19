// black_scholes.h
// The exact Black- Scholes prices, so I can check my Monte Carlo answers against them

#pragma once
#include <cmath>

// standard normal CDF, Phi(x)
// NOTE: C++ has no Phi built in but it has erfc, and Phi(x) = 0.5 * erfc(-x / sqrt(2))
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// European call:  C = S0 Phi(d1) - K e^{-rT} Phi(d2)
inline double bs_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// European put from put call parity:  P = C - S0 + K e^{-rT}
inline double bs_put(double S0, double K, double r, double sigma, double T) {
    return bs_call(S0, K, r, sigma, T) - S0 + K * std::exp(-r * T);
}