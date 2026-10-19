// estimators.h
// My three Monte Carlo estimators from step 3, moved into a header
// so that both mc_variance.cpp and mc_convergence.cpp can use them.

#pragma once
#include <cmath>
#include <vector>
#include "payoffs.h"

struct Result {
    double price;
    double std_error;
};

// ---------- 1. plain Monte Carlo ----------
inline Result plain(const Payoff& payoff, double S0, double r, double sigma, double T,
                    const std::vector<double>& z) {
    int n = z.size();
    double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double p = disc * payoff(S0 * std::exp(drift + vol * z[i]));
        sum += p;
        sum_sq += p * p;
    }
    double mean = sum / n;
    return {mean, std::sqrt((sum_sq / n - mean * mean) / n)};
}

// ---------- 2. antithetic variates ----------
// uses n/2 normals, each twice (Z and -Z), so it still costs n payoffs
inline Result antithetic(const Payoff& payoff, double S0, double r, double sigma, double T,
                         const std::vector<double>& z) {
    int pairs = z.size() / 2;
    double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < pairs; i++) {
        double up = payoff(S0 * std::exp(drift + vol * z[i]));
        double down = payoff(S0 * std::exp(drift - vol * z[i]));
        double p = disc * 0.5 * (up + down);
        sum += p;
        sum_sq += p * p;
    }
    double mean = sum / pairs;
    return {mean, std::sqrt((sum_sq / pairs - mean * mean) / pairs)};
}

// ---------- 3. control variate (the discounted stock price, whose mean is S0) ----------
inline Result control_variate(const Payoff& payoff, double S0, double r, double sigma, double T,
                              const std::vector<double>& z) {
    int n = z.size();
    double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    std::vector<double> x(n), y(n);
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol * z[i]);
        x[i] = disc * payoff(ST);
        y[i] = disc * ST;
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;

    double cov = 0.0, var_y = 0.0;
    for (int i = 0; i < n; i++) {
        cov += (x[i] - mx) * (y[i] - my);
        var_y += (y[i] - my) * (y[i] - my);
    }
    double b = cov / var_y;

    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double c = x[i] - b * (y[i] - S0);
        sum += c;
        sum_sq += c * c;
    }
    double mean = sum / n;
    return {mean, std::sqrt((sum_sq / n - mean * mean) / n)};
}
