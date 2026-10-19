// mc_payoffs.cpp
// L5 step 2: one Monte Carlo pricer that works for any payoff.
// I price a call, a put and a digital with the SAME function and check each against its formula.

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include "normals.h"
#include "black_scholes.h"
#include "payoffs.h"

// a small container for the two numbers my pricer returns
struct Result {
    double price;
    double std_error;
};

// prices any payoff by Monte Carlo
// NOTE: "const Payoff& payoff" accepts a Call, a Put or a Digital.
// The pricer never needs to know which one; it just calls payoff(ST).
Result mc_price(const Payoff& payoff, double S0, double r, double sigma, double T,
                const std::vector<double>& z) {
    int n = z.size();
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);

    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol * z[i]);
        double p = payoff(ST);            // works for every kind of payoff
        sum += p;
        sum_sq += p * p;
    }
    double mean = sum / n;
    double var = sum_sq / n - mean * mean;
    double discount = std::exp(-r * T);
    return {discount * mean, discount * std::sqrt(var / n)};
}

// prints one line: Monte Carlo vs exact
void report(std::string name, Result mc, double exact) {
    std::cout << name << "   MC " << mc.price << " (+/- " << mc.std_error << ")"
              << "   exact " << exact
              << "   error in std errors " << (mc.price - exact) / mc.std_error << "\n";
}

int main() {
    double S0 = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    int n = 1000000;

    std::mt19937_64 rng(42);
    std::vector<double> z;
    box_muller(rng, z, n);

    // create the three options
    Call call(K);
    Put put(K);
    Digital digital(K);

    std::cout.precision(4);
    std::cout << std::fixed;

    // NOTE: same random numbers for all three, so their errors are linked
    report("Call   ", mc_price(call, S0, r, sigma, T, z), bs_call(S0, K, r, sigma, T));
    report("Put    ", mc_price(put, S0, r, sigma, T, z), bs_put(S0, K, r, sigma, T));
    report("Digital", mc_price(digital, S0, r, sigma, T, z), bs_digital(S0, K, r, sigma, T));
    return 0;
}