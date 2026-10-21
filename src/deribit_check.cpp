// deribit_check.cpp
// Part 3 of my Deribit comparison: price the real Bitcoin options with MY C++ Monte Carlo
// jump-diffusion pricer, using the Merton parameters fitted to the market in calibrate.py,
// and check how close my prices come to Deribit's.
//
// I price on the forward F with no interest rate (as in calibrate.py), so at expiry:
//   ln S_T = ln F + (-lambda*k - sigma^2/2) T + sigma sqrt(T) Z1 + (sum of the N jumps)
// Puts for strikes below F, calls above (the out-of-the-money options that actually trade).

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "normals.h"

int main() {
    // ---------- read the fitted parameters: lines like "sigma,0.311" ----------
    std::map<std::string, double> p;
    std::ifstream params("data/merton_params.csv");
    std::string line;
    while (std::getline(params, line)) {
        std::size_t comma = line.find(',');
        p[line.substr(0, comma)] = std::stod(line.substr(comma + 1));
    }
    double F = p["F"], T = p["T"], sigma = p["sigma"];
    double lambda = p["lambda"], m = p["m"], delta = p["delta"];

    // ---------- read the strikes and market prices (first two columns of calibration.csv) ----------
    std::vector<double> strikes, market, formula;
    std::ifstream calib("data/calibration.csv");
    std::getline(calib, line);                       // skip the header row
    while (std::getline(calib, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        strikes.push_back(row[0]);                   // strike
        market.push_back(row[1]);                    // Deribit's price in dollars
        formula.push_back(row[3]);                   // Merton formula price from Python
    }

    // ---------- simulate one million prices at expiry ----------
    int n = 1000000;
    std::mt19937_64 rng(42);
    std::vector<double> z1, z2;
    box_muller(rng, z1, n);
    box_muller(rng, z2, n);
    std::poisson_distribution<int> poisson(lambda * T);

    double k = std::exp(m + 0.5 * delta * delta) - 1.0;
    double drift = (-lambda * k - 0.5 * sigma * sigma) * T;   // NOTE: keeps E[S_T] = F
    std::vector<double> ST(n);
    for (int i = 0; i < n; i++) {
        int jumps = poisson(rng);
        double jump_total = jumps * m + std::sqrt(jumps) * delta * z2[i];
        ST[i] = F * std::exp(drift + sigma * std::sqrt(T) * z1[i] + jump_total);
    }

    // ---------- price every strike and compare ----------
    std::cout.precision(2);
    std::cout << std::fixed;
    std::cout << "strike    type   my C++ price  (std err)   Merton formula   Deribit   miss vs Deribit\n";
    double sq_miss = 0.0, worst = 0.0;
    for (std::size_t j = 0; j < strikes.size(); j++) {
        double K = strikes[j];
        bool put = K < F;
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < n; i++) {
            double pay = put ? std::max(K - ST[i], 0.0) : std::max(ST[i] - K, 0.0);
            sum += pay;
            sum_sq += pay * pay;
        }
        double price = sum / n;
        double se = std::sqrt((sum_sq / n - price * price) / n);
        double miss = price - market[j];
        sq_miss += miss * miss;
        worst = std::max(worst, std::abs(miss));
        std::cout << K << "  " << (put ? "put " : "call") << "   " << price << "  (" << se << ")   "
                  << formula[j] << "   " << market[j] << "   " << miss << "\n";
    }
    std::cout << "\nmy C++ prices vs Deribit: RMSE $" << std::sqrt(sq_miss / strikes.size())
              << ", worst miss $" << worst << " (forward price $" << F << ")\n";
    return 0;
}
