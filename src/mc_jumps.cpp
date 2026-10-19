// mc_jumps.cpp
// L5 step 6: adding Bitcoin-style jumps to my pricer.
//
// In my first project, Bitcoin returns had fat tails: sudden big moves far more often
// than a normal distribution allows. Black-Scholes ignores that. Merton's model (1976) adds it:
// the price moves like GBM most of the time, but every so often a JUMP hits.
//   - jumps arrive at random, on average lambda times a year (a Poisson process)
//   - each jump multiplies the price by e^Y, with Y ~ N(m, delta^2)
//
// At expiry this gives:
//   ln S_T = ln S0 + (r - lambda*k - sigma^2/2) T + sigma sqrt(T) Z1 + (sum of the N jumps)
// where N ~ Poisson(lambda T) and k = E[e^Y - 1] = e^{m + delta^2/2} - 1.
// NOTE: the "- lambda*k" keeps the risk-neutral drift right: without it,
// the jumps would push the expected price away from S0 e^{rT}.
//
// I try two kinds of jumps: ones equally likely up or down, and "crash" jumps that are mostly down.
// Then I ask the key question: what volatility would Black-Scholes need to match each price?
// That's the IMPLIED volatility. Without jumps it's the same at every strike.
// With jumps it should bend: into a smile, or a skew if the jumps are mostly crashes.

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include "normals.h"
#include "black_scholes.h"

// Merton's exact price: an average of Black-Scholes prices,
// one for each possible number of jumps n, weighted by how likely n jumps is
double merton_call(double S0, double K, double r, double sigma, double T,
                   double lambda, double m, double delta) {
    double k = std::exp(m + 0.5 * delta * delta) - 1.0;
    double lam2 = lambda * (1.0 + k);
    double price = 0.0;
    double weight = std::exp(-lam2 * T);            // probability of 0 jumps
    for (int n = 0; n < 60; n++) {                  // 60 terms is plenty
        double sigma_n = std::sqrt(sigma * sigma + n * delta * delta / T);
        double r_n = r - lambda * k + n * std::log(1.0 + k) / T;
        price += weight * bs_call(S0, K, r_n, sigma_n, T);
        weight *= lam2 * T / (n + 1);               // next Poisson probability
    }
    return price;
}

// implied volatility: the sigma that makes Black-Scholes match a given price
// NOTE: the call price rises with sigma, so I can find it by bisection (halving the interval)
double implied_vol(double price, double S0, double K, double r, double T) {
    double lo = 0.001, hi = 5.0;
    for (int i = 0; i < 100; i++) {
        double mid = 0.5 * (lo + hi);
        if (bs_call(S0, K, r, mid, T) < price) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// prices calls at strikes 60 to 150 under one set of jump parameters,
// prints the results, and saves them to the CSV file
void run_case(std::string name, double lambda, double m, double delta,
              std::mt19937_64& rng, std::ofstream& file) {
    double S0 = 100.0, r = 0.05, T = 0.25;   // three-month options
    double sigma = 0.50;                     // everyday (non-jump) volatility, Bitcoin-like
    int n = 1000000;

    std::vector<double> z1, z2;
    box_muller(rng, z1, n);                  // for the everyday moves
    box_muller(rng, z2, n);                  // for the jumps
    std::poisson_distribution<int> poisson(lambda * T);

    // simulate S_T once, then price every strike with the same paths
    double k = std::exp(m + 0.5 * delta * delta) - 1.0;
    double drift = (r - lambda * k - 0.5 * sigma * sigma) * T;
    std::vector<double> ST(n);
    for (int i = 0; i < n; i++) {
        int jumps = poisson(rng);            // how many jumps on this path
        // NOTE: the sum of N normal jumps is normal with mean N*m and variance N*delta^2
        double jump_total = jumps * m + std::sqrt(jumps) * delta * z2[i];
        ST[i] = S0 * std::exp(drift + sigma * std::sqrt(T) * z1[i] + jump_total);
    }

    std::cout << name << "\n";
    std::cout << "strike   MC price   (std err)   Merton exact   implied vol\n";
    double disc = std::exp(-r * T);
    for (double K = 60.0; K <= 150.0; K += 10.0) {
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < n; i++) {
            double p = disc * std::max(ST[i] - K, 0.0);
            sum += p;
            sum_sq += p * p;
        }
        double price = sum / n;
        double se = std::sqrt((sum_sq / n - price * price) / n);
        double exact = merton_call(S0, K, r, sigma, T, lambda, m, delta);
        double iv = implied_vol(exact, S0, K, r, T);

        file << name << "," << K << "," << price << "," << se << "," << exact << "," << iv << "\n";
        std::cout << K << "   " << price << "   (" << se << ")   " << exact
                  << "        " << iv << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::mt19937_64 rng(42);
    std::ofstream file("build/smile.csv");
    file << "case,strike,mc_price,std_error,merton_price,implied_vol\n";

    std::cout.precision(4);
    std::cout << std::fixed;

    // two kinds of jumps, both about 5 a year (illustrative numbers, not fitted to data)
    run_case("up and down jumps", 5.0, 0.0, 0.20, rng, file);    // jumps equally likely either way
    run_case("crash jumps", 5.0, -0.10, 0.10, rng, file);        // jumps mostly down

    std::cout << "Without jumps, implied vol would be 0.5000 at every strike.\n";
    return 0;
}
