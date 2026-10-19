// mc_variance.cpp
// L5 step 3: variance reduction.
// Same option, same number of payoffs, three ways of averaging:
//   1. plain Monte Carlo (what I did in steps 1 and 2)
//   2. antithetic variates: use every normal twice, as Z and -Z
//   3. control variate: correct my answer using something whose true value I know exactly
// The question: how much smaller is the standard error, and how many plain paths is that worth?

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include "normals.h"
#include "black_scholes.h"
#include "payoffs.h"

struct Result {
    double price;
    double std_error;
};

// ---------- 1. plain Monte Carlo ----------
Result plain(const Payoff& payoff, double S0, double r, double sigma, double T,
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
// Every Z has a mirror image -Z that is just as likely. If Z gives a high price,
// -Z gives a low one, so averaging the pair cancels a lot of the randomness.
// NOTE: to be fair I use only n/2 normals, so I still evaluate n payoffs in total.
Result antithetic(const Payoff& payoff, double S0, double r, double sigma, double T,
                  const std::vector<double>& z) {
    int pairs = z.size() / 2;
    double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < pairs; i++) {
        double up = payoff(S0 * std::exp(drift + vol * z[i]));
        double down = payoff(S0 * std::exp(drift - vol * z[i]));   // the mirror path
        double p = disc * 0.5 * (up + down);                       // one number per PAIR
        sum += p;
        sum_sq += p * p;
    }
    // NOTE: the pairs are independent of each other, so the standard error
    // comes from the spread of the pair averages, with `pairs` observations
    double mean = sum / pairs;
    return {mean, std::sqrt((sum_sq / pairs - mean * mean) / pairs)};
}

// ---------- 3. control variate ----------
// I know one thing exactly: the discounted stock price has expected value S0
// (that's what drift r guarantees). So on each path I also record Y = e^{-rT} S_T.
// If my sample of Y came out too high, my option prices probably did too,
// so I correct:   estimate = mean(X) - b * (mean(Y) - S0)
// The best b is Cov(X, Y) / Var(Y), the slope of a regression of X on Y.
Result control_variate(const Payoff& payoff, double S0, double r, double sigma, double T,
                       const std::vector<double>& z) {
    int n = z.size();
    double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    // first pass: means
    std::vector<double> x(n), y(n);
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) {
        double ST = S0 * std::exp(drift + vol * z[i]);
        x[i] = disc * payoff(ST);    // what I want to price
        y[i] = disc * ST;            // what I already know the answer to
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;

    // second pass: the regression slope b
    double cov = 0.0, var_y = 0.0;
    for (int i = 0; i < n; i++) {
        cov += (x[i] - mx) * (y[i] - my);
        var_y += (y[i] - my) * (y[i] - my);
    }
    double b = cov / var_y;

    // third pass: the corrected values and their spread
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double c = x[i] - b * (y[i] - S0);
        sum += c;
        sum_sq += c * c;
    }
    double mean = sum / n;
    return {mean, std::sqrt((sum_sq / n - mean * mean) / n)};
}

// prints one method's result, and how many plain paths it is "worth"
// NOTE: variance shrinks like 1/n, so if my standard error is k times smaller,
// I would have needed k^2 times more plain paths to get the same accuracy
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