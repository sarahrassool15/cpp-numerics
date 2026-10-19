// mc_convergence.cpp
// L5 step 4: does the error really shrink like 1/sqrt(N)?
//
// A single run can be lucky or unlucky, so for each number of paths N
// I price the call 500 times with DIFFERENT random numbers and measure:
//   - the actual error:  RMSE = sqrt( average of (price - exact)^2 )
//   - the standard error my pricer REPORTED, averaged over the 500 runs
// If the theory is right, RMSE should fall like 1/sqrt(N) (slope -1/2 on a log-log plot),
// and the reported standard error should match the actual error.
// I save the results to a CSV and plot them in Python.

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include "normals.h"
#include "black_scholes.h"
#include "payoffs.h"
#include "estimators.h"

int main() {
    double S0 = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 1.0;
    double exact = bs_call(S0, K, r, sigma, T);
    Call call(K);

    std::vector<int> sizes = {100, 316, 1000, 3162, 10000, 31623, 100000};
    int repeats = 500;
    std::vector<std::string> methods = {"plain", "antithetic", "control variate"};

    std::mt19937_64 rng(42);
    std::vector<double> z;

    std::ofstream file("build/convergence.csv");
    file << "method,n,rmse,avg_std_error\n";

    std::cout.precision(5);
    std::cout << std::fixed;

    for (int n : sizes) {
        // running totals for each method: sum of squared errors and of reported std errors
        std::vector<double> sq_err(3, 0.0), se_sum(3, 0.0);

        for (int rep = 0; rep < repeats; rep++) {
            box_muller(rng, z, n);   // NOTE: fresh random numbers every repeat

            Result res[3] = {
                plain(call, S0, r, sigma, T, z),
                antithetic(call, S0, r, sigma, T, z),
                control_variate(call, S0, r, sigma, T, z)
            };
            for (int m = 0; m < 3; m++) {
                double err = res[m].price - exact;
                sq_err[m] += err * err;
                se_sum[m] += res[m].std_error;
            }
        }

        for (int m = 0; m < 3; m++) {
            double rmse = std::sqrt(sq_err[m] / repeats);
            double avg_se = se_sum[m] / repeats;
            file << methods[m] << "," << n << "," << rmse << "," << avg_se << "\n";
            std::cout << "N = " << n << "   " << methods[m]
                      << "   actual error " << rmse << "   reported std error " << avg_se << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}
