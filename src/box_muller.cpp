// box_muller.cpp
// My first C++ project: making normal random numbers myself with Box-Muller,
// then checking that they really are standard normal.
//
// The idea: two independent normals form a round cloud, so I can build them from
// a random DIRECTION and a random DISTANCE instead of two separate numbers.
//     distance R = sqrt(-2 ln U1)
//     angle      = 2 pi U2
//     Z1 = R cos(angle),   Z2 = R sin(angle)
// I compare my version with the normal generator built into C++.

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <fstream>

const double PI = 3.14159265358979323846;


// fills z with n normal numbers using Box-Muller
// NOTE: the & means I pass the original vector (and random engine), not a copy,
// so the function can fill it in. Without it the engine would repeat the same numbers.
void box_muller(std::mt19937_64& rng, std::vector<double>& z, int n) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    z.resize(n);

    // each pass makes TWO normals, so I jump forward by 2
    for (int i = 0; i < n; i += 2) {
        // NOTE: uniform can return exactly 0, and log(0) = -infinity.
        // 1 - u moves the range to (0, 1] so the log is always fine.
        double u1 = 1.0 - uniform(rng);
        double u2 = uniform(rng);

        double r = std::sqrt(-2.0 * std::log(u1));   // U1 becomes the distance
        double angle = 2.0 * PI * u2;                // U2 becomes the angle

        z[i] = r * std::cos(angle);
        if (i + 1 < n) {               // only matters if n is odd
            z[i + 1] = r * std::sin(angle);
        }
    }
}


// C++'s own normal generator, to compare against
void built_in(std::mt19937_64& rng, std::vector<double>& z, int n) {
    std::normal_distribution<double> normal(0.0, 1.0);
    z.resize(n);
    for (int i = 0; i < n; i++) {
        z[i] = normal(rng);
    }
}


// prints the checks I care about.
// for a standard normal I should get: mean 0, variance 1, skew 0, kurtosis 3,
// and P(|Z| > 1) = 0.3173, P(|Z| > 2) = 0.0455, P(|Z| > 3) = 0.0027
void check(std::string name, const std::vector<double>& z) {
    double n = z.size();

    // first pass: the mean
    double mean = 0.0;
    for (double x : z) mean += x;
    mean /= n;

    // second pass: spread, skew, kurtosis, and how often I land far out
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    double over1 = 0, over2 = 0, over3 = 0;
    for (double x : z) {
        double d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
        if (std::abs(x) > 1) over1++;
        if (std::abs(x) > 2) over2++;
        if (std::abs(x) > 3) over3++;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    double skew = m3 / std::pow(m2, 1.5);
    double kurt = m4 / (m2 * m2);     // the tails test: should be 3, Bitcoin was 12.9

    std::cout << name << "\n";
    std::cout << "  mean " << mean << "   variance " << m2
              << "   skew " << skew << "   kurtosis " << kurt << "\n";
    std::cout << "  P(|Z|>1) " << over1 / n << "   P(|Z|>2) " << over2 / n
              << "   P(|Z|>3) " << over3 / n << "\n\n";
}


int main() {
    int n = 10000000;               // ten million draws
    std::mt19937_64 rng(42);        // random engine, fixed seed so results repeat
    std::vector<double> z_mine, z_cpp;

    // time both generators
    auto start = std::chrono::steady_clock::now();
    box_muller(rng, z_mine, n);
    auto middle = std::chrono::steady_clock::now();
    built_in(rng, z_cpp, n);
    auto end = std::chrono::steady_clock::now();

    double ms_mine = std::chrono::duration<double, std::milli>(middle - start).count();
    double ms_cpp = std::chrono::duration<double, std::milli>(end - middle).count();

    std::cout.precision(4);
    std::cout << std::fixed;

    check("My Box-Muller", z_mine);
    check("C++ built-in", z_cpp);
    std::cout << "Theory\n"
              << "  mean 0.0000   variance 1.0000   skew 0.0000   kurtosis 3.0000\n"
              << "  P(|Z|>1) 0.3173   P(|Z|>2) 0.0455   P(|Z|>3) 0.0027\n\n";

    // Z1 and Z2 come out in pairs and should be independent,
    // so the correlation between z[0]&z[1], z[2]&z[3], ... should be about 0
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (int i = 0; i + 1 < n; i += 2) {
        sxy += z_mine[i] * z_mine[i + 1];
        sxx += z_mine[i] * z_mine[i];
        syy += z_mine[i + 1] * z_mine[i + 1];
    }
    std::cout << "Correlation inside my pairs: " << sxy / std::sqrt(sxx * syy)
              << "   (standard error " << 1.0 / std::sqrt(n / 2.0) << ")\n";

    std::cout.precision(0);
    std::cout << "Time for ten million draws: mine " << ms_mine
              << " ms, built-in " << ms_cpp << " ms\n";

    // save some of my draws so I can plot them in Python
    std::ofstream file("build/samples.csv");
    file << "box_muller\n";
    for (int i = 0; i < 200000; i++) {
        file << z_mine[i] << "\n";
    }

    return 0;
}