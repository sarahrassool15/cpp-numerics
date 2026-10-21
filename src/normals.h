// normals.h
// My Box-Muller generator from my last project, moved into a header file so other programs can use it.
// Any .cpp file that writes  #include "normals.h"  gets this function.

#pragma once            // NOTE: stops this file being included twice by accident
#include <cmath>
#include <random>
#include <vector>

// fills z with n standard normal numbers (same method as S6)
inline void box_muller(std::mt19937_64& rng, std::vector<double>& z, int n) {
    const double PI = 3.14159265358979323846;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    z.resize(n);
    for (int i = 0; i < n; i += 2) {
        double u1 = 1.0 - uniform(rng);              // never exactly 0, so log is safe
        double u2 = uniform(rng);
        double r = std::sqrt(-2.0 * std::log(u1));   // distance
        double angle = 2.0 * PI * u2;                // angle
        z[i] = r * std::cos(angle);
        if (i + 1 < n) z[i + 1] = r * std::sin(angle);
    }
}