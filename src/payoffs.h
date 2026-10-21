// payoffs.h
// Each kind of option is its own CLASS, but they all share one job:
// "given the price at expiry, tell me the payoff."
// So I write one general Payoff class, and Call, Put and Digital each fill in that job.
// Then my pricer can price ANY payoff without knowing which one it is.

#pragma once
#include <algorithm>   // std::max

// the general idea of a payoff
// NOTE: "virtual ... = 0" means every kind of payoff MUST write its own version
class Payoff {
public:
    virtual double operator()(double ST) const = 0;   // lets me write payoff(ST)
    virtual ~Payoff() {}                               // needed for classes with virtual functions
};

// call: max(S_T - K, 0)
class Call : public Payoff {          // "Call is a kind of Payoff"
public:
    Call(double K) : K(K) {}          // remember the strike when I create it
    double operator()(double ST) const override { return std::max(ST - K, 0.0); }
private:
    double K;
};

// put: max(K - S_T, 0)
class Put : public Payoff {
public:
    Put(double K) : K(K) {}
    double operator()(double ST) const override { return std::max(K - ST, 0.0); }
private:
    double K;
};

// digital call: pays 1 if S_T > K, otherwise 0
class Digital : public Payoff {
public:
    Digital(double K) : K(K) {}
    double operator()(double ST) const override { return ST > K ? 1.0 : 0.0; }
private:
    double K;
};



