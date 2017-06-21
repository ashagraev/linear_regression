#pragma once

#include "kahan.h"

// accurately computes (w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n) / (w_1 + w_2 + ... + w_n)
class TMeanCalculator {
private:
    double Mean = 0.;
    TKahanAccumulator SumWeights;
public:
    void Add(const double value, const double weight = 1.);
    double GetMean() const;
    double GetSumWeights() const;
};

// accurately computes (w_1 * x_1 * x_1 + w_2 * x_2 * x_2 + ... + w_n * x_n * x_n) / (w_1 + w_2 + ... + w_n)
class TDeviationCalculator {
private:
    double Deviation = 0.;
    TMeanCalculator MeanCalculator;
public:
    void Add(const double value, const double weight = 1.);

    double GetMean() const;
    double GetDeviation() const;
    double GetSumWeights() const;
};
