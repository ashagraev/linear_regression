#pragma once

#include "kahan.h"

class TMeanCalculator {
private:
    double Mean = 0.;
    TKahanAccumulator SumWeights;
public:
    void Add(const double value, const double weight = 1.);
    double GetMean() const;
    double GetSumWeights() const;
};

class TVarianceCalculator {
private:
    double Variance = 0.;
    TMeanCalculator MeanCalculator;
public:
    void Add(const double value, const double weight = 1.);

    double GetMean() const;
    double GetVariance() const;
};
