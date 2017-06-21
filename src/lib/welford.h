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
