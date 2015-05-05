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
};

// accurately computes (w_1 * x_1 * y_1 + w_2 * x_2 * y_2 + ... + w_n * x_n * y_n) / (w_1 + w_2 + ... + w_n)
class TCovariationCalculator {
private:
    double Covariation = 0.;
    TMeanCalculator FirstValueMeanCalculator;
    TMeanCalculator SecondValueMeanCalculator;
public:
    void Add(const double firstValue, const double secondValue, const double weight = 1.);

    double GetFirstValueMean() const;
    double GetSecondValueMean() const;

    double GetCovariation() const;
};

// accurately computes (w_1 * x_1 * x_1 + w_2 * x_2 * x_2 + ... + w_n * x_n * x_n) / (w_1 + w_2 + ... + w_n)
class TDeviationCalculator {
private:
    double Covariation = 0.;
    TMeanCalculator MeanCalculator;
public:
    void Add(const double value, const double weight = 1.);

    double GetMean() const;
    double GetDeviation() const;
};
