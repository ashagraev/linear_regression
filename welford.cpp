#include "welford.h"

#include <cmath>

void TMeanCalculator::Add(const double value, const double weight /*= 1.*/) {
    SumWeights += weight;
    if (SumWeights) {
        Mean += weight * (value - Mean) / SumWeights;
    }
}

double TMeanCalculator::GetMean() const {
    return Mean;
}

double TMeanCalculator::GetSumWeights() const {
    return SumWeights;
}

void TCovariationCalculator::Add(const double firstValue, const double secondValue, const double weight /*= 1.*/) {
    SumWeights += weight;
    if (SumWeights) {
        FirstValueMean += weight * (firstValue - FirstValueMean) / SumWeights;
        Covariation += weight * (firstValue - FirstValueMean) * (secondValue - SecondValueMean);
        SecondValueMean += weight * (secondValue - SecondValueMean) / SumWeights;
    }
}

double TCovariationCalculator::GetFirstValueMean() const {
    return FirstValueMean;
}

double TCovariationCalculator::GetSecondValueMean() const {
    return SecondValueMean;
}

double TCovariationCalculator::GetCovariation() const {
    return Covariation;
}

double TCovariationCalculator::GetSumWeights() const {
    return SumWeights;
}

void TDeviationCalculator::Add(const double value, const double weight /*= 1.*/) {
    const double lastMean = MeanCalculator.GetMean();
    MeanCalculator.Add(value, weight);
    Deviation += weight * (value - lastMean) * (value - MeanCalculator.GetMean());
}

double TDeviationCalculator::GetMean() const {
    return MeanCalculator.GetMean();
}

double TDeviationCalculator::GetDeviation() const {
    return Deviation;
}

double TDeviationCalculator::GetStdDev() const {
    const double sumWeights = GetSumWeights();
    if (!sumWeights) {
        return 0.;
    }
    return sqrt(GetDeviation() / sumWeights);
}

double TDeviationCalculator::GetSumWeights() const {
    return MeanCalculator.GetSumWeights();
}
