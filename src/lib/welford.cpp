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

double TDeviationCalculator::GetSumWeights() const {
    return MeanCalculator.GetSumWeights();
}
