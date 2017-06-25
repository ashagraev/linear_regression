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

void TVarianceCalculator::Add(const double value, const double weight /*= 1.*/) {
    const double lastMean = MeanCalculator.GetMean();
    MeanCalculator.Add(value, weight);

    const double sumWeights = MeanCalculator.GetSumWeights();
    if (!sumWeights) {
        return;
    }

    Variance += weight * ((value - lastMean) * (value - MeanCalculator.GetMean()) - Variance) / sumWeights;
}

double TVarianceCalculator::GetMean() const {
    return MeanCalculator.GetMean();
}

double TVarianceCalculator::GetVariance() const {
    return Variance;
}
