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
