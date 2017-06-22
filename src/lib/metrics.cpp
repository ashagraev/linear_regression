#include "metrics.h"

#include <algorithm>
#include <cmath>

void TRegressionMetricsCalculator::Add(const double prediction, const double target, const double weight) {
    const double diff = prediction - target;
    MSECalculator.Add(diff * diff, weight);
    VarianceCalculator.Add(target, weight);
}

double TRegressionMetricsCalculator::RMSE() const {
    return sqrt(std::max(0., MSECalculator.GetMean()));
}
