#pragma once

#include "welford.h"

class TRegressionMetricsCalculator {
private:
    TVarianceCalculator VarianceCalculator;
    TMeanCalculator MSECalculator;
public:
    void Add(const double prediction, const double target, const double weight);
    double RMSE() const;
    double DeterminationCoefficient() const;
};

template <typename TModel, typename TIterator>
double Metric(TIterator iterator, const TModel& model, double (TRegressionMetricsCalculator::*func)() const) {
    TRegressionMetricsCalculator rmc;
    for (; iterator.IsValid(); ++iterator) {
        rmc.Add(model.Prediction(*iterator), iterator->Goal, iterator->Weight);
    }
    return (rmc.*func)();
}

template <typename TModel, typename TIterator>
double RMSE(TIterator iterator, const TModel& model) {
    return Metric(iterator, model, &TRegressionMetricsCalculator::RMSE);
}

template <typename TModel, typename TIterator>
double DeterminationCoefficient(TIterator iterator, const TModel& model) {
    return Metric(iterator, model, &TRegressionMetricsCalculator::DeterminationCoefficient);
}
