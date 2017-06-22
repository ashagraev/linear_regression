#pragma once

#include "welford.h"
#include "pool.h"

class TRegressionMetricsCalculator {
private:
    TDeviationCalculator VarianceCalculator;
    TMeanCalculator MSECalculator;
public:
    void Add(const double prediction, const double target, const double weight);
    double RMSE() const;
};

template <typename TModel>
double RMSE(TPool::TPoolIterator iterator, const TModel& model) {
    TRegressionMetricsCalculator rmc;
    for (; iterator.IsValid(); ++iterator) {
        rmc.Add(model.Prediction(*iterator), iterator->Goal, iterator->Weight);
    }
    return rmc.RMSE();
}

