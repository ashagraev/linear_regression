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

    template <typename TModel, typename TIterator>
    static inline TRegressionMetricsCalculator Build(TIterator iterator, const TModel& model) {
        TRegressionMetricsCalculator rmc;
        for (; iterator.IsValid(); ++iterator) {
            rmc.Add(model.Prediction(*iterator), iterator->Goal, iterator->Weight);
        }
        return rmc;
    }
};
