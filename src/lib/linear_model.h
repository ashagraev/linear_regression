#pragma once

#include "pool.h"

#include <vector>
#include <numeric>
#include <string>

#include <fstream>

struct TLinearModel {
    std::vector<double> Coefficients;
    double Intercept;

    explicit TLinearModel(size_t featuresCount = 0);

    void SaveToFile(const std::string& modelPath) const;
    static TLinearModel LoadFromFile(const std::string& modelPath);

    template <typename T>
    double Prediction(const std::vector<T>& features) const {
        return std::inner_product(Coefficients.begin(), Coefficients.end(), features.begin(), Intercept);
    }

    double Prediction(const TInstance& instance) const {
        return Prediction(instance.Features);
    }
};

template <typename TSolver, typename TIterator>
TLinearModel Solve(TIterator iterator, double* sumSquaredErrors = nullptr) {
    TSolver solver;
    for (; iterator.IsValid(); ++iterator) {
        solver.Add(iterator->Features, iterator->Goal, iterator->Weight);
    }
    if (sumSquaredErrors) {
        *sumSquaredErrors = solver.SumSquaredErrors();
    }
    return solver.Solve();
}
