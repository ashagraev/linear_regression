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

    void SaveToFile(const std::string& modelPath);
    static TLinearModel LoadFromFile(const std::string& modelPath);

    template <typename T>
    double Prediction(const std::vector<T>& features) const {
        return inner_product(Coefficients.begin(), Coefficients.end(), features.begin(), Intercept);
    }
};

template <typename TSolver>
TLinearModel Solve(const TPool::TCVIterator& iterator) {
    TSolver solver;
    for (; iterator.IsValid(); iterator.Advance()) {
        solver.Add(iterator->Features, iterator->Goal, iterator->Weight);
    }
    return solver.Solve();
}
