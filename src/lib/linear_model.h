#pragma once

#include "pool.h"

#include <vector>
#include <numeric>
#include <string>

#include <fstream>

using namespace std;

struct TLinearModel {
    vector<double> Coefficients;
    double Intercept;

    explicit TLinearModel(size_t featuresCount = 0);

    void SaveToFile(const string& modelPath);
    static TLinearModel LoadFromFile(const string& modelPath);

    template <typename T>
    double Prediction(const vector<T>& features) const {
        return inner_product(Coefficients.begin(), Coefficients.end(), features.begin(), Intercept);
    }
};

template <typename TSolver>
TLinearModel Solve(const TPool& pool) {
    TSolver solver;
    for (const TInstance& instance : pool) {
        solver.Add(instance.Features, instance.Goal, instance.Weight);
    }
    return solver.Solve();
}