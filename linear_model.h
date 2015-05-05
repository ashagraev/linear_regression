#pragma once

#include <vector>
#include <numeric>

using namespace std;

struct TLinearModel {
    vector<double> Coefficients;
    double Intercept;

    explicit TLinearModel(size_t featuresCount = 0)
        : Coefficients(featuresCount)
        , Intercept(0.)
    {
    }

    template <typename T>
    double Prediction(const vector<T>& features) const {
        return inner_product(Coefficients.begin(), Coefficients.end(), features.begin(), Intercept);
    }
};
