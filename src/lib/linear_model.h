#pragma once

#include <vector>
#include <numeric>
#include <string>

#include <fstream>

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

    void SaveToFile(const string& modelPath) {
        ofstream modelOut(modelPath);
        modelOut.precision(20);

        modelOut << Coefficients.size() << " ";
        modelOut << Intercept << " ";

        for (const double coefficient : Coefficients) {
            modelOut << coefficient << " ";
        }
    }

    static TLinearModel LoadFromFile(const string& modelPath) {
        ifstream modelIn(modelPath);

        size_t featuresCount;
        modelIn >> featuresCount;

        TLinearModel model;
        model.Coefficients.resize(featuresCount);

        modelIn >> model.Intercept;
        for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
            modelIn >> model.Coefficients[featureIdx];
        }

        return model;
    }
};
