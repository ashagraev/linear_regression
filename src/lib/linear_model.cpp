#include "linear_model.h"

TLinearModel::TLinearModel(size_t featuresCount /*= 0*/)
    : Coefficients(featuresCount)
    , Intercept(0.)
{
}

void TLinearModel::SaveToFile(const std::string& modelPath) const {
    std::ofstream modelOut(modelPath);
    modelOut.precision(20);

    modelOut << (unsigned int)Coefficients.size() << " ";
    modelOut << Intercept << " ";

    for (const double coefficient : Coefficients) {
        modelOut << coefficient << " ";
    }
}

TLinearModel TLinearModel::LoadFromFile(const std::string& modelPath) {
    std::ifstream modelIn(modelPath);

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
