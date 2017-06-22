#include "linear_model.h"
#include "linear_regression.h"

#include <algorithm>
#include <cmath>

namespace NLinearRegressionInner {
    inline void AddFeaturesProduct(const double weight, const std::vector<double>& features, std::vector<double>& linearizedOLSTriangleMatrix);

    std::vector<double> Solve(const std::vector<double>& olsMatrix, const std::vector<double>& olsVector);

    double SumSquaredErrors(const std::vector<double>& olsMatrix,
                            const std::vector<double>& olsVector,
                            const std::vector<double>& solution,
                            const double goalsDeviation);
}


void TFastLRSolver::Add(const std::vector<double>& features, const double goal, const double weight) {
    const size_t featuresCount = features.size();

    if (LinearizedOLSMatrix.empty()) {
        LinearizedOLSMatrix.resize((featuresCount + 1) * (featuresCount + 2) / 2);
        OLSVector.resize(featuresCount + 1);
    }

    NLinearRegressionInner::AddFeaturesProduct(weight, features, LinearizedOLSMatrix);

    const double weightedGoal = goal * weight;
    std::vector<double>::iterator olsVectorElement = OLSVector.begin();
    for (const double feature : features) {
        *olsVectorElement += feature * weightedGoal;
        ++olsVectorElement;
    }
    *olsVectorElement += weightedGoal;

    SumSquaredGoals += goal * goal * weight;
}

void TWelfordLRSolver::Add(const std::vector<double>& features, const double goal, const double weight) {
    const size_t featuresCount = features.size();

    if (FeatureMeans.empty()) {
        FeatureMeans.resize(featuresCount);
        LastMeans.resize(featuresCount);
        NewMeans.resize(featuresCount);

        LinearizedOLSMatrix.resize(featuresCount * (featuresCount + 1) / 2);
        OLSVector.resize(featuresCount);
    }

    SumWeights += weight;
    if (!SumWeights) {
        return;
    }

    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        const double feature = features[featureNumber];
        double& featureMean = FeatureMeans[featureNumber];

        LastMeans[featureNumber] = weight * (feature - featureMean);
        featureMean += weight * (feature - featureMean) / SumWeights;
        NewMeans[featureNumber] = feature - featureMean;;
    }

    std::vector<double>::iterator olsMatrixElement = LinearizedOLSMatrix.begin();

    std::vector<double>::iterator lastMean = LastMeans.begin();
    std::vector<double>::iterator newMean = NewMeans.begin();
    for (; lastMean != LastMeans.end(); ++lastMean, ++newMean) {
        for (std::vector<double>::iterator secondFeatureMean = newMean; secondFeatureMean != NewMeans.end(); ++secondFeatureMean) {
            *olsMatrixElement++ += *lastMean * *secondFeatureMean;
        }
    }

    for (size_t firstFeatureNumber = 0; firstFeatureNumber < features.size(); ++firstFeatureNumber) {
        OLSVector[firstFeatureNumber] += weight * (features[firstFeatureNumber] - FeatureMeans[firstFeatureNumber]) * (goal - GoalsMean);
    }

    const double oldGoalsMean = GoalsMean;
    GoalsMean += weight * (goal - GoalsMean) / SumWeights;
    GoalsDeviation += weight * (goal - oldGoalsMean) * (goal - GoalsMean);
}

TLinearModel TFastLRSolver::Solve() const {
    TLinearModel linearModel;
    linearModel.Coefficients = NLinearRegressionInner::Solve(LinearizedOLSMatrix, OLSVector);

    if (!linearModel.Coefficients.empty()) {
        linearModel.Intercept = linearModel.Coefficients.back();
        linearModel.Coefficients.pop_back();
    }

    return linearModel;
}

TLinearModel TWelfordLRSolver::Solve() const {
    TLinearModel model;
    model.Coefficients = NLinearRegressionInner::Solve(LinearizedOLSMatrix, OLSVector);
    model.Intercept = GoalsMean;

    const size_t featuresCount = OLSVector.size();
    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        model.Intercept -= FeatureMeans[featureNumber] * model.Coefficients[featureNumber];
    }

    return model;
}

double TFastLRSolver::SumSquaredErrors() const {
    const std::vector<double> coefficients = NLinearRegressionInner::Solve(LinearizedOLSMatrix, OLSVector);
    return NLinearRegressionInner::SumSquaredErrors(LinearizedOLSMatrix, OLSVector, coefficients, SumSquaredGoals);
}

double TWelfordLRSolver::SumSquaredErrors() const {
    const std::vector<double> coefficients = NLinearRegressionInner::Solve(LinearizedOLSMatrix, OLSVector);
    return NLinearRegressionInner::SumSquaredErrors(LinearizedOLSMatrix, OLSVector, coefficients, GoalsDeviation);
}

void TWelfordSLRSolver::Add(const double feature, const double goal, const double weight) {
    SumWeights += weight;
    if (!SumWeights) {
        return;
    }

    const double weightedFeatureDiff = weight * (feature - FeaturesMean);
    const double weightedGoalDiff = weight * (goal - GoalsMean);

    FeaturesMean += weightedFeatureDiff / SumWeights;
    FeaturesDeviation += weightedFeatureDiff * (feature - FeaturesMean);

    GoalsMean += weightedGoalDiff / SumWeights;
    GoalsDeviation += weightedGoalDiff * (goal - GoalsMean);

    Covariation += weightedFeatureDiff * (goal - GoalsMean);
}

double TWelfordSLRSolver::SumSquaredErrors(const double regularizationParameter) const {
    double factor, offset;
    Solve(factor, offset, regularizationParameter);

    return factor * factor * FeaturesDeviation - 2 * factor * Covariation + GoalsDeviation;
}

namespace NLinearRegressionInner {
    // LDL matrix decomposition, see http://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
    bool LDLDecomposition(const std::vector<double>& linearizedOLSMatrix,
                          const double regularizationThreshold,
                          const double regularizationParameter,
                          std::vector<double>& decompositionTrace,
                          std::vector<std::vector<double> >& decompositionMatrix)
    {
        const size_t featuresCount = decompositionTrace.size();

        size_t olsMatrixElementIdx = 0;
        for (size_t rowNumber = 0; rowNumber < featuresCount; ++rowNumber) {
            double& decompositionTraceElement = decompositionTrace[rowNumber];
            decompositionTraceElement = linearizedOLSMatrix[olsMatrixElementIdx] + regularizationParameter;

            std::vector<double>& decompositionRow = decompositionMatrix[rowNumber];
            for (size_t i = 0; i < rowNumber; ++i) {
                decompositionTraceElement -= decompositionRow[i] * decompositionRow[i] * decompositionTrace[i];
            }

            if (fabs(decompositionTraceElement) < regularizationThreshold) {
                return false;
            }

            ++olsMatrixElementIdx;
            decompositionRow[rowNumber] = 1.;
            for (size_t columnNumber = rowNumber + 1; columnNumber < featuresCount; ++columnNumber) {
                std::vector<double>& secondDecompositionRow = decompositionMatrix[columnNumber];
                double& decompositionMatrixElement = secondDecompositionRow[rowNumber];

                decompositionMatrixElement = linearizedOLSMatrix[olsMatrixElementIdx];

                for (size_t j = 0; j < rowNumber; ++j) {
                    decompositionMatrixElement -= decompositionRow[j] * secondDecompositionRow[j] * decompositionTrace[j];
                }

                decompositionMatrixElement /= decompositionTraceElement;

                decompositionRow[columnNumber] = decompositionMatrixElement;
                ++olsMatrixElementIdx;
            }
        }

        return true;
    }

    void LDLDecomposition(const std::vector<double>& linearizedOLSMatrix,
                          std::vector<double>& decompositionTrace,
                          std::vector<std::vector<double> >& decompositionMatrix)
    {
        const double regularizationThreshold = 1e-5;
        double regularizationParameter = 0.;

        while (!LDLDecomposition(linearizedOLSMatrix,
                                 regularizationThreshold,
                                 regularizationParameter,
                                 decompositionTrace,
                                 decompositionMatrix))
        {
            regularizationParameter = regularizationParameter ? 2 * regularizationParameter : 1e-5;
        }
    }

    std::vector<double> SolveLower(const std::vector<std::vector<double> >& decompositionMatrix,
                              const std::vector<double>& decompositionTrace,
                              const std::vector<double>& olsVector)
    {
        const size_t featuresCount = olsVector.size();

        std::vector<double> solution(featuresCount);
        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            double& solutionElement = solution[featureNumber];
            solutionElement = olsVector[featureNumber];

            const std::vector<double>& decompositionRow = decompositionMatrix[featureNumber];
            for (size_t i = 0; i < featureNumber; ++i) {
                solutionElement -= solution[i] * decompositionRow[i];
            }
        }

        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            solution[featureNumber] /= decompositionTrace[featureNumber];
        }

        return solution;
    }

    std::vector<double> SolveUpper(const std::vector<std::vector<double> >& decompositionMatrix,
                              const std::vector<double>& lowerSolution)
    {
        const size_t featuresCount = lowerSolution.size();

        std::vector<double> solution(featuresCount);
        for (size_t featureNumber = featuresCount; featureNumber > 0; --featureNumber) {
            double& solutionElement = solution[featureNumber - 1];
            solutionElement = lowerSolution[featureNumber - 1];

            const std::vector<double>& decompositionRow = decompositionMatrix[featureNumber - 1];
            for (size_t i = featureNumber; i < featuresCount; ++i) {
                solutionElement -= solution[i] * decompositionRow[i];
            }
        }

        return solution;
    }

    std::vector<double> Solve(const std::vector<double>& olsMatrix, const std::vector<double>& olsVector) {
        const size_t featuresCount = olsVector.size();

        std::vector<double> decompositionTrace(featuresCount);
        std::vector<std::vector<double> > decompositionMatrix(featuresCount, std::vector<double>(featuresCount));

        LDLDecomposition(olsMatrix, decompositionTrace, decompositionMatrix);

        return SolveUpper(decompositionMatrix, SolveLower(decompositionMatrix, decompositionTrace, olsVector));
    }

    double SumSquaredErrors(const std::vector<double>& olsMatrix,
                            const std::vector<double>& olsVector,
                            const std::vector<double>& solution,
                            const double goalsDeviation)
    {
        const size_t featuresCount = olsVector.size();

        double sumSquaredErrors = goalsDeviation;
        size_t olsMatrixElementIdx = 0;
        for (size_t i = 0; i < featuresCount; ++i) {
            sumSquaredErrors += olsMatrix[olsMatrixElementIdx] * solution[i] * solution[i];
            ++olsMatrixElementIdx;
            for (size_t j = i + 1; j < featuresCount; ++j) {
                sumSquaredErrors += 2 * olsMatrix[olsMatrixElementIdx] * solution[i] * solution[j];
                ++olsMatrixElementIdx;
            }
            sumSquaredErrors -= 2 * solution[i] * olsVector[i];
        }
        return sumSquaredErrors;
    }

    inline void AddFeaturesProduct(const double weight, const std::vector<double>& features, std::vector<double>& linearizedTriangleMatrix) {
        std::vector<double>::const_iterator leftFeature = features.begin();
        std::vector<double>::iterator matrixElement = linearizedTriangleMatrix.begin();
        for (; leftFeature != features.end(); ++leftFeature, ++matrixElement) {
            const double weightedFeature = weight * *leftFeature;
            std::vector<double>::const_iterator rightFeature = leftFeature;
            for (; rightFeature != features.end(); ++rightFeature, ++matrixElement) {
                *matrixElement += weightedFeature * *rightFeature;
            }
            *matrixElement += weightedFeature;
        }
        linearizedTriangleMatrix.back() += weight;
    }
}
