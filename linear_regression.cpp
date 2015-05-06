#include "linear_model.h"
#include "linear_regression.h"

#include <algorithm>

void TLinearRegressionSolver::Add(const vector<double>& features, const double goal, const double weight) {
    const size_t featuresCount = features.size();

    if (FeatureMeanCalculators.empty()) {
        FeatureMeanCalculators.resize(featuresCount);
        LastMeans.resize(featuresCount);

        LinearizedOLSMatrix.resize(featuresCount * (featuresCount + 1) / 2);
        OLSVector.resize(featuresCount);
    }

    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        LastMeans[featureNumber] = FeatureMeanCalculators[featureNumber].GetMean();
        FeatureMeanCalculators[featureNumber].Add(features[featureNumber], weight);
    }

    size_t olsMatrixElementIdx = 0;
    for (size_t firstFeatureNumber = 0; firstFeatureNumber < featuresCount; ++firstFeatureNumber) {
        for (size_t secondFeatureNumber = firstFeatureNumber; secondFeatureNumber < featuresCount; ++secondFeatureNumber) {
            LinearizedOLSMatrix[olsMatrixElementIdx] +=
                (features[firstFeatureNumber] - LastMeans[firstFeatureNumber]) *
                (features[secondFeatureNumber] - FeatureMeanCalculators[secondFeatureNumber].GetMean()) *
                weight;
            ++olsMatrixElementIdx;
        }
        OLSVector[firstFeatureNumber].Add(features[firstFeatureNumber], goal, weight);
    }
    GoalsMeanCalculator.Add(goal, weight);
}

void TLinearRegressionSolver::Add(const TInstance& instance) {
    Add(instance.Features, instance.Goal, instance.Weight);
}

namespace {
    // LDL matrix decomposition, see http://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
    void LDLDecomposition(const vector<double>& linearizedOLSMatrix,
                          vector<double>& decompositionTrace,
                          vector<vector<double> >& decompositionMatrix);

    vector<double> SolveLower(const vector<vector<double> >& decompositionMatrix,
                              const vector<double>& decompositionTrace,
                              const vector<TCovariationCalculator>& olsVector);
    vector<double> SolveUpper(const vector<vector<double> >& decompositionMatrix,
                              const vector<double>& lowerSolution);
}

TLinearModel TLinearRegressionSolver::Solve() const {
    const size_t featuresCount = OLSVector.size();

    vector<double> decompositionTrace(featuresCount);
    vector<vector<double> > decompositionMatrix(featuresCount, vector<double>(featuresCount));

    LDLDecomposition(LinearizedOLSMatrix, decompositionTrace, decompositionMatrix);

    TLinearModel model;
    model.Coefficients = SolveUpper(decompositionMatrix, SolveLower(decompositionMatrix, decompositionTrace, OLSVector));

    model.Intercept = GoalsMeanCalculator.GetMean();
    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        model.Intercept -= OLSVector[featureNumber].GetFirstValueMean() * model.Coefficients[featureNumber];
    }
    return model;
}

void TSLRSolver::Add(const double feature, const double goal, const double weight) {
    FeaturesCalculator.Add(feature, weight);
    GoalsCalculator.Add(goal, weight);
    ProductCalculator.Add(feature, goal, weight);
}

template <typename TFloatType>
void TSLRSolver::Solve(TFloatType& factor, TFloatType& intercept) const {
    if (!FeaturesCalculator.GetDeviation()) {
        factor = 0.;
        intercept = GoalsCalculator.GetMean();
        return;
    }

    const double regularizationParameter = 0.1;
    factor = ProductCalculator.GetCovariation() / (FeaturesCalculator.GetDeviation() + regularizationParameter);
    intercept = GoalsCalculator.GetMean() - factor * FeaturesCalculator.GetMean();
}

double TSLRSolver::SumSquaredErrors() const {
    double factor, offset;
    Solve(factor, offset);

    return factor * factor * FeaturesCalculator.GetDeviation() - 2 * factor * ProductCalculator.GetCovariation() + GoalsCalculator.GetDeviation();
}

void TBestSLRSolver::Add(const vector<double>& features, const double goal, const double weight) {
    if (SLRSolvers.empty()) {
        SLRSolvers.resize(features.size());
    }

    for (size_t featureNumber = 0; featureNumber < features.size(); ++featureNumber) {
        SLRSolvers[featureNumber].Add(features[featureNumber], goal, weight);
    }
}

void TBestSLRSolver::Add(const TInstance& instance) {
    Add(instance.Features, instance.Goal, instance.Weight);
}

TLinearModel TBestSLRSolver::Solve() const {
    const TSLRSolver* bestSolver = nullptr;
    for (const TSLRSolver& solver : SLRSolvers) {
        if (!bestSolver || solver.SumSquaredErrors() < bestSolver->SumSquaredErrors()) {
            bestSolver = &solver;
        }
    }

    TLinearModel model;
    if (bestSolver) {
        model.Coefficients.resize(SLRSolvers.size());
        bestSolver->Solve(model.Coefficients[bestSolver - &*SLRSolvers.begin()], model.Intercept);
    }

    return model;
}

double TBestSLRSolver::SumSquaredErrors() const {
    if (SLRSolvers.empty()) {
        return 0.;
    }

    double sse = SLRSolvers.begin()->SumSquaredErrors();
    for (const TSLRSolver& solver : SLRSolvers) {
        sse = min(solver.SumSquaredErrors(), sse);
    }
    return sse;
}

namespace {
    bool LDLDecomposition(const vector<double>& linearizedOLSMatrix,
                          const double regularizationThreshold,
                          const double regularizationParameter,
                          vector<double>& decompositionTrace,
                          vector<vector<double> >& decompositionMatrix)
    {
        const size_t featuresCount = decompositionTrace.size();

        size_t olsMatrixElementIdx = 0;
        for (size_t rowNumber = 0; rowNumber < featuresCount; ++rowNumber) {
            double& decompositionTraceElement = decompositionTrace[rowNumber];
            decompositionTraceElement = linearizedOLSMatrix[olsMatrixElementIdx] + regularizationParameter;

            vector<double>& decompositionRow = decompositionMatrix[rowNumber];
            for (size_t i = 0; i < rowNumber; ++i) {
                decompositionTraceElement -= decompositionRow[i] * decompositionRow[i] * decompositionTrace[i];
            }

            if (fabs(decompositionTraceElement) < regularizationThreshold) {
                return false;
            }

            ++olsMatrixElementIdx;
            decompositionRow[rowNumber] = 1.;
            for (size_t columnNumber = rowNumber + 1; columnNumber < featuresCount; ++columnNumber) {
                vector<double>& secondDecompositionRow = decompositionMatrix[columnNumber];
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

    void LDLDecomposition(const vector<double>& linearizedOLSMatrix,
                          vector<double>& decompositionTrace,
                          vector<vector<double> >& decompositionMatrix)
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

    vector<double> SolveLower(const vector<vector<double> >& decompositionMatrix,
                              const vector<double>& decompositionTrace,
                              const vector<TCovariationCalculator>& olsVector)
    {
        const size_t featuresCount = olsVector.size();

        vector<double> solution(featuresCount);
        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            double& solutionElement = solution[featureNumber];
            solutionElement = olsVector[featureNumber].GetCovariation();

            const vector<double>& decompositionRow = decompositionMatrix[featureNumber];
            for (size_t i = 0; i < featureNumber; ++i) {
                solutionElement -= solution[i] * decompositionRow[i];
            }
        }

        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            solution[featureNumber] /= decompositionTrace[featureNumber];
        }

        return solution;
    }

    vector<double> SolveUpper(const vector<vector<double> >& decompositionMatrix,
                              const vector<double>& lowerSolution)
    {
        const size_t featuresCount = lowerSolution.size();

        vector<double> solution(featuresCount);
        for (size_t featureNumber = featuresCount; featureNumber > 0; --featureNumber) {
            double& solutionElement = solution[featureNumber - 1];
            solutionElement = lowerSolution[featureNumber - 1];

            const vector<double>& decompositionRow = decompositionMatrix[featureNumber - 1];
            for (size_t i = featureNumber; i < featuresCount; ++i) {
                solutionElement -= solution[i] * decompositionRow[i];
            }
        }

        return solution;
    }
}
