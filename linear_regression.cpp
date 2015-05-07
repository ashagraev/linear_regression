#include "linear_model.h"
#include "linear_regression.h"

#include <algorithm>

namespace {
    inline void AddFeaturesProduct(const double weight, const vector<double>& features, vector<double>& linearizedOLSTriangleMatrix);

    vector<double> Solve(const vector<double>& olsMatrix, const vector<double>& olsVector);

    double SumSquaredErrors(const vector<double>& olsMatrix,
                            const vector<double>& olsVector,
                            const vector<double>& solution,
                            const double goalsDeviation);
}


void TFastLinearRegressionSolver::Add(const vector<double>& features, const double goal, const double weight) {
    const size_t featuresCount = features.size();

    if (LinearizedOLSMatrix.empty()) {
        LinearizedOLSMatrix.resize((featuresCount + 1) * (featuresCount + 2) / 2);
        OLSVector.resize(featuresCount + 1);
    }

    AddFeaturesProduct(weight, features, LinearizedOLSMatrix);

    const double weightedGoal = goal * weight;
    vector<double>::iterator olsVectorElement = OLSVector.begin();
    for (const double feature : features) {
        *olsVectorElement += feature * weightedGoal;
        ++olsVectorElement;
    }
    *olsVectorElement += weightedGoal;

    SumSquaredGoals += goal * goal * weight;
}

void TLinearRegressionSolver::Add(const vector<double>& features, const double goal, const double weight) {
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

    vector<double>::iterator olsMatrixElement = LinearizedOLSMatrix.begin();

    vector<double>::iterator lastMean = LastMeans.begin();
    vector<double>::iterator newMean = NewMeans.begin();
    for (; lastMean != LastMeans.end(); ++lastMean, ++newMean) {
        for (vector<double>::iterator secondFeatureMean = newMean; secondFeatureMean != NewMeans.end(); ++secondFeatureMean) {
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

TLinearModel TFastLinearRegressionSolver::Solve() const {
    TLinearModel linearModel;
    linearModel.Coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);

    if (!linearModel.Coefficients.empty()) {
        linearModel.Intercept = linearModel.Coefficients.back();
        linearModel.Coefficients.pop_back();
    }

    return linearModel;
}

TLinearModel TLinearRegressionSolver::Solve() const {
    TLinearModel model;
    model.Coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);
    model.Intercept = GoalsMean;

    const size_t featuresCount = OLSVector.size();
    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        model.Intercept -= FeatureMeans[featureNumber] * model.Coefficients[featureNumber];
    }

    return model;
}

double TFastLinearRegressionSolver::SumSquaredErrors() const {
    vector<double> coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);
    return ::SumSquaredErrors(LinearizedOLSMatrix, OLSVector, coefficients, SumSquaredGoals);
}

double TLinearRegressionSolver::SumSquaredErrors() const {
    vector<double> coefficients = ::Solve(LinearizedOLSMatrix, OLSVector);
    return ::SumSquaredErrors(LinearizedOLSMatrix, OLSVector, coefficients, GoalsDeviation);
}

void TSLRSolver::Add(const double feature, const double goal, const double weight) {
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

double TSLRSolver::SumSquaredErrors(const double regularizationParameter) const {
    double factor, offset;
    Solve(factor, offset, regularizationParameter);

    return factor * factor * FeaturesDeviation - 2 * factor * Covariation + GoalsDeviation;
}

namespace {
    // LDL matrix decomposition, see http://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2
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
                              const vector<double>& olsVector)
    {
        const size_t featuresCount = olsVector.size();

        vector<double> solution(featuresCount);
        for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
            double& solutionElement = solution[featureNumber];
            solutionElement = olsVector[featureNumber];

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

    vector<double> Solve(const vector<double>& olsMatrix, const vector<double>& olsVector) {
        const size_t featuresCount = olsVector.size();

        vector<double> decompositionTrace(featuresCount);
        vector<vector<double> > decompositionMatrix(featuresCount, vector<double>(featuresCount));

        LDLDecomposition(olsMatrix, decompositionTrace, decompositionMatrix);

        return SolveUpper(decompositionMatrix, SolveLower(decompositionMatrix, decompositionTrace, olsVector));
    }

    double SumSquaredErrors(const vector<double>& olsMatrix,
                            const vector<double>& olsVector,
                            const vector<double>& solution,
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

    inline void AddFeaturesProduct(const double weight, const vector<double>& features, vector<double>& linearizedTriangleMatrix) {
        vector<double>::const_iterator leftFeature = features.begin();
        vector<double>::iterator matrixElement = linearizedTriangleMatrix.begin();
        for (; leftFeature != features.end(); ++leftFeature, ++matrixElement) {
            const double weightedFeature = weight * *leftFeature;
            vector<double>::const_iterator rightFeature = leftFeature;
            for (; rightFeature != features.end(); ++rightFeature, ++matrixElement) {
                *matrixElement += weightedFeature * *rightFeature;
            }
            *matrixElement += weightedFeature;
        }
        linearizedTriangleMatrix.back() += weight;
    }
}
