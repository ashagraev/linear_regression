#pragma once

#include "linear_model.h"
#include "welford.h"

class TFastLRSolver {
private:
    TKahanAccumulator SumSquaredGoals;

    std::vector<double> LinearizedOLSMatrix;
    std::vector<double> OLSVector;
public:
    void Add(const std::vector<double>& features, const double goal, const double weight = 1.);
    TLinearModel Solve() const;
    double SumSquaredErrors() const;
};

class TWelfordLRSolver {
private:
    double GoalsMean = 0.;
    double GoalsDeviation = 0.;

    std::vector<double> FeatureMeans;
    std::vector<double> FeatureWeightedDeviationFromLastMean;
    std::vector<double> FeatureDeviationFromNewMean;
    std::vector<double> LinearizedOLSMatrix;

    std::vector<double> OLSVector;

    TKahanAccumulator SumWeights;
public:
    void Add(const std::vector<double>& features, const double goal, const double weight = 1.);
    TLinearModel Solve() const;
    double SumSquaredErrors() const;
private:
    bool PrepareMeans(const std::vector<double>& features, const double weight);
};

class TPreciseWelfordLRSolver {
private:
    double GoalsMean = 0.;
    double GoalsDeviation = 0.;

    std::vector<double> FeatureMeans;
    std::vector<double> FeatureWeightedDeviationFromLastMean;
    std::vector<double> FeatureDeviationFromNewMean;
    std::vector<double> LinearizedOLSMatrix;

    std::vector<double> OLSVector;

    TKahanAccumulator SumWeights;
public:
    void Add(const std::vector<double>& features, const double goal, const double weight = 1.);
    TLinearModel Solve() const;
    double MeanSquaredError() const;
private:
    bool PrepareMeans(const std::vector<double>& features, const double weight);
};
