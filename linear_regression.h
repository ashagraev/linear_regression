#pragma once

#include "linear_model.h"
#include "pool.h"
#include "welford.h"

#include <vector>

using namespace std;

class TLinearRegressionSolver {
private:
    TMeanCalculator GoalsMeanCalculator;

    vector<TCovariationCalculator> LinearizedTriangleOLSMatrix;
    vector<TCovariationCalculator> OLSVector;
public:
    void Add(const vector<double>& features, const double goal, const double weight);
    void Add(const TInstance& instance);

    TLinearModel Solve() const;
};

class TSLRSolver {
private:
    TDeviationCalculator FeaturesCalculator;
    TDeviationCalculator GoalsCalculator;

    TCovariationCalculator ProductCalculator;
public:
    void Add(const double feature, const double goal, const double weight);

    template <typename TFloatType>
    void Solve(TFloatType& factor, TFloatType& intercept) const;

    double SumSquaredErrors() const;
};

class TBestSLRSolver {
private:
    vector<TSLRSolver> SLRSolvers;
public:
    void Add(const vector<double>& features, const double goal, const double weight);
    void Add(const TInstance& instance);

    TLinearModel Solve() const;

    double SumSquaredErrors() const;
};
