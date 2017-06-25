#include "simple_linear_regression.h"

#include <algorithm>
#include <cmath>

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
