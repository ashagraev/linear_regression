#pragma once

#include "kahan.h"
#include "linear_model.h"
#include "welford.h"

#define DefaultRegularizationParameter (1e-10)

template <typename TStoreType>
class TTypedFastSLRSolver {
private:
    TStoreType SumFeatures = TStoreType();
    TStoreType SumSquaredFeatures = TStoreType();

    TStoreType SumGoals = TStoreType();
    TStoreType SumSquaredGoals = TStoreType();

    TStoreType SumProducts = TStoreType();

    TStoreType SumWeights = TStoreType();
public:
    void Add(const double feature, const double goal, const double weight = 1.) {
        SumFeatures += feature * weight;
        SumSquaredFeatures += feature * feature * weight;

        SumGoals += goal * weight;
        SumSquaredGoals += goal * goal * weight;

        SumProducts += goal * feature * weight;

        SumWeights += weight;
    }

    template <typename TFloatType>
    void Solve(TFloatType& factor, TFloatType& intercept, const double regularizationParameter = DefaultRegularizationParameter) const {
        if (!(double) SumGoals) {
            factor = intercept = TFloatType();
            return;
        }

        double productsDeviation, featuresDeviation;
        SetupSolutionFactors(productsDeviation, featuresDeviation);

        if (!featuresDeviation) {
            factor = TFloatType();
            intercept = (double) SumGoals / (double) SumWeights;
            return;
        }

        factor = productsDeviation / (featuresDeviation + regularizationParameter);
        intercept = (double) SumGoals / (double) SumWeights - factor * (double) SumFeatures / (double) SumWeights;
    }

    double SumSquaredErrors(const double regularizationParameter = DefaultRegularizationParameter) const {
        if (!(double) SumWeights) {
            return 0.;
        }

        const double sumGoalSquaredDeviations = (double) SumSquaredGoals - (double) SumGoals / (double) SumWeights * (double) SumGoals;

        double productsDeviation, featuresDeviation;
        SetupSolutionFactors(productsDeviation, featuresDeviation);
        if (!featuresDeviation) {
            return sumGoalSquaredDeviations;
        }

        const double factor = productsDeviation / (featuresDeviation + regularizationParameter);
        const double sumSquaredErrors = factor * factor * featuresDeviation - 2 * factor * productsDeviation + sumGoalSquaredDeviations;

        return std::max(0., sumSquaredErrors);
    }

    static const std::string Name() {
        return "fast";
    }
private:
    void SetupSolutionFactors(double& productsDeviation, double& featuresDeviation) const {
        if (!(double) SumWeights) {
            productsDeviation = featuresDeviation = 0.;
            return;
        }

        featuresDeviation = (double) SumSquaredFeatures - (double) SumFeatures / (double) SumWeights * (double) SumFeatures;
        if (!featuresDeviation) {
            return;
        }
        productsDeviation = (double)  SumProducts - (double) SumFeatures / (double) SumWeights * (double) SumGoals;
    }
};

class TWelfordSLRSolver {
protected:
    double FeaturesMean = 0.;
    double FeaturesDeviation = 0.;

    double GoalsMean = 0.;
    double GoalsDeviation = 0.;

    TKahanAccumulator SumWeights;

    double Covariation = 0.;
public:
    void Add(const double feature, const double goal, const double weight = 1.);

    template <typename TFloatType>
    void Solve(TFloatType& factor, TFloatType& intercept, const double regularizationParameter = DefaultRegularizationParameter) const {
        if (!FeaturesDeviation) {
            factor = 0.;
            intercept = GoalsMean;
            return;
        }

        factor = Covariation / (FeaturesDeviation + regularizationParameter);
        intercept = GoalsMean - factor * FeaturesMean;
    }

    double SumSquaredErrors(const double regularizationParameter = DefaultRegularizationParameter) const;

    static const std::string Name() {
        return "welford";
    }
};

class TNormalizedWelfordSLRSolver : public TWelfordSLRSolver {
public:
    void Add(const double feature, const double goal, const double weight = 1.);

    static const std::string Name() {
        return "normalized welford";
    }
};

template <typename TSLRSolverType>
class TTypedBestSLRSolver {
private:
    std::vector<TSLRSolverType> SLRSolvers;
public:
    void Add(const std::vector<double>& features, const double goal, const double weight = 1.) {
        if (SLRSolvers.empty()) {
            SLRSolvers.resize(features.size());
        }

        for (size_t featureNumber = 0; featureNumber < features.size(); ++featureNumber) {
            SLRSolvers[featureNumber].Add(features[featureNumber], goal, weight);
        }
    }

    TLinearModel Solve(const double regularizationParameter = DefaultRegularizationParameter) const {
        const TSLRSolverType* bestSolver = nullptr;
        for (const TSLRSolverType& solver : SLRSolvers) {
            if (!bestSolver || solver.SumSquaredErrors(regularizationParameter) < bestSolver->SumSquaredErrors(regularizationParameter)) {
                bestSolver = &solver;
            }
        }

        TLinearModel model;
        if (bestSolver) {
            model.Coefficients.resize(SLRSolvers.size());
            bestSolver->Solve(model.Coefficients[bestSolver - &*SLRSolvers.begin()], model.Intercept, regularizationParameter);
        }

        return model;
    }

    double SumSquaredErrors(const double regularizationParameter = DefaultRegularizationParameter) const {
        if (SLRSolvers.empty()) {
            return 0.;
        }

        double sse = SLRSolvers.begin()->SumSquaredErrors(regularizationParameter);
        for (const TWelfordSLRSolver& solver : SLRSolvers) {
            sse = std::min(solver.SumSquaredErrors(regularizationParameter), sse);
        }
        return sse;
    }

    static const std::string Name() {
        return TSLRSolverType::Name() + " bslr";
    }
};

using TFastSLRSolver = TTypedFastSLRSolver<double>;
using TKahanSLRSolver = TTypedFastSLRSolver<TKahanAccumulator>;

using TFastBestSLRSolver = TTypedBestSLRSolver<TFastSLRSolver>;
using TKahanBestSLRSolver = TTypedBestSLRSolver<TKahanSLRSolver>;
using TWelfordBestSLRSolver = TTypedBestSLRSolver<TWelfordSLRSolver>;
using TNormalizedWelfordBestSLRSolver = TTypedBestSLRSolver<TNormalizedWelfordSLRSolver>;
