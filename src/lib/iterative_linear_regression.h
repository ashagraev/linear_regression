#pragma once

#include "linear_regression.h"

class TIterativeLRSolver {
public:
    struct TOptions {
        size_t EpochsCount = 10;

        size_t MutationsMultiplier = 10;
        double Spread = 0.1;
    };
private:
    TOptions Options;

    TLinearModel LinearModel;

    class TRandomProvider {
    private:
        std::mt19937_64 Mersenne;
        std::normal_distribution<double> Distribution;
    public:
        TRandomProvider(const TOptions& options)
            : Distribution(0., options.Spread)
        {
        }

        double Mutate(const double origin) {
            return origin * (1. + Distribution(Mersenne));
        }
    };

    class TLinearModelAccumulator {
    private:
        TLinearModel LinearModel;
        TKahanAccumulator SumWeights = 0.;
    public:
        void Add(const TLinearModel& other, const double weight) {
            if (LinearModel.Coefficients.empty()) {
                LinearModel.Coefficients.resize(other.Coefficients.size());
            }

            SumWeights += weight;
            for (size_t featureIdx = 0; featureIdx < LinearModel.Coefficients.size(); ++featureIdx) {
                LinearModel.Coefficients[featureIdx] += weight * (other.Coefficients[featureIdx] - LinearModel.Coefficients[featureIdx]) / SumWeights;
            }
            LinearModel.Intercept += weight * (other.Intercept - LinearModel.Intercept) / SumWeights;
        }

        TLinearModel GetLinearModel() const {
            return LinearModel;
        }
    };
public:
    template <typename TIteratorType>
    void Learn(TIteratorType iterator) {
        if (!iterator.IsValid()) {
            return;
        }

        const size_t featuresCount = iterator->Features.size();
        TRandomProvider randomProvider(Options);

        LinearModel.Coefficients.resize(featuresCount);

        {
            std::vector<TWelfordSLRSolver> solvers(featuresCount);
            TMeanCalculator meanGoalCalculator;
            for (TIteratorType it = iterator; it.IsValid(); ++it) {
                const std::vector<double>& features = it->Features;
                const double goal = it->Goal;

                for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
                    solvers[featureIdx].Add(features[featureIdx], goal, it->Weight);
                }
                meanGoalCalculator.Add(goal, it->Weight);
            }


            for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
                double factor, offset;
                solvers[featureIdx].Solve(factor, offset);
                LinearModel.Coefficients[featureIdx] = factor;
            }
            LinearModel.Intercept = meanGoalCalculator.GetMean();
        }

        std::cerr << MeanSquaredError(iterator, LinearModel) << std::endl;
        for (size_t epoch = 0; epoch < Options.EpochsCount * 100; ++epoch) {
            TLinearModelAccumulator weightedAccumulator;
            TLinearModelAccumulator avgAccumulator;

            const size_t iterationsCount = featuresCount * Options.MutationsMultiplier;
            for (size_t iteration = 0; iteration < iterationsCount; ++iteration) {
                const TLinearModel nextNodel = NextModel(randomProvider);
                const double mse = MeanSquaredError(iterator, nextNodel);

                weightedAccumulator.Add(nextNodel, mse);
                avgAccumulator.Add(nextNodel, 1.);
            }

            const TLinearModel weightedModel = weightedAccumulator.GetLinearModel();
            const TLinearModel avgModel(avgAccumulator.GetLinearModel());

            double coeff = 100.;

            const double originalMSE = MeanSquaredError(iterator, LinearModel);

            for (size_t i = 0; i < 10; ++i) {
                TLinearModel newModel = LinearModel;
                for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
                    newModel.Coefficients[featureIdx] += coeff * (avgModel.Coefficients[featureIdx] - weightedModel.Coefficients[featureIdx]);
                }
                newModel.Intercept += coeff * (avgModel.Intercept - weightedModel.Intercept);

                const double mse = MeanSquaredError(iterator, newModel);
                if (mse < originalMSE) {
                    LinearModel = newModel;
                    break;
                }

                coeff /= 2;
            }

            std::cerr << MeanSquaredError(iterator, LinearModel) << std::endl;
        }

        TWelfordLRSolver goodSolver;
        for (TIteratorType it = iterator; it.IsValid(); ++it) {
            goodSolver.Add(it->Features, it->Goal, it->Weight);
        }
        const TLinearModel goodModel = goodSolver.Solve();
        std::cerr << MeanSquaredError(iterator, goodModel) << std::endl;
    }
private:
    TLinearModel NextModel(TRandomProvider& randomProvider) const {
        TLinearModel nextModel(LinearModel);
        for (double& param : nextModel.Coefficients) {
            param = randomProvider.Mutate(param);
        }
        nextModel.Intercept = randomProvider.Mutate(nextModel.Intercept);
        return nextModel;
    }

    template <typename TIteratorType>
    double MeanSquaredError(TIteratorType iterator, const TLinearModel& linearModel) const {
        TMeanCalculator avgSquaredError;
        for (; iterator.IsValid(); ++iterator) {
            const double prediction = linearModel.Prediction(iterator->Features);
            const double goal = iterator->Goal;

            const double error = goal - prediction;

            avgSquaredError.Add(error * error, iterator->Weight);
        }

        return avgSquaredError.GetMean();
    }
};
