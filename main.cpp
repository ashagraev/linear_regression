#include "linear_regression.h"
#include "pool.h"

#include <iostream>
#include <time.h>

template <typename TLRSolver>
void QualityBenchmark(const TPool& originalPool) {
    auto measure = [&](const double injureFactor, const double injureOffset) {
        TPool injuredPool = originalPool.InjurePool(injureFactor, injureOffset);

        static const size_t runsCount = 10;
        static const size_t foldsCount = 10;

        TMeanCalculator determinationCoefficientCalculator;

        TPool::TCVIterator learnIterator = injuredPool.CrossValidationIterator(foldsCount, TPool::LearnIterator);
        TPool::TCVIterator testIterator = injuredPool.CrossValidationIterator(foldsCount, TPool::TestIterator);

        for (size_t runNumber = 0; runNumber < runsCount; ++runNumber) {
            for (size_t foldNumber = 0; foldNumber < foldsCount; ++foldNumber) {
                learnIterator.ResetShuffle();
                learnIterator.SetTestFold(foldNumber);
                testIterator.ResetShuffle();
                testIterator.SetTestFold(foldNumber);

                TLRSolver solver;
                for (; learnIterator.IsValid(); ++learnIterator) {
                    solver.Add(learnIterator->Features, learnIterator->Goal, learnIterator->Weight);
                }
                TLinearModel model = solver.Solve();

                TDeviationCalculator goalsCalculator;
                TKahanAccumulator errorsCalculator;
                for (; testIterator.IsValid(); ++testIterator) {
                    const double prediction = model.Prediction(testIterator->Features);
                    const double goal = testIterator->Goal;
                    const double weight = testIterator->Weight;
                    const double error = goal - prediction;

                    goalsCalculator.Add(goal, weight);
                    errorsCalculator += error * error * weight;
                }

                const double determinationCoefficient = 1 - errorsCalculator / goalsCalculator.GetDeviation();
                determinationCoefficientCalculator.Add(determinationCoefficient);
            }
        }

        return determinationCoefficientCalculator.GetMean();
    };

    cout << typeid(TLRSolver()).name() << ":\n";
    printf("\tbase    : %.10lf\n", measure(1., 0.));
    printf("\tinjure1 : %.10lf\n", measure(1e-1, 1e+1));
    printf("\tinjure2 : %.10lf\n", measure(1e-3, 1e+4));
    printf("\tinjure3 : %.10lf\n", measure(1e-3, 1e+5));
    printf("\tinjure4 : %.10lf\n", measure(1e-3, 1e+6));
    printf("\tinjure5 : %.10lf\n", measure(1e-4, 1e+6));
    printf("\tinjure6 : %.10lf\n", measure(1e-4, 1e+7));
    cout << endl;
}

template <typename TLRSolver>
void SpeedBenchmark(const TPool& originalPool) {
    for (const int factorsFraction : {1,2,4,8,16}) {
        TPool modifiedPool(originalPool);
        const size_t featuresCount = (size_t) originalPool[0].Features.size() / factorsFraction;

        for (TInstance& instance : modifiedPool) {
            instance.Features.erase(instance.Features.begin() + featuresCount, instance.Features.end());
        }

        size_t startTime = clock();

        static const size_t runsCount = 1000;
        for (size_t runNumber = 0; runNumber < runsCount; ++runNumber) {
            TLRSolver solver;
            for (const TInstance& instance : modifiedPool) {
                solver.Add(instance.Features, instance.Goal, instance.Weight);
            }
            TLinearModel model = solver.Solve();
        }

        size_t endTime = clock();
        printf("1/%d\t%.5lf: %s\n", factorsFraction, (double) (endTime - startTime) / CLOCKS_PER_SEC / runsCount, typeid(TLRSolver()).name());
    }
    cout << "\n";
}

int main(int argc, const char** argv) {
    for (int taskNumber = 1; taskNumber < argc; ++taskNumber) {
        TPool pool;
        pool.ReadFromFeatures(argv[taskNumber]);

        cout << argv[taskNumber] << ":" << endl;
        QualityBenchmark<TFastBestSLRSolver>(pool);
        QualityBenchmark<TKahanBestSLRSolver>(pool);
        QualityBenchmark<TBestSLRSolver>(pool);

        QualityBenchmark<TLinearRegressionSolver>(pool);
        QualityBenchmark<TFastLinearRegressionSolver>(pool);

        SpeedBenchmark<TFastBestSLRSolver>(pool);
        SpeedBenchmark<TKahanBestSLRSolver>(pool);
        SpeedBenchmark<TBestSLRSolver>(pool);

        SpeedBenchmark<TLinearRegressionSolver>(pool);
        SpeedBenchmark<TFastLinearRegressionSolver>(pool);
    }

    return 0;
}
