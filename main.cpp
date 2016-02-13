#include "linear_regression.h"
#include "pool.h"

#include <iostream>
#include <time.h>

template <typename TSolver>
TLinearModel Solve(const TPool& pool) {
    TSolver solver;
    for (const TInstance& instance : pool) {
        solver.Add(instance.Features, instance.Goal, instance.Weight);
    }
    return solver.Solve();
}

int main(int argc, const char** argv) {
    (void) argc;
    string mode = argv[1];

    string featuresFilePath = argv[2];
    string modelFilePath = argv[3];

    TPool pool;
    pool.ReadFromFeatures(featuresFilePath);

    if (mode == "learn") {
        string learningMode = argv[4];

        TLinearModel linearModel;
        if (learningMode == "fast_bslr") {
            linearModel = Solve<TFastBestSLRSolver>(pool);
        }
        if (learningMode == "kahan_bslr") {
            linearModel = Solve<TKahanBestSLRSolver>(pool);
        }
        if (learningMode == "welford_bslr") {
            linearModel = Solve<TWelfordBestSLRSolver>(pool);
        }
        if (learningMode == "fast_lr") {
            linearModel = Solve<TFastLRSolver>(pool);
        }
        if (learningMode == "welford_lr") {
            linearModel = Solve<TWelfordLRSolver>(pool);
        }

        linearModel.SaveToFile(modelFilePath);
    }

    if (mode == "predict") {
        string predictionsPath = argv[4];

        ofstream predictionsOut(predictionsPath);
        predictionsOut.precision(20);

        TLinearModel linearModel = TLinearModel::LoadFromFile(modelFilePath);

        for (const TInstance& instance : pool) {
            predictionsOut << linearModel.Prediction(instance.Features) << "\n";
        }
    }

    return 0;
}
