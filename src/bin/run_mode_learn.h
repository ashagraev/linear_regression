#pragma once

#include "args.h"
#include "timer.h"

#include "../lib/linear_regression.h"
#include "../lib/simple_linear_regression.h"

#include "../lib/metrics.h"
#include "../lib/pool.h"

#include <time.h>

template <typename TIteratorType>
TLinearModel Solve(TIteratorType iterator, const std::string& learningMode) {
    TLinearModel linearModel;
    if (learningMode == "fast_bslr") {
        linearModel = Solve<TFastBestSLRSolver>(iterator);
    }
    if (learningMode == "kahan_bslr") {
        linearModel = Solve<TKahanBestSLRSolver>(iterator);
    }
    if (learningMode == "welford_bslr") {
        linearModel = Solve<TWelfordBestSLRSolver>(iterator);
    }
    if (learningMode == "normalized_welford_bslr") {
        linearModel = Solve<TNormalizedWelfordBestSLRSolver>(iterator);
    }
    if (learningMode == "fast_lr") {
        linearModel = Solve<TFastLRSolver>(iterator);
    }
    if (learningMode == "welford_lr") {
        linearModel = Solve<TWelfordLRSolver>(iterator);
    }
    if (learningMode == "normalized_welford_lr") {
        linearModel = Solve<TNormalizedWelfordLRSolver>(iterator);
    }
    return linearModel;
}

int DoLearn(int argc, const char** argv) {
    std::string featuresPath;
    std::string modelPath;

    std::string learningMode = "welford_lr";

    {
        TArgsParser argsParser;

        argsParser.AddHandler("features", &featuresPath, "features file path").Required();

        argsParser.AddHandler("model", &modelPath, "resulting model path").Optional();
        argsParser.AddHandler("method", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr, normalized_welford_lr").Optional();

        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    {
        TTimer timer("pool read in");
        pool.ReadFromFeatures(featuresPath);
    }

    TPool::TSimpleIterator learnIterator(pool);
    TLinearModel linearModel;
    {
        TTimer timer("model learned in");
        linearModel = Solve(learnIterator, learningMode);
    }

    if (!modelPath.empty()) {
        linearModel.SaveToFile(modelPath);
    }

    TRegressionMetricsCalculator rmc = TRegressionMetricsCalculator::Build(learnIterator, linearModel);
    std::cout << "learn rmse: " << rmc.RMSE() << std::endl;
    std::cout << "learn R^2:  " << rmc.DeterminationCoefficient() << std::endl;

    return 0;
}
