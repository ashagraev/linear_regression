#pragma once

#include "args.h"
#include "run_mode_learn.h"

#include <iostream>

struct TCrossValidationResult {
    double MeanDeterminationCoefficient;
    double LearningTimeInSeconds;
};

TCrossValidationResult CrossValidation(
    const TPool& pool,
    const size_t foldsCount,
    const size_t runsCount,
    const std::string& learningMode,
    const std::string verboseMode,
    const bool verbose) {
    double learningTime = 0;

    TPool::TCVIterator learnIterator = pool.LearnIterator(foldsCount);
    TPool::TCVIterator testIterator = pool.TestIterator(foldsCount);

    TMeanCalculator meanDCCalculator;
    for (size_t runIdx = 0; runIdx < runsCount; ++runIdx) {
        learnIterator.ResetShuffle();
        testIterator.ResetShuffle();

        TMeanCalculator meanFoldDCCalculator;
        for (size_t fold = 0; fold < foldsCount; ++fold) {
            learnIterator.SetTestFold(fold);
            testIterator.SetTestFold(fold);

            TLinearModel linearModel;
            {
                TTimer timer;
                linearModel = Solve(learnIterator, learningMode);
                learningTime += timer.GetSecondsPassed();
            }
            const double determinationCoefficient = TRegressionMetricsCalculator::Build(testIterator, linearModel).DeterminationCoefficient();

            if (verbose && verboseMode == "folds") {
                std::cout << "    ";
                if (runsCount > 1) {
                    std::cout << "    run #" << runIdx << ", ";
                }
                std::cout << "fold #" << fold << ": R^2 = " << determinationCoefficient << std::endl;
            }

            meanFoldDCCalculator.Add(determinationCoefficient);
        }

        if (verbose && verboseMode != "overall") {
            if (runsCount > 1) {
                std::cout << "    run #" << runIdx << ", ";
            }
            std::cout << "CV R^2: " << meanFoldDCCalculator.GetMean() << std::endl;
        }

        meanDCCalculator.Add(meanFoldDCCalculator.GetMean());
    }

    if (verbose && runsCount > 1) {
        std::cout << "CV RMSE over " << runsCount << " runs: " << meanDCCalculator.GetMean() << std::endl;
    }

    return {meanDCCalculator.GetMean(), learningTime};
}

int DoCrossValidation(int argc, const char** argv) {
    std::string featuresPath;

    std::string learningMode = "welford_lr";
    size_t foldsCount = 5;
    size_t runsCount = 1;

    std::string verboseMode = "folds";

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path").Required();
        argsParser.AddHandler("method", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr, normalized_welford_lr").Optional();

        argsParser.AddHandler("folds", &foldsCount, "cross-validation folds count").Optional();
        argsParser.AddHandler("runs", &runsCount, "cross-validation runs count").Optional();

        argsParser.AddHandler("verbose", &verboseMode, "verbose mode, one of: folds, cv, overall").Optional();

        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    {
        TTimer timer("pool read in");
        pool.ReadFromFeatures(featuresPath);
    }

    CrossValidation(pool, foldsCount, runsCount, learningMode, verboseMode, true);

    return 0;
}
