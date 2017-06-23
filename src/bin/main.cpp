#include "args.h"
#include "tests.h"

#include "../lib/linear_regression.h"
#include "../lib/metrics.h"
#include "../lib/pool.h"

#include <iostream>
#include <unordered_set>
#include <time.h>

struct TRunData {
    std::string Mode;

    std::string FeaturesFilePath;
    std::string ModelFilePath;

    std::string LearningMode;
    std::string PredictionsPath;

    double InjureFactor = 1.;
    double InjureOffset = 0.;
};

int PrintHelp() {
    std::cerr << "modes:" << std::endl;
    std::cerr << "    linear_regression learn" << std::endl;
    std::cerr << "    linear_regression predict" << std::endl;
    std::cerr << "    linear_regression cv" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    linear_regression injure-pool" << std::endl;
    std::cerr << "    linear_regression to-vowpal-wabbit" << std::endl;
    std::cerr << "    linear_regression to-svm-light" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    linear_regression test" << std::endl;
    std::cerr << std::endl;

    return 1;
}

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
    if (learningMode == "fast_lr") {
        linearModel = Solve<TFastLRSolver>(iterator);
    }
    if (learningMode == "welford_lr") {
        linearModel = Solve<TWelfordLRSolver>(iterator);
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
        argsParser.AddHandler("learning-mode", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr").Optional();

        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);

    TPool::TSimpleIterator learnIterator(pool);
    const TLinearModel linearModel = Solve(learnIterator, learningMode);

    if (!modelPath.empty()) {
        linearModel.SaveToFile(modelPath);
    }

    std::cout << "learn rmse: " << RMSE(learnIterator, linearModel) << std::endl;

    return 0;
}

int DoPredict(int argc, const char** argv) {
    std::string featuresPath;
    std::string modelPath;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path").Required();
        argsParser.AddHandler("model", &modelPath, "resulting model path").Required();
        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);

    std::cout.precision(20);

    const TLinearModel linearModel = TLinearModel::LoadFromFile(modelPath);

    for (const TInstance& instance : pool) {
        std::cout << instance.QueryId << "\t"
             << instance.Goal << "\t"
             << instance.Url << "\t"
             << instance.Weight << "\t"
             << linearModel.Prediction(instance.Features) << "\n";
    }

    return 0;
}

double CrossValidation(
    const TPool& pool,
    const size_t foldsCount,
    const size_t runsCount,
    const std::string& learningMode,
    const std::string verboseMode, bool verbose)
{
    TPool::TCVIterator learnIterator = pool.LearnIterator(foldsCount);
    TPool::TCVIterator testIterator = pool.TestIterator(foldsCount);

    TMeanCalculator meanRMSECalculator;
    for (size_t runIdx = 0; runIdx < runsCount; ++runIdx) {
        learnIterator.ResetShuffle();
        testIterator.ResetShuffle();

        TMeanCalculator meanFoldRMSECalculator;
        for (size_t fold = 0; fold < foldsCount; ++fold) {
            learnIterator.SetTestFold(fold);
            testIterator.SetTestFold(fold);

            const TLinearModel linearModel = Solve(learnIterator, learningMode);
            const double rmse = RMSE(testIterator, linearModel);

            if (verbose && verboseMode == "folds") {
                std::cout << "    ";
                if (runsCount > 1) {
                    std::cout << "    run #" << runIdx << ", ";
                }
                std::cout << "fold #" << fold << ": RMSE = " << rmse << std::endl;
            }

            meanFoldRMSECalculator.Add(rmse);
        }

        if (verbose && verboseMode != "overall") {
            if (runsCount > 1) {
                std::cout << "    run #" << runIdx << ", ";
            }
            std::cout << "CV RMSE: " << meanFoldRMSECalculator.GetMean() << std::endl;
        }

        meanRMSECalculator.Add(meanFoldRMSECalculator.GetMean());
    }

    if (verbose && runsCount > 1) {
        std::cout << "CV RMSE over " << runsCount << " runs: " << meanRMSECalculator.GetMean() << std::endl;
    }

    return meanRMSECalculator.GetMean();
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
        argsParser.AddHandler("learning-mode", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr").Optional();

        argsParser.AddHandler("folds", &foldsCount, "cross-validation folds count").Optional();
        argsParser.AddHandler("runs", &runsCount, "cross-validation runs count").Optional();

        argsParser.AddHandler("verbose", &verboseMode, "verbose mode, one of: folds, cv, overall").Optional();

        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);

    CrossValidation(pool, foldsCount, runsCount, learningMode, verboseMode, true);

    return 0;
}

int DoInjurePool(int argc, const char** argv) {
    std::string featuresPath;
    double injureFactor = 1e-3;
    double injureOffset = 1e+3;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path").Required();
        argsParser.AddHandler("injure-factor", &injureFactor, "pool injure factor, feature = feature * factor + offset").Optional();
        argsParser.AddHandler("injure-offset", &injureOffset, "pool injure offset, feature = feature * factor + offset").Optional();
        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);
    pool.InjurePool(injureFactor, injureOffset);
    pool.PrintForFeatures(std::cout);
    return 0;
}

int ToVowpalWabbit(int argc, const char** argv) {
    std::string featuresPath;
    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path").Required();
        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);
    pool.PrintForVowpalWabbit(std::cout);
    return 0;
}

int ToSVMLight(int argc, const char** argv) {
    std::string featuresPath;
    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path").Required();
        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);
    pool.PrintForSVMLight(std::cout);
    return 0;
}

int main(int argc, const char** argv) {
    TModeChooser modeChooser;

    modeChooser.Add("learn", &DoLearn, "learn model from features");
    modeChooser.Add("predict", &DoPredict, "apply learned model to features");
    modeChooser.Add("cv", &DoCrossValidation, "run cross-validation check");
    modeChooser.Add("injure-pool", &DoInjurePool, "create injured pool from source features");
    modeChooser.Add("to-vowpal-wabbit", &ToVowpalWabbit, "create VowpalWabbit-compatible pool");
    modeChooser.Add("to-svm-light", &ToSVMLight, "create SVMLight-compatible pool");
    modeChooser.Add("test", &DoTest, "run tests");

    return modeChooser.Run(argc, argv);
}
