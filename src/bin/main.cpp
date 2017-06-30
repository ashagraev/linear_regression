#include "args.h"
#include "tests.h"
#include "timer.h"

#include "../lib/linear_regression.h"
#include "../lib/simple_linear_regression.h"

#include "../lib/metrics.h"
#include "../lib/pool.h"

#include <iostream>
#include <unordered_set>
#include <time.h>

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
    if (learningMode == "precise_welford_lr") {
        linearModel = Solve<TPreciseWelfordLRSolver>(iterator);
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
        argsParser.AddHandler("method", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr, precise_welford_lr").Optional();

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

int DoCrossValidation(int argc, const char** argv) {
    std::string featuresPath;

    std::string learningMode = "welford_lr";
    size_t foldsCount = 5;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path").Required();
        argsParser.AddHandler("method", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr, precise_welford_lr").Optional();

        argsParser.AddHandler("folds", &foldsCount, "cross-validation folds count").Optional();

        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    {
        TTimer timer("pool read in");
        pool.ReadFromFeatures(featuresPath);
    }

    TPool::TCVIterator learnIterator = pool.LearnIterator(foldsCount);
    TPool::TCVIterator testIterator = pool.TestIterator(foldsCount);

    {
        TTimer timer("cross validation taken");

        TMeanCalculator meanDCCalculator;
        for (size_t fold = 0; fold < foldsCount; ++fold) {
            learnIterator.SetTestFold(fold);
            testIterator.SetTestFold(fold);

            const TLinearModel linearModel = Solve(learnIterator, learningMode);
            const double determinationCoefficient = TRegressionMetricsCalculator::Build(testIterator, linearModel).DeterminationCoefficient();

            std::cout << "fold #" << fold << ": R^2 = " << determinationCoefficient << std::endl;

            meanDCCalculator.Add(determinationCoefficient);
        }

        std::cout << "CV R^2: " << meanDCCalculator.GetMean() << std::endl;
    }

    return 0;
}

int DoInjurePool(int argc, const char** argv) {
    std::string featuresPath;
    double injureFactor = 1000;
    double injureOffset = 1000;

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
