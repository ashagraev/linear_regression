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

TLinearModel Solve(TPool::TPoolIterator iterator, const std::string& learningMode) {
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
    std::string learningMode;
    std::string modelPath;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path");
        argsParser.AddHandler("learning-mode", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr");
        argsParser.AddHandler("model", &modelPath, "resulting model path");
        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);

    TPool::TPoolIterator learnIterator = pool.LearnIterator();

    const TLinearModel linearModel = Solve(learnIterator, learningMode);
    linearModel.SaveToFile(modelPath);

    return 0;
}

int DoPredict(int argc, const char** argv) {
    std::string featuresPath;
    std::string modelPath;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path");
        argsParser.AddHandler("model", &modelPath, "resulting model path");
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
    std::string learningMode;
    size_t foldsCount;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path");
        argsParser.AddHandler("learning-mode", &learningMode, "learning mode, one from: fast_bslr, kahan_bslr, welford_bslr, fast_lr, welford_lr");
        argsParser.AddHandler("folds", &foldsCount, "cross-validation folds count");
        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);

    TPool::TPoolIterator learnIterator = pool.LearnIterator(foldsCount);
    TPool::TPoolIterator testIterator = pool.TestIterator(foldsCount);

    TMeanCalculator meanRMSECalculator;
    for (size_t fold = 0; fold < foldsCount; ++fold) {
        learnIterator.SetTestFold(fold);
        testIterator.SetTestFold(fold);

        const TLinearModel linearModel = Solve(learnIterator, learningMode);
        const double rmse = RMSE(testIterator, linearModel);

        std::cout << "fold #" << fold << ": RMSE = " << rmse << std::endl;

        meanRMSECalculator.Add(rmse);
    }

    std::cout << "CV RMSE: " << meanRMSECalculator.GetMean() << std::endl;

    return 0;
}

int DoInjurePool(int argc, const char** argv) {
    std::string featuresPath;
    double injureFactor;
    double injureOffset;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path");
        argsParser.AddHandler("injure-factor", &injureFactor, "pool injure factor, feature = feature * factor + offset");
        argsParser.AddHandler("injure-offset", &injureOffset, "pool injure offset, feature = feature * factor + offset");
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
        argsParser.AddHandler("features", &featuresPath, "features file path");
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
        argsParser.AddHandler("features", &featuresPath, "features file path");
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
