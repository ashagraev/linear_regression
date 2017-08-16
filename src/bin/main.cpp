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
    if (learningMode == "normalized_welford_bslr") {
        linearModel = Solve<TWelfordBestSLRSolver>(iterator);
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

struct TCrossValidationResult {
    double MeanDeterminationCoefficient;
    double LearningTimeInSeconds;
};

TCrossValidationResult CrossValidation(
    const TPool& pool,
    const size_t foldsCount,
    const size_t runsCount,
    const std::string& learningMode,
    const std::string verboseMode, bool verbose)
{
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

struct TResearchOptions {
    std::string FeaturesPath;

    size_t FoldsCount = 5;
    size_t RunsCount = 1;

    size_t TasksCount = 5;
    double DegradeFactor = 0.2;

    void AddOpts(TArgsParser& argsParser) {
        argsParser.AddHandler("features", &FeaturesPath, "features file path").Required();

        argsParser.AddHandler("tasks", &TasksCount, "number of research tasks").Optional();
        argsParser.AddHandler("degrade", &DegradeFactor, "task-to-task degrade level").Optional();

        argsParser.AddHandler("folds", &FoldsCount, "cross-validation folds count").Optional();
        argsParser.AddHandler("runs", &RunsCount, "cross-validation runs count").Optional();
    }

    std::vector<std::pair<double, double>> GetInjureFactorsAndOffsets() const {
        std::vector<std::pair<double, double>> injureFactorsAndOffsets;

        double injureFactor = 1.;
        double injureOffset = 1.;

        for (size_t taskIdx = 0; taskIdx < TasksCount; ++taskIdx) {
            injureFactorsAndOffsets.push_back(std::make_pair(injureFactor, injureOffset));

            injureFactor *= DegradeFactor;
            injureOffset /= DegradeFactor;
        }
        return injureFactorsAndOffsets;
    }
};

int DoResearchMethods(const TResearchOptions& researchOptions,
                      const std::vector<std::string>& learningModes)
{
    TPool pool;
    pool.ReadFromFeatures(researchOptions.FeaturesPath);

    const std::vector<std::pair<double, double>> injureFactorsAndOffsets = researchOptions.GetInjureFactorsAndOffsets();

    std::vector<std::vector<double>> scores(learningModes.size());
    std::vector<double> fullLearnTime(learningModes.size());

    for (const std::pair<double, double>& injureFactorAndOffset : injureFactorsAndOffsets) {
        const double injureFactor = injureFactorAndOffset.first;
        const double injureOffset = injureFactorAndOffset.second;

        const TPool injuredPool = pool.InjuredPool(injureFactor, injureOffset);

        std::cerr << "injure factor: " << injureFactor << std::endl;
        std::cerr << "injure offset: " << injureOffset << std::endl;

        for (size_t methodIdx = 0; methodIdx < learningModes.size(); ++methodIdx) {
            const TCrossValidationResult cvResult = CrossValidation(injuredPool, researchOptions.FoldsCount, researchOptions.RunsCount, learningModes[methodIdx], "", false);

            std::stringstream ss;
            ss << "   ";
            ss << learningModes[methodIdx];
            while (ss.str().size() < 50) {
                ss << " ";
            }

            ss.precision(5);

            ss << "time: " << cvResult.LearningTimeInSeconds << "    " << "R^2: " << cvResult.MeanDeterminationCoefficient;

            std::cerr << ss.str() << std::endl;

            scores[methodIdx].push_back(cvResult.MeanDeterminationCoefficient);
            fullLearnTime[methodIdx] += cvResult.LearningTimeInSeconds;
        }
        std::cerr << std::endl;
    }

    std::cerr << "full learning time:" << std::endl;
    for (size_t methodIdx = 0; methodIdx < learningModes.size(); ++methodIdx) {
        std::stringstream ss;
        ss << "   ";
        ss << learningModes[methodIdx];
        while (ss.str().size() < 50) {
            ss << " ";
        }
        ss.precision(5);

        ss << fullLearnTime[methodIdx] << "s";

        std::cerr << ss.str() << std::endl;
    }

    return 0;
}

int DoResearchBSLRMethods(int argc, const char** argv) {
    TResearchOptions researchOptions;

    {
        TArgsParser argsParser;
        researchOptions.AddOpts(argsParser);
        argsParser.DoParse(argc, argv);
    }

    const std::vector<std::string> learningModes = { "fast_bslr", "kahan_bslr", "welford_bslr", "normalized_welford_bslr" };
    return DoResearchMethods(researchOptions, learningModes);
}

int DoResearchLRMethods(int argc, const char** argv) {
    TResearchOptions researchOptions;

    {
        TArgsParser argsParser;
        researchOptions.AddOpts(argsParser);
        argsParser.DoParse(argc, argv);
    }

    const std::vector<std::string> learningModes = { "fast_lr", "welford_lr", "normalized_welford_lr" };
    return DoResearchMethods(researchOptions, learningModes);
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
    pool = pool.InjuredPool(injureFactor, injureOffset);
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
    modeChooser.Add("research-bslr", &DoResearchBSLRMethods, "research simple regression learning methods on set of injured pools");
//    modeChooser.Add("research-lr", &DoResearchLRMethods, "research linear regression learning methods on set of injured pools");
    modeChooser.Add("injure-pool", &DoInjurePool, "create injured pool from source features");
    modeChooser.Add("to-vowpal-wabbit", &ToVowpalWabbit, "create VowpalWabbit-compatible pool");
    modeChooser.Add("to-svm-light", &ToSVMLight, "create SVMLight-compatible pool");
    modeChooser.Add("test", &DoTest, "run tests");

    return modeChooser.Run(argc, argv);
}
