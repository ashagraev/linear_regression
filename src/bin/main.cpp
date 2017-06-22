#include "../lib/linear_regression.h"
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

    static TRunData Load(int argc, const char** argv) {
        TRunData runData;

        runData.Mode = argv[1];

        if (argc > 2) {
            runData.FeaturesFilePath = argv[2];
        }
        if (argc > 3) {
            runData.ModelFilePath = argv[3];
        }

        if (argc > 4) {
            runData.LearningMode = argv[4];
            runData.PredictionsPath = argv[4];
            try {
                runData.InjureFactor = atof(argv[3]);
                runData.InjureOffset = atof(argv[4]);
            } catch (...) {
                runData.InjureFactor = 1.;
                runData.InjureOffset = 0.;
            }
        }

        return runData;
    }

    static bool ParametersAreCorrect(int argc, const char** argv) {
        if (argc == 1) {
            return false;
        }
        if (strcmp(argv[1], "test") == 0) {
            return argc == 2;
        }
        if (strcmp(argv[1], "predict") == 0) {
            return argc == 4;
        }
        if (strcmp(argv[1], "to-vowpal-wabbit") == 0 ||
            strcmp(argv[1], "to-svm-light") == 0)
        {
            return argc == 3;
        }
        return argc == 5;
    }
};

int PrintHelp() {
    std::cerr << "usage:" << std::endl;
    std::cerr << "    linear_regression learn features_path model_path learning_method" << std::endl;
    std::cerr << "    linear_regression predict features_path model_path" << std::endl;
    std::cerr << "    linear_regression cv features_path learning_method" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    linear_regression injure-pool features_path injure_factor injure_offset" << std::endl;
    std::cerr << "    linear_regression to-vowpal-wabbit features_path" << std::endl;
    std::cerr << "    linear_regression to-svm-light features_path" << std::endl;
    std::cerr << std::endl;
    std::cerr << "    linear_regression test" << std::endl;
    std::cerr << std::endl;
    std::cerr << "available learn modes:" << std::endl;
    std::cerr << "    fast_bslr for simple linear regression" << std::endl;
    std::cerr << "    kahan_bslr for simple linear regression with Kahan's summator" << std::endl;
    std::cerr << "    welford_bslr for simple linear regression with Welford's method" << std::endl;
    std::cerr << "    fast_lr for fast linear regression" << std::endl;
    std::cerr << "    welford_lr for linear regression with Welford's method" << std::endl;

    return 1;
}

int DoLearn(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);

    TPool::TCVIterator learnIterator = pool.CrossValidationIterator(1, TPool::LearnIterator);

    TLinearModel linearModel;
    if (runData.LearningMode == "fast_bslr") {
        linearModel = Solve<TFastBestSLRSolver>(learnIterator);
    }
    if (runData.LearningMode == "kahan_bslr") {
        linearModel = Solve<TKahanBestSLRSolver>(learnIterator);
    }
    if (runData.LearningMode == "welford_bslr") {
        linearModel = Solve<TWelfordBestSLRSolver>(learnIterator);
    }
    if (runData.LearningMode == "fast_lr") {
        linearModel = Solve<TFastLRSolver>(learnIterator);
    }
    if (runData.LearningMode == "welford_lr") {
        linearModel = Solve<TWelfordLRSolver>(learnIterator);
    }

    linearModel.SaveToFile(runData.ModelFilePath);

    return 0;
}

int DoPredict(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);

    std::cout.precision(20);

    const TLinearModel linearModel = TLinearModel::LoadFromFile(runData.ModelFilePath);

    for (const TInstance& instance : pool) {
        std::cout << instance.QueryId << "\t"
             << instance.Goal << "\t"
             << instance.Url << "\t"
             << instance.Weight << "\t"
             << linearModel.Prediction(instance.Features) << "\n";
    }

    return 0;
}

int DoInjurePool(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);
    pool.InjurePool(runData.InjureFactor, runData.InjureOffset);
    pool.PrintForFeatures(std::cout);
    return 0;
}

int ToVowpalWabbit(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);
    pool.PrintForVowpalWabbit(std::cout);
    return 0;
}

int ToSVMLight(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);
    pool.PrintForSVMLight(std::cout);
    return 0;
}

int DoTest() {
    std::mt19937 mersenne;
    std::normal_distribution<double> randGen;

    const std::vector<double> actualCoefficients = {1., -2., 3., 0., 3., 1., 8., 0.1, -0.1};

    const size_t instancesCount = 1000;
    const size_t featuresCount = actualCoefficients.size();

    TPool pool;
    for (size_t instanceIdx = 0; instanceIdx < instancesCount; ++instanceIdx) {
        TInstance instance;

        for (size_t fIdx = 0; fIdx < featuresCount; ++fIdx) {
            instance.Features.push_back(randGen(mersenne));
        }
        instance.Goal = std::inner_product(instance.Features.begin(), instance.Features.end(), actualCoefficients.begin(), 0.);
        instance.Weight = 1.;
        instance.QueryId = 1;

        pool.push_back(instance);
    }

    size_t errorsCount = 0;
    {
        TPool::TCVIterator iterator = pool.CrossValidationIterator(1, TPool::LearnIterator);
        for (size_t i = 0; i < pool.size(); ++i, ++iterator) {
            if (iterator.GetInstanceIdx() != i) {
                std::cerr << "got error in instance idx for CV iterator on step " << i << std::endl;
                ++errorsCount;
            }
            if (!iterator.IsValid()) {
                std::cerr << "got validation error in CV iterator on step " << i << std::endl;
                ++errorsCount;
            }
        }
        if (iterator.IsValid()) {
            std::cerr << "got valid CV iterator after pool ends" << std::endl;
            ++errorsCount;
        }
    }

    {
        const size_t foldsCount = 10;

        TPool::TCVIterator learnIterator = pool.CrossValidationIterator(foldsCount, TPool::LearnIterator);
        TPool::TCVIterator testIterator = pool.CrossValidationIterator(foldsCount, TPool::TestIterator);

        std::vector<std::unordered_set<size_t> > learnIndexes(foldsCount);
        std::vector<std::unordered_set<size_t> > testIndexes(foldsCount);

        for (size_t fold = 0; fold < foldsCount; ++fold) {
            learnIterator.SetTestFold(0);
            testIterator.SetTestFold(0);

            for (; learnIterator.IsValid(); ++learnIterator) {
                learnIndexes[fold].insert(learnIterator.GetInstanceIdx());
            }
            for (; testIterator.IsValid(); ++testIterator) {
                testIndexes[fold].insert(testIterator.GetInstanceIdx());
                if (learnIndexes[fold].find(testIterator.GetInstanceIdx()) != learnIndexes[fold].end()) {
                    std::cerr << "got iterators error: test instance " << testIterator.GetInstanceIdx() << " is in learn set" << std::endl;
                    ++errorsCount;
                }
            }

            if (learnIndexes[fold].size() + testIndexes[fold].size() != pool.size()) {
                std::cerr << "got iterators error: learn + test size unequal to pool size on fold " << fold
                          << "; learn: " << learnIndexes[fold].size()
                          << ", test: " << testIndexes[fold].size()
                          << ", needed: " << pool.size()
                          << std::endl;
                ++errorsCount;
            }
        }
    }

    TLinearModel fbslrModel = Solve<TFastBestSLRSolver>(pool.CrossValidationIterator(1, TPool::LearnIterator));
    TLinearModel kbslrModel = Solve<TKahanBestSLRSolver>(pool.CrossValidationIterator(1, TPool::LearnIterator));
    TLinearModel wbslrModel = Solve<TWelfordBestSLRSolver>(pool.CrossValidationIterator(1, TPool::LearnIterator));

    TLinearModel flrModel = Solve<TFastLRSolver>(pool.CrossValidationIterator(1, TPool::LearnIterator));
    TLinearModel wlrModel = Solve<TWelfordLRSolver>(pool.CrossValidationIterator(1, TPool::LearnIterator));

    for (size_t fIdx = 0; fIdx < featuresCount; ++fIdx) {
        const double present = flrModel.Coefficients[fIdx];
        const double actual = actualCoefficients[fIdx];
        const double diff = fabs(present - actual);

        if (diff / std::max(actual, 1.) > 0.05) {
            std::cerr << "coefficients error for fast lr solver: got " << present << " while " << actual << " is needed for feature #" << fIdx << std::endl;
            ++errorsCount;
        }
    }
    for (size_t fIdx = 0; fIdx < featuresCount; ++fIdx) {
        const double present = wlrModel.Coefficients[fIdx];
        const double actual = actualCoefficients[fIdx];
        const double diff = fabs(present - actual);

        if (diff / std::max(actual, 1.) > 0.05) {
            std::cerr << "coefficients error for welford lr solver: got " << present << " while " << actual << " is needed for feature #" << fIdx << std::endl;
            ++errorsCount;
        }
    }

    if (errorsCount) {
        std::cerr << std::endl;
    }
    std::cerr << "total errors count: " << errorsCount << std::endl;

    return errorsCount;
}

int main(int argc, const char** argv) {
    if (!TRunData::ParametersAreCorrect(argc, argv)) {
        return PrintHelp();
    }

    TRunData runData = TRunData::Load(argc, argv);

    if (runData.Mode == "learn") {
        return DoLearn(runData);
    }
    if (runData.Mode == "predict") {
        return DoPredict(runData);
    }
    if (runData.Mode == "injure-pool") {
        return DoInjurePool(runData);
    }
    if (runData.Mode == "to-vowpal-wabbit") {
        return ToVowpalWabbit(runData);
    }
    if (runData.Mode == "to-svm-light") {
        return ToSVMLight(runData);
    }
    if (runData.Mode == "test") {
        return DoTest();
    }

    return PrintHelp();
}
