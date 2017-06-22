#include "../lib/linear_regression.h"
#include "../lib/pool.h"

#include <iostream>
#include <time.h>

struct TRunData {
    string Mode;

    string FeaturesFilePath;
    string ModelFilePath;

    string LearningMode;
    string PredictionsPath;

    double InjureFactor = 1.;
    double InjureOffset = 0.;

    static TRunData Load(int argc, const char** argv) {
        TRunData runData;

        runData.Mode = argv[1];

        runData.FeaturesFilePath = argv[2];

        if (runData.Mode == "cv") {
            runData.LearningMode = argv[3];
        } else if (argc > 3) {
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

        if (strcmp(argv[1], "predict") == 0) {
            return argc == 4;
        }
        if (strcmp(argv[1], "to-vowpal-wabbit") == 0 ||
            strcmp(argv[1], "to-svm-light") == 0 ||
            strcmp(argv[1], "cv") == 0)
        {
            return argc == 3;
        }
        return argc == 5;
    }
};

int PrintHelp() {
    cerr << "usage:" << endl;
    cerr << "    linear_regression learn features_path model_path learning_method" << endl;
    cerr << "    linear_regression predict features_path model_path" << endl;
    cerr << "    linear_regression cv features_path learning_method" << endl;
    cerr << endl;
    cerr << "    linear_regression injure-pool features_path injure_factor injure_offset" << endl;
    cerr << "    linear_regression to-vowpal-wabbit features_path" << endl;
    cerr << "    linear_regression to-svm-light features_path" << endl;
    cerr << endl;
    cerr << "available learn modes:" << endl;
    cerr << "    fast_bslr for simple linear regression" << endl;
    cerr << "    kahan_bslr for simple linear regression with Kahan's summator" << endl;
    cerr << "    welford_bslr for simple linear regression with Welford's method" << endl;
    cerr << "    fast_lr for fast linear regression" << endl;
    cerr << "    welford_lr for linear regression with Welford's method" << endl;

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

    cout.precision(20);

    const TLinearModel linearModel = TLinearModel::LoadFromFile(runData.ModelFilePath);

    for (const TInstance& instance : pool) {
        cout << instance.QueryId << "\t"
             << instance.Goal << "\t"
             << instance.Url << "\t"
             << instance.Weight << "\t"
             << linearModel.Prediction(instance.Features) << "\n";
    }

    return 0;
}

int DoCrossValidation(const TRunData &runData) {
    (void) runData;
    //TPool pool;
    //pool.ReadFromFeatures(runData.FeaturesFilePath);
    return 0;
}

int DoInjurePool(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);
    pool.InjurePool(runData.InjureFactor, runData.InjureOffset);
    pool.PrintForFeatures(cout);
    return 0;
}

int ToVowpalWabbit(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);
    pool.PrintForVowpalWabbit(cout);
    return 0;
}

int ToSVMLight(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);
    pool.PrintForSVMLight(cout);
    return 0;
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
    if (runData.Mode == "cv") {
        return DoCrossValidation(runData);
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

    return PrintHelp();
}
