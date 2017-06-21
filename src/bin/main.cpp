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
        runData.ModelFilePath = argv[3];

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
        return (argc > 1 && strcmp(argv[1], "predict") == 0 && argc == 4) || argc == 5;
    }
};

int PrintHelp() {
    cerr << "usage:" << endl;
    cerr << "    linear_regression learn features_path model_path learning_method" << endl;
    cerr << "    linear_regression predict features_path model_path" << endl;
    cerr << "    linear_regression injure-pool features_path injure_factor injure_offset" << endl;
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

    TLinearModel linearModel;
    if (runData.LearningMode == "fast_bslr") {
        linearModel = Solve<TFastBestSLRSolver>(pool);
    }
    if (runData.LearningMode == "kahan_bslr") {
        linearModel = Solve<TKahanBestSLRSolver>(pool);
    }
    if (runData.LearningMode == "welford_bslr") {
        linearModel = Solve<TWelfordBestSLRSolver>(pool);
    }
    if (runData.LearningMode == "fast_lr") {
        linearModel = Solve<TFastLRSolver>(pool);
    }
    if (runData.LearningMode == "welford_lr") {
        linearModel = Solve<TWelfordLRSolver>(pool);
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

int DoInjurePool(const TRunData &runData) {
    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);

    pool.InjurePool(runData.InjureFactor, runData.InjureOffset);

    for (const TInstance& instance : pool) {
        cout << instance.ToFeaturesString() << "\n";
    }

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
    if (runData.Mode == "injure-pool") {
        return DoInjurePool(runData);
    }

    return PrintHelp();
}
