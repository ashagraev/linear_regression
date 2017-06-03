#include "../lib/linear_regression.h"
#include "../lib/pool.h"

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

struct TRunData {
    string Mode;

    string FeaturesFilePath;
    string ModelFilePath;

    string LearningMode;
    string PredictionsPath;

    static TRunData Load(const char** argv) {
        TRunData runData;

        runData.Mode = argv[1];

        runData.FeaturesFilePath = argv[2];
        runData.ModelFilePath = argv[3];

        runData.LearningMode = argv[4];
        runData.PredictionsPath = argv[4];

        return runData;
    }

    static bool ParametersAreCorrect(int argc, const char** argv) {
        return argc == 5;
    }
};

int PrintHelp() {
    cerr << "usage:" << endl;
    cerr << "    linear_regression learn features_path model_path learning_method" << endl;
    cerr << "    linear_regression predict features_path model_path predictions_path" << endl;
    cerr << "available learn modes:" << endl;
    cerr << "    fast_bslr for simple linear regression" << endl;
    cerr << "    kahan_bslr for simple linear regression with Kahan's summator" << endl;
    cerr << "    welford_bslr for simple linear regression with Welford's method" << endl;
    cerr << "    fast_lr for fast linear regression" << endl;
    cerr << "    welford_lr for linear regression with Welford's method" << endl;

    return 1;
}

int main(int argc, const char** argv) {
    if (!TRunData::ParametersAreCorrect(argc, argv)) {
        return PrintHelp();
    }

    TRunData runData = TRunData::Load(argv);

    TPool pool;
    pool.ReadFromFeatures(runData.FeaturesFilePath);

    if (runData.Mode == "learn") {
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

    if (runData.Mode == "predict") {
        ofstream predictionsOut(runData.PredictionsPath);
        predictionsOut.precision(20);

        TLinearModel linearModel = TLinearModel::LoadFromFile(runData.ModelFilePath);

        for (const TInstance& instance : pool) {
            predictionsOut << linearModel.Prediction(instance.Features) << "\n";
        }

        return 0;
    }

    return PrintHelp();
}
