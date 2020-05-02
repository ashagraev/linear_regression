#include "args.h"

#include "run_mode_cross_validation.h"
#include "run_mode_injure_pool.h"
#include "run_mode_learn.h"
#include "run_mode_predict.h"
#include "run_mode_research.h"
#include "run_mode_tests.h"
#include "run_mode_to_svm_light.h"
#include "run_mode_to_vowpal_wabbit.h"

#include "../lib/iterative_linear_regression.h"

int main(int argc, const char** argv) {
    TPool pool;
    pool.ReadFromFeatures("C:/GitHub/linear_regression/data/features/cpu_act.features");

    TIterativeLRSolver adaptiveSolver;
    adaptiveSolver.Learn(pool.Iterator());

    return 0;

    TModeChooser modeChooser;

    modeChooser.Add("learn", &DoLearn, "learn model from features");
    modeChooser.Add("predict", &DoPredict, "apply learned model to features");
    modeChooser.Add("cv", &DoCrossValidation, "run cross-validation check");
    modeChooser.Add("research-bslr", &DoResearchBSLRMethods, "research simple regression learning methods on set of injured pools");
    modeChooser.Add("research-lr", &DoResearchLRMethods, "research linear regression learning methods on set of injured pools");
    modeChooser.Add("injure-pool", &DoInjurePool, "create injured pool from source features");
    modeChooser.Add("to-vowpal-wabbit", &ToVowpalWabbit, "create VowpalWabbit-compatible pool");
    modeChooser.Add("to-svm-light", &ToSVMLight, "create SVMLight-compatible pool");
    modeChooser.Add("test", &DoTest, "run tests");

    return modeChooser.Run(argc, argv);
}
