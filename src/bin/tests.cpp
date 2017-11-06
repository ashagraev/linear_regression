#include "../lib/linear_regression.h"
#include "../lib/simple_linear_regression.h"

#include "../lib/metrics.h"
#include "../lib/pool.h"

#include <iostream>
#include <unordered_set>

namespace {
    bool DoublesAreQuiteSimilar(const double present, const double target, const double possibleError = 0.01) {
        const double diff = fabs(present - target);
        const double normalizer = std::max(fabs(target), 1.);
        const double actualError = diff / normalizer;
        return actualError < possibleError;
    }

    const std::vector<double> SampleLinearCoefficients() {
        return{ 1., -2., 3., 0., 3., 1., 8., 0.1, -0.1, 0., -50. };
    }

    TPool MakeRandomPool() {
        std::mt19937 mersenne;
        std::normal_distribution<double> randGen;

        const std::vector<double> actualCoefficients = SampleLinearCoefficients();

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

        return pool;
    }

    size_t DoTestIterators(const TPool& pool) {
        size_t errorsCount = 0;

        TPool::TSimpleIterator iterator = pool.Iterator();
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

        std::cout << "iterator errors: " << errorsCount << std::endl;
        return errorsCount;
    }

    size_t DoTestCrossValidationIterators(const TPool& pool) {
        size_t errorsCount = 0;

        const size_t foldsCount = 10;

        TPool::TCVIterator learnIterator = pool.LearnIterator(foldsCount);
        TPool::TCVIterator testIterator = pool.TestIterator(foldsCount);

        std::vector<std::unordered_set<size_t>> learnIndexes(foldsCount);
        std::vector<std::unordered_set<size_t>> testIndexes(foldsCount);

        for (size_t fold = 0; fold < foldsCount; ++fold) {
            learnIterator.SetTestFold(fold);
            testIterator.SetTestFold(fold);

            std::unordered_set<size_t>& currentLearnIndexes = learnIndexes[fold];
            std::unordered_set<size_t>& currentTestIndexes = testIndexes[fold];

            for (; learnIterator.IsValid(); ++learnIterator) {
                currentLearnIndexes.insert(learnIterator.GetInstanceIdx());
            }
            for (; testIterator.IsValid(); ++testIterator) {
                currentTestIndexes.insert(testIterator.GetInstanceIdx());
                if (currentLearnIndexes.find(testIterator.GetInstanceIdx()) != currentLearnIndexes.end()) {
                    std::cerr << "got iterators error: test instance " << testIterator.GetInstanceIdx() << " is in learn set" << std::endl;
                    ++errorsCount;
                }
            }

            if (currentLearnIndexes.size() + currentTestIndexes.size() != pool.size()) {
                std::cerr << "got iterators error: learn + test size unequal to pool size on fold " << fold
                    << "; learn: " << currentLearnIndexes.size()
                    << ", test: " << currentTestIndexes.size()
                    << ", needed: " << pool.size()
                    << std::endl;
                ++errorsCount;
            }
        }

        std::unordered_set<size_t> allTestIndexes;
        for (const std::unordered_set<size_t>& sampleTestIndexes : testIndexes) {
            allTestIndexes.insert(sampleTestIndexes.begin(), sampleTestIndexes.end());
        }

        if (allTestIndexes.size() != pool.size()) {
            std::cerr << "got error: union of all test sets unequal to original pool: got " << allTestIndexes.size() << " while " << pool.size() << " are needed" << std::endl;
            ++errorsCount;
        }

        std::cout << "cv iterator errors: " << errorsCount << std::endl;

        return errorsCount;
    }

    size_t DoTestLRModels(const TPool& pool) {
        TPool::TSimpleIterator learnIterator = pool.Iterator();

        TLinearModel fbslrModel = Solve<TFastBestSLRSolver>(learnIterator);
        TLinearModel kbslrModel = Solve<TKahanBestSLRSolver>(learnIterator);
        TLinearModel wbslrModel = Solve<TWelfordBestSLRSolver>(learnIterator);

        TLinearModel flrModel = Solve<TFastLRSolver>(learnIterator);
        TLinearModel wlrModel = Solve<TWelfordLRSolver>(learnIterator);
        TLinearModel nwlrModel = Solve<TNormalizedWelfordLRSolver>(learnIterator);

        const double fbslrRMSE = TRegressionMetricsCalculator::Build(learnIterator, fbslrModel).RMSE();
        const double kbslrRMSE = TRegressionMetricsCalculator::Build(learnIterator, kbslrModel).RMSE();
        const double wbslrRMSE = TRegressionMetricsCalculator::Build(learnIterator, wbslrModel).RMSE();

        const double flrRMSE = TRegressionMetricsCalculator::Build(learnIterator, flrModel).RMSE();
        const double wlrRMSE = TRegressionMetricsCalculator::Build(learnIterator, wlrModel).RMSE();
        const double nwlrRMSE = TRegressionMetricsCalculator::Build(learnIterator, nwlrModel).RMSE();

        size_t errorsCount = 0;

        auto checkRMSE = [&errorsCount](const double rmse, const double targetRMSE, const std::string& title) {
            if (!DoublesAreQuiteSimilar(rmse, targetRMSE)) {
                std::cerr << title << std::endl;
                ++errorsCount;
            }
        };

        checkRMSE(fbslrRMSE, kbslrRMSE, "fast & kahan bslr models are different");
        checkRMSE(fbslrRMSE, wbslrRMSE, "fast & welford bslr models are different");
        checkRMSE(flrRMSE, wlrRMSE, "fast & welford lr models are different");
        checkRMSE(flrRMSE, 0, "fast lr model is not enough precise");
        checkRMSE(wlrRMSE, 0, "welformd lr model is not enough precise");
        checkRMSE(nwlrRMSE, 0, "normalized welformd lr model is not enough precise");

        const size_t featuresCount = pool.FeaturesCount();
        const std::vector<double> actualCoefficients = SampleLinearCoefficients();

        auto testModel = [&](const TLinearModel& model, const std::string& title) {
            for (size_t fIdx = 0; fIdx < featuresCount; ++fIdx) {
                const double present = model.Coefficients[fIdx];
                const double actual = actualCoefficients[fIdx];
    
                if (!DoublesAreQuiteSimilar(present, actual)) {
                    std::cerr << "coefficients error for " << title << ": got " << present << " while " << actual << " is needed for feature #" << fIdx << std::endl;
                    ++errorsCount;
                }
            }    
        };

        testModel(flrModel, "fast lr solver");
        testModel(wlrModel, "welford lr solver");
        testModel(nwlrModel, "normalized welford lr solver");

        std::cout << "lr model errors: " << errorsCount << std::endl;
        std::cout << "    wbslr RMSE: " << wbslrRMSE << std::endl;
        std::cout << "    wlr RMSE:   " << wlrRMSE << std::endl;
        std::cout << "    nwlr RMSE:  " << nwlrRMSE << std::endl;

        return errorsCount;
    }
}

int DoTest(int argc, const char** argv) {
    (void) (argc && argv);

    TPool pool = MakeRandomPool();

    size_t errorsCount = 0;
    errorsCount += DoTestIterators(pool);
    errorsCount += DoTestCrossValidationIterators(pool);
    errorsCount += DoTestLRModels(pool);

    std::cerr << std::endl;
    std::cerr << "total errors count: " << errorsCount << std::endl;

    return errorsCount == 0;
}
