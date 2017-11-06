#include "../lib/linear_regression.h"
#include "../lib/simple_linear_regression.h"

#include "../lib/metrics.h"
#include "../lib/pool.h"

#include <iostream>
#include <map>
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

    template <typename TSLRSolver>
    size_t CheckModelPrecision(const TPool& pool, std::map<std::string, size_t>& testCounters) {
        TPool::TSimpleIterator learnIterator = pool.Iterator();
        const TLinearModel model = Solve<TSLRSolver>(learnIterator);
        const double rmse = TRegressionMetricsCalculator::Build(learnIterator, model).RMSE();

        size_t errorsCount = 0;
        if (!DoublesAreQuiteSimilar(rmse, 0.)) {
            std::cerr << TSLRSolver::Name() << " is not enough precise" << std::endl;
            ++errorsCount;
        }

        ++testCounters[TSLRSolver::Name()];

        return errorsCount;
    }

    template <typename TFirstSLRSolver, typename TSecondSRLSolver>
    size_t CheckIfModelsAreEqual(const TPool& pool, std::map<std::string, size_t>& testCounters) {
        TPool::TSimpleIterator learnIterator = pool.Iterator();

        double firstSumSquaredErrorsPrediction, secondSumSquaredErrorsPrediction;
        const TLinearModel firstModel = Solve<TFirstSLRSolver>(learnIterator, &firstSumSquaredErrorsPrediction);
        const TLinearModel secondModel = Solve<TSecondSRLSolver>(learnIterator, &secondSumSquaredErrorsPrediction);

        const double firstRMSEPrediction = sqrt(firstSumSquaredErrorsPrediction / pool.size());
        const double secondRMSEPrediction = sqrt(secondSumSquaredErrorsPrediction / pool.size());

        const double firstRMSE = TRegressionMetricsCalculator::Build(learnIterator, firstModel).RMSE();
        const double secondRMSE = TRegressionMetricsCalculator::Build(learnIterator, secondModel).RMSE();

        size_t errorsCount = 0;
        if (!DoublesAreQuiteSimilar(firstRMSE, secondRMSE)) {
            std::cerr << TFirstSLRSolver::Name() << " & " << TSecondSRLSolver::Name() << " models are different" << std::endl;
            ++errorsCount;
        }
        if (!DoublesAreQuiteSimilar(firstRMSE, firstRMSEPrediction)) {
            std::cerr << TFirstSLRSolver::Name() << " got wrong rmse prediction" << std::endl;
            ++errorsCount;
        }
        if (!DoublesAreQuiteSimilar(secondRMSE, secondRMSEPrediction)) {
            std::cerr << TSecondSRLSolver::Name() << " got wrong rmse prediction" << std::endl;
            ++errorsCount;
        }

        testCounters[TFirstSLRSolver::Name()] += 2;
        testCounters[TSecondSRLSolver::Name()] += 2;

        return errorsCount;
    }

    template <typename TSLRSolver>
    size_t CheckModelCoefficients(const TPool& pool, const std::vector<double>& targetCoefficients, std::map<std::string, size_t>& testCounters) {
        TPool::TSimpleIterator learnIterator = pool.Iterator();
        const TLinearModel model = Solve<TSLRSolver>(learnIterator);

        size_t errorsCount = 0;

        const size_t featuresCount = targetCoefficients.size();
        for (size_t fIdx = 0; fIdx < featuresCount; ++fIdx) {
            const double present = model.Coefficients[fIdx];
            const double actual = targetCoefficients[fIdx];

            if (!DoublesAreQuiteSimilar(present, actual)) {
                std::cerr << "coefficients error for " << TSLRSolver::Name() << ": got " << present << " while " << actual << " is needed for feature #" << fIdx << std::endl;
                ++errorsCount;
            }
        }

        ++testCounters[TSLRSolver::Name()];

        return errorsCount;
    }

    size_t DoTestLRModels(const TPool& pool) {
        std::mt19937 mersenne;
        std::normal_distribution<double> randGen;

        std::vector<TPool> nonZeroMSEPools;
        nonZeroMSEPools.push_back(pool);
        const size_t nonZeroMSEPoolsCount = 5;
        for (size_t nonZeroPoolIdx = 0; nonZeroPoolIdx < nonZeroMSEPoolsCount; ++nonZeroPoolIdx) {
            TPool nonZeroMSEPool(pool);
            for (TInstance& instance : nonZeroMSEPool) {
                instance.Goal += randGen(mersenne) / 10;
            }
            nonZeroMSEPools.push_back(nonZeroMSEPool);
        }

        size_t errorsCount = 0;

        std::map<std::string, size_t> testCounters;

        errorsCount += CheckModelPrecision<TFastLRSolver>(pool, testCounters);
        errorsCount += CheckModelPrecision<TWelfordLRSolver>(pool, testCounters);
        errorsCount += CheckModelPrecision<TNormalizedWelfordLRSolver>(pool, testCounters);

        for (const TPool& nonZeroMSEPool : nonZeroMSEPools) {
            errorsCount += CheckIfModelsAreEqual<TFastBestSLRSolver, TKahanBestSLRSolver>(nonZeroMSEPool, testCounters);
            errorsCount += CheckIfModelsAreEqual<TFastBestSLRSolver, TWelfordBestSLRSolver>(nonZeroMSEPool, testCounters);
            errorsCount += CheckIfModelsAreEqual<TFastBestSLRSolver, TNormalizedWelfordBestSLRSolver>(nonZeroMSEPool, testCounters);

            errorsCount += CheckIfModelsAreEqual<TFastLRSolver, TWelfordLRSolver>(nonZeroMSEPool, testCounters);
            errorsCount += CheckIfModelsAreEqual<TFastLRSolver, TNormalizedWelfordLRSolver>(nonZeroMSEPool, testCounters);

            errorsCount += CheckModelCoefficients<TFastLRSolver>(nonZeroMSEPool, SampleLinearCoefficients(), testCounters);
            errorsCount += CheckModelCoefficients<TWelfordLRSolver>(nonZeroMSEPool, SampleLinearCoefficients(), testCounters);
            errorsCount += CheckModelCoefficients<TNormalizedWelfordLRSolver>(nonZeroMSEPool, SampleLinearCoefficients(), testCounters);
        }

        std::cout << "linear regression errors: " << errorsCount << std::endl;

        for (auto&& nameWithCount : testCounters) {
            std::cout << "\t" << "test runs for " << nameWithCount.first << ": " << nameWithCount.second << std::endl;
        }

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
