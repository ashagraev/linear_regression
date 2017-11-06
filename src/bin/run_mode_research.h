#pragma once

#include "args.h"
#include "run_mode_cross_validation.h"

#include "../lib/pool.h"

#include <iostream>

struct TResearchOptions {
    std::string FeaturesPath;

    size_t FoldsCount = 5;
    size_t RunsCount = 1;

    size_t TasksCount = 5;
    double DegradeFactor = 0.1;

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
