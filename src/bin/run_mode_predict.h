#pragma once

#include "args.h"

#include "../lib/linear_model.h"
#include "../lib/pool.h"

#include <iostream>

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
