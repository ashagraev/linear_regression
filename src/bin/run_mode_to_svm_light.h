#pragma once

#include "args.h"

#include "../lib/pool.h"

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
