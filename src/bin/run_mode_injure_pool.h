#pragma once

#include "args.h"

#include "../lib/pool.h"

int DoInjurePool(int argc, const char** argv) {
    std::string featuresPath;
    double injureFactor = 1e-3;
    double injureOffset = 1e+3;

    {
        TArgsParser argsParser;
        argsParser.AddHandler("features", &featuresPath, "features file path").Required();
        argsParser.AddHandler("injure-factor", &injureFactor, "pool injure factor, feature = feature * factor + offset").Optional();
        argsParser.AddHandler("injure-offset", &injureOffset, "pool injure offset, feature = feature * factor + offset").Optional();
        argsParser.DoParse(argc, argv);
    }

    TPool pool;
    pool.ReadFromFeatures(featuresPath);
    pool = pool.InjuredPool(injureFactor, injureOffset);
    pool.PrintForFeatures(std::cout);
    return 0;
}
