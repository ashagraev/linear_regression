#include "args.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

void TArgsParser::DoParse(int argc, const char** argv) const {
    --argc;
    ++argv;

    std::unordered_set<std::string> usedKeys;

    while (argc) {
        std::string parameter = argv[0];
        if (argc < 2) {
            std::string message = "missing parameter for " + parameter + " argument";
            throw std::exception(message.c_str());
        }

        auto parser = Parsers.find(parameter);
        if (parser == Parsers.end()) {
            std::string message = "unknown parameter: " + parameter;
            throw std::exception(message.c_str());
        }

        usedKeys.insert(parameter);
        parser->second->SetValue(argv[1]);

        argc -= 2;
        argv += 2;
    }

    std::vector<std::string> lostParameters;
    for (auto&& keyWithParser : Parsers) {
        const std::string& key = keyWithParser.first;
        if (usedKeys.find(key) == usedKeys.end()) {
            lostParameters.push_back(key);
        }
    }

    if (!lostParameters.empty()) {
        std::string message = "those parameters are needed: ";
        message += lostParameters.front();
        for (size_t i = 1; i < lostParameters.size(); ++i) {
            message += ", " + lostParameters[i];
        }
        throw std::exception(message.c_str());
    }
}
