#include "args.h"

#include <memory>
#include <unordered_map>

void TArgsParser::DoParse(int argc, const char** argv) const {
    --argc;
    ++argv;

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

        parser->second->SetValue(argv[1]);

        argc -= 2;
        argv += 2;
    }
}
