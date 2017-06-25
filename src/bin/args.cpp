#include "args.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

void TArgsParser::DoParse(int argc, const char** argv) const {
    try {
        std::unordered_set<std::string> usedKeys;

        while (argc) {
            std::string argument = argv[0];

            auto parser = Parsers.find(argument);
            if (parser == Parsers.end()) {
                std::string message = "unknown argument: " + argument;
                throw std::exception(message.c_str());
            }

            if (argc < 2) {
                std::string message = "missing parameter for " + argument;
                throw std::exception(message.c_str());
            }

            usedKeys.insert(argument);
            parser->second->SetValue(argv[1]);

            argc -= 2;
            argv += 2;
        }

        std::vector<std::string> lostArguments;
        for (auto&& keyWithParser : Parsers) {
            if (!keyWithParser.second->IsRequired()) {
                continue;
            }

            const std::string& key = keyWithParser.first;
            if (usedKeys.find(key) == usedKeys.end()) {
                lostArguments.push_back(key);
            }
        }

        if (!lostArguments.empty()) {
            std::string message = "those arguments are required: ";
            message += lostArguments.front();
            for (size_t i = 1; i < lostArguments.size(); ++i) {
                message += ", " + lostArguments[i];
            }
            throw std::exception(message.c_str());
        }
    } catch (std::exception ex) {
        std::cerr << ex.what();
        std::cerr << std::endl;
        PrintHelp();

        exit(1);
    } 
}

void TArgsParser::PrintHelp() const {
    size_t maxKeyLength = 0;
    for (const std::string& key : ArgumentNames) {
        maxKeyLength = std::max(maxKeyLength, key.length());
    }

    size_t maxDefaultLength = 0;
    for (auto&& parser : Parsers) {
        if (parser.second->IsRequired()) {
            continue;
        }
        maxDefaultLength = std::max(maxDefaultLength, parser.second->GetValue().size());
    }

    const std::string tab = "    ";
    const std::string halfTab = "  ";
    const std::string requiredStr = "REQUIRED";
    const std::string optionalStr = "OPTIONAL";
    const std::string defaultStr = "DEFAULT: ";

    const size_t commonPrefixLength =
        tab.length() * 3 +
        halfTab.length() +
        std::max(requiredStr.length(), optionalStr.length()) +
        defaultStr.length() +
        maxKeyLength +
        maxDefaultLength;

    for (const std::string& key : ArgumentNames) {
        std::stringstream line;
        line << tab << key;
        for (size_t i = key.length(); i < maxKeyLength + tab.length(); ++i) {
            line << " ";
        }

        auto&& parser = Parsers.find(key);
        if (parser->second->IsRequired()) {
            line << requiredStr << halfTab;
        } else {
            line << optionalStr << halfTab;
            line << defaultStr << parser->second->GetValue();
        }

        for (size_t i = line.str().length(); i < commonPrefixLength; ++i) {
            line << " ";
        }
        line << parser->second->GetDescription();

        std::cerr << line.str() << std::endl;
    }
}

int TModeChooser::Run(int argc, const char** argv) {
    if (argc < 2) {
        PrintHelp();
        exit(1);
    }

    const std::string arg = argv[1];
    auto mainFunc = Functions.find(arg);
    if (mainFunc == Functions.end()) {
        std::cerr << "unknown mode: " << arg << std::endl;
        std::cerr << std::endl;
        PrintHelp();
        exit(1);
    }

    return (*mainFunc->second)(argc - 2, argv + 2);
}

void TModeChooser::PrintHelp() const {
    size_t maxKeyLength = 0;
    for (const TFuncInfo& funcInfo : FunctionInfos) {
        maxKeyLength = std::max(maxKeyLength, funcInfo.Argument.length());
    }

    std::cerr << "available modes:" << std::endl;
    for (const TFuncInfo& funcInfo : FunctionInfos) {
        std::stringstream line;
        line << "    " << funcInfo.Argument;
        for (size_t i = funcInfo.Argument.length(); i < maxKeyLength + 8; ++i) {
            line << " ";
        }

        line << funcInfo.Description;

        std::cerr << line.str() << std::endl;
    }
}
