#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class TArgParser {
public:
    virtual void SetValue(const std::string& arg) = 0;
};

template <typename TValue>
class TSomeArgParser : public TArgParser {
private:
    TValue* Target = nullptr;
public:
    TSomeArgParser(TValue* target)
        : Target(target)
    {
    }

    void SetValue(const std::string& arg) override {
        std::stringstream ss(arg);
        ss >> *Target;
    }
};

class TArgsParser {
private:
    std::vector<std::string> ArgumentNames;
    std::unordered_map<std::string, std::shared_ptr<TArgParser> > Parsers;
public:
    template <typename TValue>
    void AddHandler(std::string key, TValue* target) {
        key = "--" + key;
        ArgumentNames.push_back(key);
        Parsers[key] = std::shared_ptr<TArgParser>(new TSomeArgParser<TValue>(target));
    }

    void DoParse(int argc, const char** argv) const;
};
