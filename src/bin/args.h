#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class TArgParser {
public:
    virtual void SetValue(const std::string& arg) = 0;

    virtual std::string GetDescription() const = 0;
    virtual bool IsRequired() const = 0;
};

template <typename TValue>
class TSomeArgParser : public TArgParser {
private:
    TValue* Target = nullptr;
    std::string Description;
    bool IsOptional = false;

    TSomeArgParser(TValue* target, const std::string& description)
        : Target(target)
        , Description(description)
    {
    }

    void SetValue(const std::string& arg) override {
        std::stringstream ss(arg);
        ss >> *Target;
    }

    std::string GetDescription() const override {
        return Description;
    }

    bool IsRequired() const override {
        return !IsOptional;
    }
public:
    void Required() {
        IsOptional = false;
    }
    void Optional() {
        IsOptional = true;
    }
private:
    friend class TArgsParser;
};

class TArgsParser {
private:
    std::vector<std::string> ArgumentNames;
    std::unordered_map<std::string, std::shared_ptr<TArgParser> > Parsers;

    std::unordered_set<std::string> RequiredArguments;
public:
    template <typename TValue>
    TSomeArgParser<TValue>& AddHandler(std::string key, TValue* target, const std::string& description) {
        key = "--" + key;
        ArgumentNames.push_back(key);
        Parsers[key] = std::shared_ptr<TArgParser>(new TSomeArgParser<TValue>(target, description));
    }

    void DoParse(int argc, const char** argv) const;
    void PrintHelp() const;
};

class TModeChooser {
private:
    struct TFuncInfo {
        std::string Argument;
        std::string Description;
    };

    using TMainFunc = int(int argc, const char** argv);

    std::vector<TFuncInfo> FunctionInfos;
    std::unordered_map<std::string, TMainFunc*> Functions;
public:
    void Add(const std::string& arg, TMainFunc* function, const std::string description) {
        FunctionInfos.push_back({arg, description});
        Functions[arg] = function;
    }

    int Run(int argc, const char** argv);
    void PrintHelp() const;
};
