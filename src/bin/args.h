#pragma once

#include <string>

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

    void SetValue(const TValue& arg) override {
        *Target = arg;
    }
};

class TArgsParser {
private:
    vector<std::string> 
    std::unordered_map<std::string, std::shared_ptr<TArgParser> > Parsers;
public:
    template <typename TValue>
    void AddHandler(const std::string& key, TValue* target) {
        Parsers["--" + key] = new TSomeArgParser<TValue>(target));
    }

    void DoParse(int argc, const char** argv) const;
};
