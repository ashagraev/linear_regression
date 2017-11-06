#pragma once

#include <chrono>
#include <iostream>
#include <string>

class TTimer {
private:
    using TClockType = std::chrono::high_resolution_clock;
    using TTimeType = std::chrono::time_point<TClockType>;
    TTimeType Start;

    std::string Title;

public:
    TTimer(const std::string& title = std::string())
        : Start(TClockType::now())
        , Title(title) 
    {
    }

    ~TTimer() {
        if (!Title.empty()) {
            std::cout << Title << " " << GetSecondsPassed() << "s" << std::endl;
        }
    }

    double GetSecondsPassed() const {
        TTimeType now = TClockType::now();
        std::chrono::microseconds diff = std::chrono::duration_cast<std::chrono::microseconds>(now - Start);
        return (double)diff.count() / 1000000;
    }
};
