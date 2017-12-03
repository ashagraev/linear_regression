#pragma once

class TKahanAccumulator {
private:
    double Sum;
    double Addition;

public:
    TKahanAccumulator(const double value = 0.)
        : Sum(value)
        , Addition(0.)
    {
    }

    TKahanAccumulator& operator+=(const double value) {
        const double y = value - Addition;
        const double t = Sum + y;
        Addition = (t - Sum) - y;
        Sum = t;
        return *this;
    }

    TKahanAccumulator& operator+=(const TKahanAccumulator& other) {
        return *this += (double)other;
    }

    operator double() const {
        return Sum + Addition;
    }
};
