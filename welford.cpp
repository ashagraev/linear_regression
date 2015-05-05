#include "welford.h"

void TMeanCalculator::Add(const double value, const double weight /*= 1.*/) {
    SumWeights += weight;
    Mean += weight * (value - Mean) / SumWeights;
}

double TMeanCalculator::GetMean() const {
    return Mean;
}

void TCovariationCalculator::Add(const double firstValue, const double secondValue, const double weight /*= 1.*/) {
    FirstValueMeanCalculator.Add(firstValue, weight);
    Covariation += weight * (firstValue - FirstValueMeanCalculator.GetMean()) * (secondValue - SecondValueMeanCalculator.GetMean());
    SecondValueMeanCalculator.Add(secondValue, weight);
}

double TCovariationCalculator::GetFirstValueMean() const {
    return FirstValueMeanCalculator.GetMean();
}

double TCovariationCalculator::GetSecondValueMean() const {
    return SecondValueMeanCalculator.GetMean();
}

double TCovariationCalculator::GetCovariation() const {
    return Covariation;
}

void TDeviationCalculator::Add(const double value, const double weight /*= 1.*/) {
    const double lastMean = MeanCalculator.GetMean();
    MeanCalculator.Add(value, weight);
    Covariation += weight * (value - lastMean) * (value - MeanCalculator.GetMean());
}

double TDeviationCalculator::GetMean() const {
    return MeanCalculator.GetMean();
}

double TDeviationCalculator::GetCovariation() const {
    return Covariation;
}
