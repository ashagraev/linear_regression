#include "pool.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

TInstance TInstance::FromFeaturesString(const std::string& featuresString) {
    TInstance instance;

    std::stringstream featuresStream(featuresString);

    std::string queryId, url;
    featuresStream >> instance.QueryId;
    featuresStream >> instance.Goal;
    featuresStream >> instance.Url;
    featuresStream >> instance.Weight;
    instance.Weight = 1.;

    double feature;
    while (featuresStream >> feature) {
        instance.Features.push_back(feature);
    }

    return instance;
}

std::string TInstance::ToFeaturesString() const {
    std::stringstream ss;

    ss << QueryId << "\t";
    ss << Goal << "\t";
    ss << Url << "\t";
    ss << Weight;

    ss.precision(20);

    for (const double feature : Features) {
        ss << "\t" << feature;
    }

    return ss.str();
}

std::string TInstance::ToVowpalWabbitString() const {
    std::stringstream ss;

    ss << Goal << " ";
    ss << Weight << " ";
    ss << "|";

    ss.precision(20);

    for (size_t featureIdx = 0; featureIdx < Features.size(); ++ featureIdx) {
        ss << " " << featureIdx << ":" << Features[featureIdx];
    }
    ss << "\t" << QueryId;

    return ss.str();
}

std::string TInstance::ToSVMLightString() const {
    std::stringstream ss;

    ss << Goal;

    ss.precision(20);

    for (size_t featureIdx = 0; featureIdx < Features.size(); ++featureIdx) {
        ss << " " << (featureIdx + 1) << ":" << Features[featureIdx];
    }
    ss << " # " << QueryId;

    return ss.str();
}

const TInstance& TPool::TCVIterator::operator* () const {
    return ParentPool[Current - InstanceFoldNumbers.begin()];
}

const TInstance* TPool::TCVIterator::operator->() const {
    return &ParentPool[Current - InstanceFoldNumbers.begin()];
}

TPool::TCVIterator& TPool::TCVIterator::operator++() {
    Advance();
    return *this;
}

void TPool::TCVIterator::Advance() {
    while (IsValid()) {
        ++Current;
        if (IsValid() && TakeCurrent()) {
            break;
        }
    }
}

bool TPool::TCVIterator::TakeCurrent() const {
    switch (IteratorType) {
    case IT_LEARN: return *Current != TestFoldNumber;
    case IT_TEST: return *Current == TestFoldNumber;
    }
    return false;
}

size_t TPool::TCVIterator::GetInstanceIdx() const {
    return Current - InstanceFoldNumbers.begin();
}

size_t TPool::FeaturesCount() const {
    if (this->empty()) {
        return 0;
    }

    return this->front().Features.size();
}

void TPool::ReadFromFeatures(const std::string& featuresPath) {
    std::ifstream featuresIn(featuresPath);

    std::string featuresString;
    while (getline(featuresIn, featuresString)) {
        if (featuresString.empty()) {
            continue;
        }
        this->push_back(TInstance::FromFeaturesString(featuresString));
    }
}

TPool TPool::InjurePool(const double injureFactor, const double injureOffset) const {
    TPool injuredPool(*this);

    for (TInstance& instance : injuredPool) {
        for (double& feature : instance.Features) {
            feature = feature * injureFactor + injureOffset;
        }
    }

    return injuredPool;
}

void TPool::PrintForFeatures(std::ostream& out) const {
    for (const TInstance& instance : *this) {
        out << instance.ToFeaturesString() << "\n";
    }
}

void TPool::PrintForVowpalWabbit(std::ostream& out) const {
    for (const TInstance& instance : *this) {
        out << instance.ToVowpalWabbitString() << "\n";
    }
}

void TPool::PrintForSVMLight(std::ostream& out) const {
    for (const TInstance& instance : *this) {
        out << instance.ToSVMLightString() << "\n";
    }
}

TPool::TSimpleIterator TPool::Iterator() const {
    return TSimpleIterator(*this);
}

TPool::TCVIterator TPool::LearnIterator(const size_t foldsCount) const {
    return TPool::TCVIterator(*this, foldsCount, TPool::IT_LEARN);
}

TPool::TCVIterator TPool::TestIterator(const size_t foldsCount) const {
    return TPool::TCVIterator(*this, foldsCount, TPool::IT_TEST);
}

TPool::TSimpleIterator::TSimpleIterator(const TPool& parentPool)
    : ParentPool(parentPool)
    , Current(ParentPool.begin())
{
}

bool TPool::TSimpleIterator::IsValid() const {
    return Current != ParentPool.end();
}

const TInstance& TPool::TSimpleIterator::operator*() const {
    return *Current;
}

const TInstance* TPool::TSimpleIterator::operator->() const {
    return &*Current;

}

TPool::TSimpleIterator& TPool::TSimpleIterator::operator++() {
    ++Current;
    return *this;
}

size_t TPool::TSimpleIterator::GetInstanceIdx() const {
    return Current - ParentPool.begin();
}

TPool::TCVIterator::TCVIterator(const TPool& parentPool, const size_t foldsCount, const TPool::ECVIteratorType iteratorType)
    : ParentPool(parentPool)
    , FoldsCount(foldsCount)
    , TestFoldNumber((size_t)-1)
    , IteratorType(iteratorType)
    , InstanceFoldNumbers(ParentPool.size())
    , Current(InstanceFoldNumbers.begin())
{
    ResetShuffle();
}

TPool::TCVIterator::TCVIterator(const TCVIterator& source)
    : ParentPool(source.ParentPool)
    , FoldsCount(source.FoldsCount)
    , IteratorType(source.IteratorType)
    , TestFoldNumber(source.TestFoldNumber)
    , InstanceFoldNumbers(source.InstanceFoldNumbers)
    , Current(InstanceFoldNumbers.begin() + (source.Current - source.InstanceFoldNumbers.begin()))
    , RandomGenerator(source.RandomGenerator)
{
}

void TPool::TCVIterator::ResetShuffle() {
    std::vector<size_t> instanceNumbers(ParentPool.size());
    for (size_t instanceNumber = 0; instanceNumber < ParentPool.size(); ++instanceNumber) {
        instanceNumbers[instanceNumber] = instanceNumber;
    }
    shuffle(instanceNumbers.begin(), instanceNumbers.end(), RandomGenerator);

    for (size_t instancePosition = 0; instancePosition < ParentPool.size(); ++instancePosition) {
        InstanceFoldNumbers[instanceNumbers[instancePosition]] = instancePosition % FoldsCount;
    }
    Current = InstanceFoldNumbers.begin();
}

void TPool::TCVIterator::SetTestFold(const size_t testFoldNumber) {
    TestFoldNumber = testFoldNumber;
    Current = InstanceFoldNumbers.begin();
    if (!TakeCurrent()) {
        Advance();
    }
}

bool TPool::TCVIterator::IsValid() const {
    return Current != InstanceFoldNumbers.end();
}
