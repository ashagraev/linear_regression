#include "pool.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

TInstance TInstance::FromFeaturesString(const string& featuresString) {
    TInstance instance;

    stringstream featuresStream(featuresString);

    string queryId, url;
    featuresStream >> queryId;
    featuresStream >> instance.Goal;
    featuresStream >> url;
    featuresStream >> instance.Weight;
    instance.Weight = 1.;

    double feature;
    while (featuresStream >> feature) {
        instance.Features.push_back(feature);
    }

    return instance;
}

TPool::TCVIterator::TCVIterator(const TPool& parentPool, const size_t foldsCount, const EIteratorType iteratorType)
    : ParentPool(parentPool)
    , FoldsCount(foldsCount)
    , IteratorType(iteratorType)
    , InstanceFoldNumbers(ParentPool.size())
{
}

void TPool::TCVIterator::ResetShuffle() {
    vector<size_t> instanceNumbers(ParentPool.size());
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
    Advance();
}

bool TPool::TCVIterator::IsValid() const {
    return Current != InstanceFoldNumbers.end();
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
    case LearnIterator: return *Current != TestFoldNumber;
    case TestIterator: return *Current == TestFoldNumber;
    }
    return false;
}

void TPool::ReadFromFeatures(const string& featuresPath) {
    ifstream featuresIn(featuresPath);

    string featuresString;
    while (getline(featuresIn, featuresString)) {
        if (featuresString.empty()) {
            continue;
        }
        this->push_back(TInstance::FromFeaturesString(featuresString));
    }
}

TPool::TCVIterator TPool::CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType) const {
    return TPool::TCVIterator(*this, foldsCount, iteratorType);
}

TPool TPool::InjurePool(const double injureFactor, const double injureOffset) const {
    TPool injuredPool(*this);

    for (TInstance& instance : injuredPool) {
        for (double& feature : instance.Features) {
            feature = feature * injureFactor + injureOffset;
        }
        instance.Goal = instance.Goal * injureFactor + injureOffset;
    }

    return injuredPool;
}
