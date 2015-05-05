#include "pool.h"

#include <algorithm>
#include <fstream>
#include <sstream>

TInstance TInstance::FromFeaturesString(const string& featuresString, const size_t idx) {
    TInstance instance;
    instance.Idx = idx;

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

TPool::TCVIterator::TCVIterator(const TPool& parentPool,
                                const size_t foldsCount,
                                const TPool::EIteratorType iteratorType)
    : ParentPool(parentPool)
    , FoldsCount(foldsCount)
    , InstanceFoldNumbers(ParentPool.size())
    , IteratorType(iteratorType)
    , RandomGenerator(0)
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

const TInstance& TPool::TCVIterator::operator*() {
    return ParentPool[Current - InstanceFoldNumbers.begin()];
}

const TInstance* TPool::TCVIterator::operator->() {
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

TPool::TCVIterator TPool::CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType) const {
    return TPool::TCVIterator(*this, foldsCount, iteratorType);
}

TPool TPool::ReadPoolFromFeatures(const string& featuresPath) {
    TPool pool;

    ifstream featuresIn(featuresPath);

    size_t idx = 0;
    string featuresString;
    while (getline(featuresIn, featuresString)) {
        if (featuresString.empty()) {
            continue;
        }
        pool.push_back(TInstance::FromFeaturesString(featuresString, idx++));
    }
    return pool;
}

void TPool::SaveToFeatures(const string path) const {
    ofstream featuresOut(path);
    for (const TInstance& instance : *this) {
        featuresOut << 1 << "\t";
        featuresOut << instance.Goal << "\t";
        featuresOut << 1 << "\t";
        featuresOut << instance.Weight;

        for (const double feature : instance.Features) {
            char dest[20];
            sprintf_s(dest, "%.10lf", feature);
            featuresOut << dest << ",";
        }
        featuresOut << "\n";
    }
}

void TPool::SaveToArff(const string path) const {
    ofstream arffOut(path);

    if (this->empty()) {
        return;
    }

    arffOut << "@relation rel\n";
    size_t featuresCount = this->front().Features.size();
    for (size_t featureNumber = 0; featureNumber < featuresCount; ++featureNumber) {
        arffOut << "@attribute attr" << featureNumber << " real\n";
    }
    arffOut << "@attribute goal real\n";
    arffOut << "@data\n";

    for (const TInstance& instance : *this) {
        for (const double feature : instance.Features) {
            char dest[20];
            sprintf_s(dest, "%.10lf", feature);
            arffOut << dest << ",";
        }
        arffOut << instance.Goal << "\n";
    }
}
