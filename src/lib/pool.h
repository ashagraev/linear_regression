#pragma once

#include <algorithm>
#include <vector>
#include <random>
#include <string>

using namespace std;

struct TInstance {
    string QueryId;
    string Url;

    vector<double> Features;
    double Goal;
    double Weight;

    static TInstance FromFeaturesString(const string& featuresString);
    string ToFeaturesString() const;
    string ToVowpalWabbitString() const;
};

struct TPool : public vector<TInstance> {
    enum EIteratorType {
        LearnIterator,
        TestIterator,
    };

    class TCVIterator {
    private:
        const TPool& ParentPool;

        size_t FoldsCount;

        EIteratorType IteratorType;
        size_t TestFoldNumber;

        vector<size_t> InstanceFoldNumbers;
        vector<size_t>::const_iterator Current;

        mt19937 RandomGenerator;
    public:
        TCVIterator(const TPool& parentPool,
                    const size_t foldsCount,
                    const EIteratorType iteratorType);

        void ResetShuffle();

        void SetTestFold(const size_t testFoldNumber);

        bool IsValid() const;

        const TInstance& operator * () const;
        const TInstance* operator ->() const;
        TPool::TCVIterator& operator++();
    private:
        void Advance();
        bool TakeCurrent() const;
    };

    void ReadFromFeatures(const string& featuresPath);
    TCVIterator CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType) const;

    TPool InjurePool(const double injureFactor, const double injureOffset) const;
};
