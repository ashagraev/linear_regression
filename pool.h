#pragma once

#include <vector>
#include <random>
#include <string>

using namespace std;

struct TInstance {
    size_t Idx;

    double Goal;
    double Weight;
    vector<double> Features;

    static TInstance FromFeaturesString(const string& featuresString, const size_t idx);
};

class TPool : public vector<TInstance> {
public:
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
        vector<size_t>::iterator Current;

        mt19937 RandomGenerator;
    public:
        TCVIterator(const TPool& parentPool,
                    const size_t foldsCount,
                    const EIteratorType iteratorType);
        void ResetShuffle();
        void SetTestFold(const size_t testFoldNumber);
        bool IsValid() const;

        const TInstance& operator*();
        const TInstance* operator->();

        TPool::TCVIterator& operator++();
    private:
        void Advance();
        bool TakeCurrent() const;
    };

    TCVIterator CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType) const;

    static TPool ReadPoolFromFeatures(const string& featuresPath);

    void SaveToFeatures(const string path) const;
    void SaveToArff(const string path) const;
};
