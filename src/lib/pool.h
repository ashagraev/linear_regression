#pragma once

#include <algorithm>
#include <vector>
#include <random>
#include <string>

struct TInstance {
    std::string QueryId;
    std::string Url;

    std::vector<double> Features;
    double Goal;
    double Weight;

    static TInstance FromFeaturesString(const std::string& featuresString);
    std::string ToFeaturesString() const;
    std::string ToVowpalWabbitString() const;
    std::string ToSVMLightString() const;
};

struct TPool : public std::vector<TInstance> {
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

        std::vector<size_t> InstanceFoldNumbers;
        std::vector<size_t>::const_iterator Current;

        std::mt19937 RandomGenerator;
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

        size_t GetInstanceIdx() const;
    private:
        void Advance();
        bool TakeCurrent() const;
    };

    void ReadFromFeatures(const std::string& featuresPath);
    TCVIterator CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType) const;

    TPool InjurePool(const double injureFactor, const double injureOffset) const;

    void PrintForFeatures(std::ostream& out) const;
    void PrintForVowpalWabbit(std::ostream& out) const;
    void PrintForSVMLight(std::ostream& out) const;
};
