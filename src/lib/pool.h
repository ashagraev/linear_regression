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

class TPool : public std::vector<TInstance> {
private:
    enum ECVIteratorType {
        IT_LEARN,
        IT_TEST,
    };
public:
    class TSimpleIterator;
    class TCVIterator;

    size_t FeaturesCount() const;

    void ReadFromFeatures(const std::string& featuresPath);

    TPool InjuredPool(const double injureFactor, const double injureOffset) const;

    void PrintForFeatures(std::ostream& out) const;
    void PrintForVowpalWabbit(std::ostream& out) const;
    void PrintForSVMLight(std::ostream& out) const;

    TSimpleIterator Iterator() const;

    TCVIterator LearnIterator(const size_t foldsCount) const;
    TCVIterator TestIterator(const size_t foldsCount) const;

    class TSimpleIterator {
    private:
        const TPool& ParentPool;
        TPool::const_iterator Current;
    public:
        TSimpleIterator(const TPool& parentPool);

        bool IsValid() const;
        const TInstance& operator * () const;
        const TInstance* operator ->() const;
        TSimpleIterator& operator++();
        size_t GetInstanceIdx() const;
    };

    class TCVIterator {
    private:
        const TPool& ParentPool;

        size_t FoldsCount;

        TPool::ECVIteratorType IteratorType;
        size_t TestFoldNumber;

        std::vector<size_t> InstanceFoldNumbers;
        std::vector<size_t>::const_iterator Current;

        std::mt19937 RandomGenerator;
    public:
        TCVIterator(const TPool& parentPool,
            const size_t foldsCount,
            const TPool::ECVIteratorType iteratorType);
        TCVIterator(const TCVIterator& source);

        void ResetShuffle();

        void SetTestFold(const size_t testFoldNumber);

        bool IsValid() const;

        const TInstance& operator * () const;
        const TInstance* operator ->() const;
        TCVIterator& operator++();

        size_t GetInstanceIdx() const;
    private:
        void Advance();
        bool TakeCurrent() const;
    };
};
