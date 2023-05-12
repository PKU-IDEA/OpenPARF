#ifndef __SITECOLUMN_H__
#define __SITECOLUMN_H__

#include <algorithm>
#include <vector>

#include "ops/clock_network_planner/src/utplacefx/Types.h"

OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {
    /// Class to represent a column of site (e.g., SLICEL, SLICEM, DSP and RAM)
    class SiteColumn {
    public:
        using YIter = std::vector<IndexType>::const_iterator;

        // Class to store a segment of site column
        class Segment {
        public:
            explicit Segment(SiteColumn::YIter beg, YIter end)
                : _begin(beg), _end(end) {}

            // Getters
            SiteColumn::YIter begin() const { return _begin; }
            SiteColumn::YIter end() const { return _end; }
            IndexType numSites() const { return _end - _begin; }

            // Setters
            void setBegin(SiteColumn::YIter it) { _begin = it; }
            void setEnd(SiteColumn::YIter it) { _end = it; }

        private:
            SiteColumn::YIter _begin;// The begin of site Y
            SiteColumn::YIter _end;  // The end of site Y
        };

    public:
        explicit SiteColumn() = default;

        // Getters
        IndexType x() const { return _x; }
        IndexType crX() const { return _crX; }
        const std::vector<IndexType> &yArray() const { return _yArray; }
        std::vector<IndexType> &yArray() { return _yArray; }
        IndexType yLo() const { return _yArray.front(); }
        IndexType yHi() const { return _yArray.back(); }
        YIter begin() const { return _yArray.cbegin(); }
        YIter end() const { return _yArray.cend(); }
        IndexType numSites() const { return _yArray.size(); }
        const Segment &segmentOfCrY(IndexType crY) const { return _crYToSegment.at(crY); }
        Segment &segmentOfCrY(IndexType crY) { return _crYToSegment.at(crY); }
        const std::vector<Segment> &crYToSegment() const { return _crYToSegment; }
        std::vector<Segment> &crYToSegment() { return _crYToSegment; }

        // Setter
        void setX(IndexType x) { _x = x; }
        void setCrX(IndexType crX) { _crX = crX; }

    private:
        IndexType _x = INDEX_TYPE_MAX;     // The x coordinate of this site column
        IndexType _crX = INDEX_TYPE_MAX;   // The clock region X of this site column
        std::vector<IndexType> _yArray;    // All sites' y coordinates in this column from low to high
        std::vector<Segment> _crYToSegment;// Clock region Y to segment mapping
    };

    /// Class for an array of site columns
    class SiteColumnArray {
    public:
        using SCIter = std::vector<SiteColumn>::const_iterator;

        // Class to store a range of site columns
        class Range {
        public:
            explicit Range(SCIter begin, SCIter end)
                : _begin(begin), _end(end) {}

            // Getters
            SCIter begin() const { return _begin; }
            SCIter end() const { return _end; }
            IndexType numColumns() const { return _end - _begin; }

            // Setters
            void setBegin(SCIter it) { _begin = it; }
            void setEnd(SCIter it) { _end = it; }

        private:
            SCIter _begin;// The begin site column
            SCIter _end;  // The end site column
        };

    public:
        explicit SiteColumnArray() = default;

        // Getters
        const std::vector<SiteColumn> &siteColumnArray() const { return _siteColumnArray; }
        std::vector<SiteColumn> &siteColumnArray() { return _siteColumnArray; }
        const std::vector<Range> &crXToRange() const { return _crXToRange; }
        std::vector<Range> &crXToRange() { return _crXToRange; }
        const Range &rangeOfCrX(IndexType crX) const { return _crXToRange.at(crX); }
        Range &rangeOfCrX(IndexType crX) { return _crXToRange.at(crX); }
        SCIter begin() const { return _siteColumnArray.begin(); }
        SCIter end() const { return _siteColumnArray.end(); }
        IndexType numColumns() const { return _siteColumnArray.size(); }

    private:
        std::vector<SiteColumn> _siteColumnArray;
        std::vector<Range> _crXToRange;
    };
}// namespace utplacefx
OPENPARF_END_NAMESPACE

#endif// __SITECOLUMN_H__
