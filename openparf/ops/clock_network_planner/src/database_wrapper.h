/**
 * @file   wrapper.h
 * @author Yibai Meng
 * @date   Sep 2020
 * @brief  Wrapper class that emulates the database class needed for UTPlaceFX's ClockNetworkPlanner to run.
 */
#ifndef CNP_DATABASE_WRAPPER_H
#define CNP_DATABASE_WRAPPER_H
#include <map>
#include <vector>

#include "database/clock_region.h"
#include "database/layout.h"
#include "database/site.h"

#include "ops/clock_network_planner/src/utplacefx/SiteColumn.h"
#include "ops/clock_network_planner/src/utplacefx/Types.h"

OPENPARF_BEGIN_NAMESPACE

namespace utplacefx {

    // This class interfaces database::Layout to utplacefx::Database, so that we can
    // use utplacefx's code with minimal changes
    class WrapperDatabase {
    public:
        WrapperDatabase(database::Layout const &l) : _layout(l), _crMap(l.clockRegionMap()),
                                                     _siteMap(l.siteMap()), _siteTypeMap(l.siteTypeMap()), _resourceMap(l.resourceMap()),
                                                     _half_column_regions(l.halfColumnRegions()) {
            generateIdSiteTypeMapping();
            buildSiteColumns();
            initSiteColumnClockRegionInformation();
        }
        const database::ClockRegion &cr(IndexType id) const { return _crMap.at(id); }
        //database::ClockRegion &cr(IndexType id) { return _crMap.at(id); }
        const database::ClockRegion &cr(IndexType x, IndexType y) const { return _crMap.at(x, y); }
        //database::ClockRegion &cr(IndexType x, IndexType y) { return _crMap.at(x, y); }
        const database::ClockRegion &cr(const XY<IndexType> &xy) const { return _crMap.at(xy.x(), xy.y()); }
        //database::ClockRegion &cr(const XY<IndexType> &xy) { return _crMap.at(xy.x(), xy.y()); }

        IndexType      numCrX() const { return _crMap.width(); }
        IndexType      numCrY() const { return _crMap.height(); }
        Box<IndexType> crBndBox() const { return Box<IndexType>(0, 0, numCrX() - 1, numCrY() - 1); }
        IndexType      xyToCrX(IndexType x, IndexType y) const {
            openparfAssert(existSiteAtXY(x, y));
            auto ckId = _siteMap.at(x, y)->clockRegionId();
            openparfAssertMsg(ckId >= 0 and ckId < _crMap.size(), "ckId must be valid. ckid %i site x %i site y %i", ckId, x, y);
            return ckId / numCrY();
        }
        IndexType xyToCrY(IndexType x, IndexType y) const {
            auto ckId = _siteMap.at(x, y)->clockRegionId();
            openparfAssert(existSiteAtXY(x, y));
            openparfAssert(ckId >= 0 and ckId < _crMap.size());
            return ckId % numCrY();
        }
        bool existSiteAtXY(IndexType x, IndexType y) const {
            auto const site_ptr = _siteMap.at(x, y);
            if (site_ptr) return true;
            else
                return false;
        }
        IndexType numSiteX() const { return _siteMap.width(); }
        IndexType numSiteY() const { return _siteMap.height(); }
        // TODO: assert point is dereferenceable?
        // If site map have one, then use it. Otherwise, we compute it ourselves
        IndexType siteCrId(IndexType x, IndexType y) const { return _crMap.crIdAtLoc(x, y); }
        IndexType siteCrId(const XY<IndexType> &xy) const { return _crMap.crIdAtLoc(xy.x(), xy.y()); }
        IndexType siteCrId(const XY<RealType> &xy) const { return _crMap.crIdAtLoc(xy.x(), xy.y()); }

        IndexType numSitesOfType(SiteType st) const;

        //        const SiteColumnArray &siteColumnArrayOfType(SiteType st) const;
        SiteColumnArray &siteColumnArrayOfType(SiteType st);

        IndexType hcSize() { return _half_column_regions.size(); }
        SiteType  idToSiteType(IndexType);
        IndexType siteTypeToId(SiteType);

        database::HalfColumnRegion::BoxType getHalfColumnRegionBbox(IndexType hcid) {
            return _half_column_regions[hcid].bbox();
        }

    private:
        // database::Layout is represent the FPGA architecture
        database::Layout const &                       _layout;
        database::ClockRegionMap const &               _crMap;
        database::SiteMap const &                      _siteMap;
        database::SiteTypeMap const &                  _siteTypeMap;
        database::ResourceMap const &                  _resourceMap;
        const std::vector<database::HalfColumnRegion> &_half_column_regions;

        SiteColumnArray _sllSiteColumnArray;   // SLICEL site columns
        SiteColumnArray _slmSiteColumnArray;   // SLICEM site columns
        SiteColumnArray _dspSiteColumnArray;   // DSP site columns
        SiteColumnArray _ramSiteColumnArray;   // RAM site columns
        // Cached site count for each site type
        IndexType _numSLLSites = 0;   // SLILCEL site
        IndexType _numSLMSites = 0;   // SLILCEM site
        IndexType _numDSPSites = 0;   // DSP site
        IndexType _numRAMSites = 0;   // RAM site

        std::map<IndexType, SiteType> _idToSiteTypeMap;
        std::map<SiteType, IndexType> _siteTypeToIdMap;
        void                          generateIdSiteTypeMapping();
        void                          buildSiteColumns();
        void                          initSiteColumnClockRegionInformation();

        IndexType &numSitesOfType(SiteType st);
    };
    /*
    const SiteColumnArray &WrapperDatabase::siteColumnArrayOfType(SiteType st) const {
        switch (st) {
            case SiteType::SLICEL:
                return _sllSiteColumnArray;
            case SiteType::SLICEM:
                return _slmSiteColumnArray;
            case SiteType::DSP:
                return _dspSiteColumnArray;
            case SiteType::RAM:
                return _ramSiteColumnArray;
            default:
                openparfAssert(false);
        }
    }*/


}   // namespace utplacefx


OPENPARF_END_NAMESPACE
#endif