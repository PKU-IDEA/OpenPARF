/**
 * @file   database_wrapper.cpp
 * @author Yibai Meng
 * @date   Sep 2020
 */
#include <algorithm>
#include <array>
#include <iterator>
#include <string>
#include <vector>

#include "util/message.h"

// #include "db/LayoutInfo.h"
// #include "util/Interval.h"
#include "ops/clock_network_planner/src/database_wrapper.h"
#include "ops/clock_network_planner/src/utplacefx/SiteColumn.h"


OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {


static constexpr const std::array<SiteType, 4> SLICEL_SLICEM_DSP_RAM_SITE_TYPES = {SiteType::SLICEL, SiteType::SLICEM,
        SiteType::DSP, SiteType::RAM};
// FIXME(Jing Mai):
// This is a dirty solution, but I don't think other methods of checking is better.
// I think we can add user speciable atrributes for elements.
// This code needs a rewrite when we deal with benchmarks that have multiple types
// of DSP, RAM, SLICEL etc. sites
void                                           WrapperDatabase::generateIdSiteTypeMapping() {
  int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0, cnt5 = 0;
  for (auto p = _siteTypeMap.begin(); p != _siteTypeMap.end(); p++) {
    IndexType   id = std::distance(_siteTypeMap.begin(), p);
    std::string n  = (*p).name();
    if (n.find("DSP", 0) != std::string::npos || n.find("RAMBT", 0) != std::string::npos) {
      cnt1++;
      _idToSiteTypeMap[id] = SiteType::DSP;
    } else if (n.find("RAM", 0) != std::string::npos || n.find("RAMAT", 0) != std::string::npos) {
      _idToSiteTypeMap[id] = SiteType::RAM;
      cnt2++;
    } else if (n.find("SLICEL", 0) != std::string::npos || n.rfind("SLICE", 0) != std::string::npos ||
               n.find("FUAT", 0) != std::string::npos) {
      // We just take SLICE to mean SLICEL
      _idToSiteTypeMap[id] = SiteType::SLICEL;
      cnt3++;
    } else if (n.find("SLICEM", 0) != std::string::npos || n.find("FUBT", 0) != std::string::npos) {
      _idToSiteTypeMap[id] = SiteType::SLICEM;
      cnt4++;
    } else if (n.find("IO", 0) != std::string::npos) {
      _idToSiteTypeMap[id] = SiteType::IO;
      cnt5++;
    }
  }
  openparfAssertMsg(cnt1 == 1, "There should be only one kind of DSP site. Now there's %u kind\n", cnt1);
  openparfAssertMsg(cnt2 == 1, "There should be only one kind of RAM site. Now there's %u kind\n", cnt2);
  openparfAssertMsg(cnt3 == 1, "There should be only one kind of SLICEL site. Now there's %u kind\n", cnt3);
  openparfAssertMsg(cnt4 <= 1, "There should be at most one kind of SLICEM site. Now there's %u kind\n", cnt4);
  //   openparfAssertMsg(cnt5 == 1, "There should be only one kind of IO site. Now there's %u kind\n", cnt5);
  for (auto p : _idToSiteTypeMap) {
    _siteTypeToIdMap[p.second] = p.first;
  }
}
// Helper function that turns OpenPARF's sitetype id into
// utplacefx's hardcoded enum class SiteType
SiteType WrapperDatabase::idToSiteType(IndexType i) {
  if (_idToSiteTypeMap.find(i) == _idToSiteTypeMap.end()) openparfAssert(false);
  return _idToSiteTypeMap[i];
}
IndexType WrapperDatabase::siteTypeToId(SiteType s) {
  // Openfort does not yet differenciate SLICEL and SLICEM
  openparfAssertMsg(s != SiteType::SLICEM, "SLICEM not yet supported in OpenPARF");
  if (_siteTypeToIdMap.find(s) == _siteTypeToIdMap.end()) openparfAssert(false);
  return _siteTypeToIdMap[s];
}
// Copied for utplacefx: src/LayoutInfo.cpp

/// Build site columns
void WrapperDatabase::buildSiteColumns() {

  // Reset all site columns and reserve memory
  for (SiteType st : SLICEL_SLICEM_DSP_RAM_SITE_TYPES) {
    siteColumnArrayOfType(st).siteColumnArray().clear();
    siteColumnArrayOfType(st).siteColumnArray().reserve(numSiteX());
  }
  // Build a site column for each x
  // The assumption here is that each column has only one site type,
  for (IndexType x = 0; x < numSiteX(); ++x) {
    SiteColumn sc;
    sc.yArray().reserve(numSiteY());
    SiteType  siteType = SiteType::INVALID;
    IndexType crX      = InvalidIndex<IndexType>::value;
    for (IndexType y = 0; y < numSiteY(); ++y) {
      auto const site_ptr = _siteMap.at(x, y);
      if (not site_ptr) continue;
      auto const &ofSite     = site_ptr.value();
      IndexType   siteTypeId = ofSite.siteTypeId();
      SiteType    st         = idToSiteType(siteTypeId);
      if (siteType == SiteType::INVALID) {
        siteType = st;
      } else {
        openparfAssertMsg(st == siteType, "The column at x = %u has multiple site types.\n", x);
      }
      sc.yArray().push_back(y);
      IndexType tmpCrX = xyToCrX(x, y);
      openparfAssertMsg(crX == InvalidIndex<IndexType>::value or crX == tmpCrX,
              "All sites of the same column must have the same X index.");
      crX = tmpCrX;
    }
    openparfAssertMsg(crX != InvalidIndex<IndexType>::value, "empty column at column %i", x);
    if (std::find(SLICEL_SLICEM_DSP_RAM_SITE_TYPES.begin(), SLICEL_SLICEM_DSP_RAM_SITE_TYPES.end(), siteType) !=
            SLICEL_SLICEM_DSP_RAM_SITE_TYPES.end()) {
      // This a site type that is under consideration ? wtf?
      sc.setX(x);
      sc.setCrX(crX);
      siteColumnArrayOfType(siteType).siteColumnArray().emplace_back(sc);
    }
  }

  // Shrink site column arrays to fit to save memory
  for (SiteType st : SLICEL_SLICEM_DSP_RAM_SITE_TYPES) {
    siteColumnArrayOfType(st).siteColumnArray().shrink_to_fit();
  }

  // Cache site count for each site type
  for (SiteType st : SLICEL_SLICEM_DSP_RAM_SITE_TYPES) {
    IndexType &numSites = numSitesOfType(st);
    numSites            = 0;
    for (const auto &sc : siteColumnArrayOfType(st).siteColumnArray()) {
      numSites += sc.numSites();
    }
  }
}

// Copied for utplacefx: src/LayoutInfo.cpp
/// Initialize clock region informations in site columns
/// REQUIRE 'buildSiteColumns' has been called
void WrapperDatabase::initSiteColumnClockRegionInformation() {
  // Initialize site column range for each clock region X
  for (SiteType st : SLICEL_SLICEM_DSP_RAM_SITE_TYPES) {
    const auto &scArray    = siteColumnArrayOfType(st).siteColumnArray();
    auto       &crXToRange = siteColumnArrayOfType(st).crXToRange();
    crXToRange.clear();
    crXToRange.resize(numCrX(), SiteColumnArray::Range(scArray.cend(), scArray.cend()));
    if (scArray.empty()) {
      // This device does not have sites with type 'st'
      continue;
    }
    auto      it  = scArray.cbegin();
    IndexType crX = it->crX();
    crXToRange.at(crX).setBegin(it);
    for (; it != scArray.end(); ++it) {
      if (it->crX() != crX) {
        // 'it' points to the end of site columns in clock reigon crX
        // and it is also the begin of stie columns in clock region it->crX()
        crXToRange.at(crX).setEnd(it);
        crXToRange.at(it->crX()).setBegin(it);
        crX = it->crX();
      }
    }
  }
  // Build site column segments in each site column
  for (SiteType st : SLICEL_SLICEM_DSP_RAM_SITE_TYPES) {
    for (auto &sc : siteColumnArrayOfType(st).siteColumnArray()) {
      // Reset the site column sements in this site column
      sc.crYToSegment().clear();
      sc.crYToSegment().resize(numCrY(), SiteColumn::Segment(sc.end(), sc.end()));

      // Set segment begin and end for each clock region Y
      auto      it     = sc.begin();
      IndexType oldCrY = xyToCrY(sc.x(), *it);
      sc.segmentOfCrY(oldCrY).setBegin(it);

      for (; it != sc.end(); ++it) {
        IndexType newCrY = xyToCrY(sc.x(), *it);
        if (newCrY != oldCrY) {
          // 'it' is the end of Y in clock region 'oldCrY'
          // and the begin of Y in clock region 'newCrY'
          sc.segmentOfCrY(oldCrY).setEnd(it);
          sc.segmentOfCrY(newCrY).setBegin(it);
          oldCrY = newCrY;
        }
      }
    }
  }

  // Cache site count for each site type in each clock region
  // MOD TODO: OpenPARF's clockregion already contains those information
  // I'll just do the format conversion when cr.numSitesOfType is called, as I don't
  // want to hack into OpenPARF's clock region
  /*
for (auto &cr : _crGrid)
{
  for (SiteType st : SLICEL_SLICEM_DSP_RAM_SITE_TYPES)
  {
      IndexType &numSites = cr.numSitesOfType(st);
      numSites = 0;
      const auto &range = siteColumnArrayOfType(st).rangeOfCrX(cr.crX());
      for (const auto &sc : range)
      {
          numSites += sc.segmentOfCrY(cr.crY()).numSites();
      }
  }
}*/
}
SiteColumnArray &WrapperDatabase::siteColumnArrayOfType(SiteType st) {
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
}

IndexType WrapperDatabase::numSitesOfType(SiteType st) const {
  switch (st) {
    case SiteType::SLICEL:
      return _numSLLSites;
    case SiteType::SLICEM:
      return _numSLMSites;
    case SiteType::DSP:
      return _numDSPSites;
    case SiteType::RAM:
      return _numRAMSites;
    default:
      openparfAssert(false);
  }
  return -1;
}

IndexType &WrapperDatabase::numSitesOfType(SiteType st) {
  switch (st) {
    case SiteType::SLICEL:
      return _numSLLSites;
    case SiteType::SLICEM:
      return _numSLMSites;
    case SiteType::DSP:
      return _numDSPSites;
    case SiteType::RAM:
      return _numRAMSites;
    default:
      openparfAssert(false);
  }
}


}   // namespace utplacefx
OPENPARF_END_NAMESPACE
