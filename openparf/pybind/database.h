/**
 * @file   database.h
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_PYBIND_DATABASE_H_
#define OPENPARF_PYBIND_DATABASE_H_

#include "database/database.h"
#include "database/placedb.h"
#include "pybind/util.h"

namespace database = OPENPARF_NAMESPACE::database;

using Object = database::Object;
using ModelPin = database::ModelPin;
using VectorModelPin = std::vector<ModelPin>;
using Pin = database::Pin;
using VectorPin = std::vector<Pin>;
using VectorPinPtr = std::vector<Pin *>;
using Inst = database::Inst;
using VectorInst = std::vector<Inst>;
using VectorInstPtr = std::vector<Inst *>;
using ModuleInst = database::ModuleInst;
using VectorModuleInst = std::vector<ModuleInst>;
using ModuleInstNetlist = database::ModuleInstNetlist;
using Net = database::Net;
using VectorNet = std::vector<Net>;
using VectorNetPtr = std::vector<Net *>;
using Model = database::Model;
using VectorModel = std::vector<Model>;
using NetlistImpl = database::Netlist<Inst, Net, Pin>;
using BaseAttr = database::BaseAttr;
using InstAttr = database::InstAttr;
using VectorInstAttr = std::vector<InstAttr>;
using NetAttr = database::NetAttr;
using VectorNetAttr = std::vector<NetAttr>;
using PinAttr = database::PinAttr;
using VectorPinAttr = std::vector<PinAttr>;
using Design = database::Design;
using Resource = database::Resource;
using VectorResource = std::vector<Resource>;
using ResourceMap = database::ResourceMap;
using SiteType = database::SiteType;
using VectorSiteType = std::vector<SiteType>;
using SiteTypeMap = database::SiteTypeMap;
using Site = database::Site;
using VectorSite = std::vector<Site>;
using SiteMap = database::SiteMap;
using ClockRegion = database::ClockRegion;
using VectorClockRegion = std::vector<ClockRegion>;
using ClockRegionMap = database::ClockRegionMap;
using HalfColumnRegion = database::HalfColumnRegion;
using VectorHalfColumnRegion = std::vector<HalfColumnRegion>;
using ShapeConstr = database::ShapeConstr;
using VectorShapeConstr = std::vector<ShapeConstr>;
using VectorShapeElement = std::vector<ShapeConstr::ShapeElement>;
using Layout = database::Layout;
using Database = database::Database;
using ModelAreaType = database::ModelAreaType;
using VectorModelAreaType = std::vector<ModelAreaType>;
using VectorModelAreaTypeL2 = std::vector<std::vector<ModelAreaType>>;
using PlaceParams = database::PlaceParams;
using PlaceDB = database::PlaceDB;

#endif
