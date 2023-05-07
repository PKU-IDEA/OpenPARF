/**
 * @file   database.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "pybind/database.h"

#include <cstdint>
#include <vector>

#include "pybind/container.h"
#include "pybind/geometry.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(Object);

PYBIND11_MAKE_OPAQUE(ModelPin);
PYBIND11_MAKE_OPAQUE(VectorModelPin);

PYBIND11_MAKE_OPAQUE(Pin);
PYBIND11_MAKE_OPAQUE(VectorPin);

PYBIND11_MAKE_OPAQUE(Inst);
PYBIND11_MAKE_OPAQUE(VectorInst);

PYBIND11_MAKE_OPAQUE(ModuleInst);

PYBIND11_MAKE_OPAQUE(ModuleInstNetlist);

PYBIND11_MAKE_OPAQUE(Net);
PYBIND11_MAKE_OPAQUE(VectorNet);

PYBIND11_MAKE_OPAQUE(Model);
PYBIND11_MAKE_OPAQUE(VectorModel);

PYBIND11_MAKE_OPAQUE(NetlistImpl);

PYBIND11_MAKE_OPAQUE(BaseAttr);

PYBIND11_MAKE_OPAQUE(InstAttr);
PYBIND11_MAKE_OPAQUE(VectorInstAttr);

PYBIND11_MAKE_OPAQUE(NetAttr);
PYBIND11_MAKE_OPAQUE(VectorNetAttr);

PYBIND11_MAKE_OPAQUE(PinAttr);
PYBIND11_MAKE_OPAQUE(VectorPinAttr);

PYBIND11_MAKE_OPAQUE(Design);

PYBIND11_MAKE_OPAQUE(Resource);
PYBIND11_MAKE_OPAQUE(VectorResource);

PYBIND11_MAKE_OPAQUE(ResourceMap);

PYBIND11_MAKE_OPAQUE(SiteType);
PYBIND11_MAKE_OPAQUE(VectorSiteType);

PYBIND11_MAKE_OPAQUE(SiteTypeMap);

PYBIND11_MAKE_OPAQUE(Site);
PYBIND11_MAKE_OPAQUE(VectorSite);

PYBIND11_MAKE_OPAQUE(SiteMap);

PYBIND11_MAKE_OPAQUE(ClockRegion);
PYBIND11_MAKE_OPAQUE(VectorClockRegion);

PYBIND11_MAKE_OPAQUE(ClockRegionMap);

PYBIND11_MAKE_OPAQUE(HalfColumnRegion);
PYBIND11_MAKE_OPAQUE(VectorHalfColumnRegion);

PYBIND11_MAKE_OPAQUE(ShapeConstr);
PYBIND11_MAKE_OPAQUE(VectorShapeConstr);

PYBIND11_MAKE_OPAQUE(ShapeConstr::ShapeElement);
PYBIND11_MAKE_OPAQUE(VectorShapeElement);

PYBIND11_MAKE_OPAQUE(Layout);
PYBIND11_MAKE_OPAQUE(Database);
PYBIND11_MAKE_OPAQUE(Database::Params);

PYBIND11_MAKE_OPAQUE(ModelAreaType);
PYBIND11_MAKE_OPAQUE(VectorModelAreaType);
PYBIND11_MAKE_OPAQUE(VectorModelAreaTypeL2);
PYBIND11_MAKE_OPAQUE(PlaceParams);
PYBIND11_MAKE_OPAQUE(PlaceDB);

void bind_database(py::module &m) {
  m.doc()              = R"pbdoc(
        Pybind11 plugin
        -----------------------
        .. currentmodule:: database
        .. autosummary::
           :toctree: _generate
           Object
           ModelPin
           Pin
           Inst
           ModuleInst
           ModuleInst::Netlist
           Net
           Model
           Netlist
           Attributes
           BaseAttr
           InstAttr
           NetAttr
           PinAttr
           Design
           Resource
           ResourceMap
           SiteType
           SiteTypeMap
           Site
           SiteMap
           ClockRegion
           ClockRegionMap
           HalfColumnRegion
           ShapeConstr
           ShapeConstr::ShapeElement
           Layout
           Database
           Database::Params
           ModelAreaType
           PlaceParams
           PlaceDB
    )pbdoc";

  using IndexType      = Object::IndexType;
  using CoordinateType = Object::CoordinateType;

  // py::bind_vector<std::vector<CoordinateType>>(m, "VectorCoordinate");
  // py::bind_vector<std::vector<IndexType>>(m, "VectorIndex");
  // py::bind_vector<std::vector<std::string>>(m, "VectorString");
  py::bind_map<std::unordered_map<std::string, IndexType>>(m, "MapString2Index");

  // bind base Object
  py::class_<Object>(m, "Object")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("id", &Object::id)
          .def("setId", (void(Object::*)(IndexType)) & Object::setId)
          .def("__repr__", [](Object const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });

  // bind design related classes
  py::class_<ModelPin, Object>(m, "ModelPin")
          .def(py::init<>())
          .def(py::init<IndexType, std::string>())
          .def("name", &ModelPin::name)
          .def("setName", (void(ModelPin::*)(std::string const &)) & ModelPin::setName)
          .def("offset", &ModelPin::offset)
          .def("setOffset", (void(ModelPin::*)(ModelPin::PointType const &)) & ModelPin::setOffset)
          .def("signalDirect", &ModelPin::signalDirect)
          .def("setSignalDirect", (void(ModelPin::*)(SignalDirection const &)) & ModelPin::setSignalDirect)
          .def("signalType", &ModelPin::signalType)
          .def("setSignalType", (void(ModelPin::*)(SignalType const &)) & ModelPin::setSignalType)
          .def("memory", &ModelPin::memory)
          .def("__repr__", [](ModelPin const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Pin, Object>(m, "Pin")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("attr", (PinAttr & (Pin::*) ()) & Pin::attr, py::return_value_policy::reference_internal)
          .def("modelPinId", &Pin::modelPinId)
          .def("setModelPinId", (void(Pin::*)(IndexType)) & Pin::setModelPinId)
          .def("instId", &Pin::instId)
          .def("setInstId", (void(Pin::*)(IndexType)) & Pin::setInstId)
          .def("netId", &Pin::netId)
          .def("setNetId", (void(Pin::*)(IndexType)) & Pin::setNetId)
          .def("memory", &Pin::memory)
          .def("__repr__", [](Pin const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Inst, Object>(m, "Inst")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("attr", (InstAttr & (Inst::*) ()) & Inst::attr, py::return_value_policy::reference_internal)
          .def("numPins", &Inst::numPins)
          .def("pinId", (void(Inst::*)(IndexType)) & Inst::pinId)
          .def("pinIds", (std::vector<IndexType> & (Inst::*) ()) & Inst::pinIds,
                  py::return_value_policy::reference_internal)
          .def("addPin", (void(Inst::*)(IndexType)) & Inst::addPin)
          .def("memory", &Inst::memory)
          .def("__repr__", [](Inst const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<ModuleInstNetlist>(m, "ModuleInstNetlist")
          .def(py::init<ModuleInst::NetlistType &>())
          // instance
          .def("numInsts", &ModuleInstNetlist::numInsts)
          .def("instId", (IndexType(ModuleInstNetlist::*)(IndexType)) & ModuleInstNetlist::instId)
          .def("instIds", &ModuleInstNetlist::instIds, py::return_value_policy::reference_internal)
          .def("inst", (Inst & (ModuleInstNetlist::*) (IndexType)) & ModuleInstNetlist::inst,
                  py::return_value_policy::reference_internal)
          .def("addInst", (void(ModuleInstNetlist::*)(IndexType)) & ModuleInstNetlist::addInst)
          // net
          .def("numNets", &ModuleInstNetlist::numNets)
          .def("netId", (IndexType(ModuleInstNetlist::*)(IndexType)) & ModuleInstNetlist::netId)
          .def("netIds", &ModuleInstNetlist::netIds, py::return_value_policy::reference_internal)
          .def("net", (Net & (ModuleInstNetlist::*) (IndexType)) & ModuleInstNetlist::net,
                  py::return_value_policy::reference_internal)
          .def("addNet", (void(ModuleInstNetlist::*)(IndexType)) & ModuleInstNetlist::addNet)
          // pin
          .def("numPins", &ModuleInstNetlist::numPins)
          .def("pinId", (IndexType(ModuleInstNetlist::*)(IndexType)) & ModuleInstNetlist::pinId)
          .def("pinIds", &ModuleInstNetlist::pinIds, py::return_value_policy::reference_internal)
          .def("pin", (Pin & (ModuleInstNetlist::*) (IndexType)) & ModuleInstNetlist::pin,
                  py::return_value_policy::reference_internal)
          .def("addPin", (void(ModuleInstNetlist::*)(IndexType)) & ModuleInstNetlist::addPin)
          .def("memory", &ModuleInstNetlist::memory)
          .def("__repr__", [](ModuleInstNetlist const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<ModuleInst, Inst>(m, "ModuleInst")
          .def(py::init<ModuleInst::NetlistType &>())
          .def(py::init<IndexType, ModuleInst::NetlistType &>())
          .def("netlist", (ModuleInstNetlist & (ModuleInst::*) ()) & ModuleInst::netlist,
                  py::return_value_policy::reference_internal)
          .def("setNetlist", (void(ModuleInst::*)(ModuleInstNetlist const &)) & ModuleInst::setNetlist)
          .def("memory", &ModuleInst::memory)
          .def("__repr__", [](ModuleInst const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Net, Object>(m, "Net")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("attr", (NetAttr & (Net::*) ()) & Net::attr, py::return_value_policy::reference_internal)
          .def("numPins", &Net::numPins)
          .def("pinId", (IndexType(Net::*)(IndexType)) & Net::pinId)
          .def("pinIds", (std::vector<IndexType> & (Net::*) ()) & Net::pinIds,
                  py::return_value_policy::reference_internal)
          .def("addPin", (void(Net::*)(IndexType)) & Net::addPin)
          .def("memory", &Net::memory)
          .def("__repr__", [](Net const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Model, Object>(m, "Model")
          .def(py::init<>())
          .def("modelType", &Model::modelType)
          .def("setModelType", (void(Model::*)(ModelType const &)) & Model::setModelType)
          .def("numModelPins", &Model::numModelPins)
          .def(
                  "modelPin", [](Model &rhs, IndexType i) { return rhs.modelPin(i); },
                  py::return_value_policy::reference_internal)
          .def("tryAddModelPin", (ModelPin & (Model::*) (std::string const &) ) & Model::tryAddModelPin,
                  py::return_value_policy::reference_internal)
          .def("modelPinId", (IndexType(Model::*)(std::string const &)) & Model::modelPinId)
          .def(
                  "modelPin", [](Model &rhs, std::string const &name) { return rhs.modelPin(name).get(); },
                  py::return_value_policy::reference_internal)
          .def("name", &Model::name)
          .def("setName", (void(Model::*)(std::string const &)) & Model::setName)
          .def("size", (Model::SizeType const &(Model::*) ()) & Model::size)
          .def("setSize", (void(Model::*)(Model::SizeType const &)) & Model::setSize)
          .def("netlist", (boost::optional<Model::NetlistType> & (Model::*) ()) & Model::netlist,
                  py::return_value_policy::reference_internal)
          .def("setNetlist", (void(Model::*)(Model::NetlistType const &)) & Model::setNetlist)
          // instance
          .def("tryAddInst", (Inst & (Model::*) (std::string const &) ) & Model::tryAddInst,
                  py::return_value_policy::reference_internal)
          .def("instId", (IndexType(Model::*)(std::string const &)) & Model::instId)
          .def(
                  "inst", [](Model &rhs, std::string const &name) { return rhs.inst(name).get(); },
                  py::return_value_policy::reference_internal)
          // net
          .def("tryAddNet", (Net & (Model::*) (std::string const &) ) & Model::tryAddNet,
                  py::return_value_policy::reference_internal)
          .def("netId", (IndexType(Model::*)(std::string const &)) & Model::netId)
          .def(
                  "net", [](Model &rhs, std::string const &name) { return rhs.net(name).get(); },
                  py::return_value_policy::reference_internal)
          // pin
          .def("addPin", (Pin & (Model::*) ()) & Model::addPin, py::return_value_policy::reference_internal)
          .def("memory", &Model::memory)
          .def("__repr__", [](Model const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<NetlistImpl, Object>(m, "Netlist")
          .def(py::init<>())
          // instance
          .def("numInsts", &NetlistImpl::numInsts)
          .def("inst", (Inst & (NetlistImpl::*) (IndexType)) & NetlistImpl::inst,
                  py::return_value_policy::reference_internal)
          .def("insts", (std::vector<Inst> & (NetlistImpl::*) ()) & NetlistImpl::insts,
                  py::return_value_policy::reference_internal)
          .def("addInst", py::overload_cast<>((Inst & (NetlistImpl::*) ()) & NetlistImpl::addInst),
                  py::return_value_policy::reference_internal)
          .def("addInst", py::overload_cast<IndexType>((Inst & (NetlistImpl::*) (IndexType)) & NetlistImpl::addInst),
                  py::return_value_policy::reference_internal)
          .def("addInst",
                  py::overload_cast<Inst const &>((Inst & (NetlistImpl::*) (Inst const &) ) & NetlistImpl::addInst),
                  py::return_value_policy::reference_internal)
          // nets
          .def("numNets", &NetlistImpl::numNets)
          .def("net", (Net & (NetlistImpl::*) (IndexType)) & NetlistImpl::net,
                  py::return_value_policy::reference_internal)
          .def("nets", (std::vector<Net> & (NetlistImpl::*) ()) & NetlistImpl::nets,
                  py::return_value_policy::reference_internal)
          .def("addNet", py::overload_cast<>((Net & (NetlistImpl::*) ()) & NetlistImpl::addNet),
                  py::return_value_policy::reference_internal)
          .def("addNet", py::overload_cast<IndexType>((Net & (NetlistImpl::*) (IndexType)) & NetlistImpl::addNet),
                  py::return_value_policy::reference_internal)
          .def("addNet", py::overload_cast<Net const &>((Net & (NetlistImpl::*) (Net const &) ) & NetlistImpl::addNet),
                  py::return_value_policy::reference_internal)
          // pins
          .def("numPins", &NetlistImpl::numPins)
          .def("pin", (Pin & (NetlistImpl::*) (IndexType)) & NetlistImpl::pin,
                  py::return_value_policy::reference_internal)
          .def("pins", (std::vector<Pin> & (NetlistImpl::*) ()) & NetlistImpl::pins,
                  py::return_value_policy::reference_internal)
          .def("addPin", py::overload_cast<>((Pin & (NetlistImpl::*) ()) & NetlistImpl::addPin),
                  py::return_value_policy::reference_internal)
          .def("addPin", py::overload_cast<IndexType>((Pin & (NetlistImpl::*) (IndexType)) & NetlistImpl::addPin),
                  py::return_value_policy::reference_internal)
          .def("addPin", py::overload_cast<Pin const &>((Pin & (NetlistImpl::*) (Pin const &) ) & NetlistImpl::addPin),
                  py::return_value_policy::reference_internal)
          .def("memory", &NetlistImpl::memory)
          .def("__repr__", [](NetlistImpl const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<BaseAttr, Object>(m, "BaseAttr").def(py::init<>()).def(py::init<IndexType>());
  py::class_<InstAttr, BaseAttr>(m, "InstAttr")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("modelId", &InstAttr::modelId)
          .def("setModelId", (void(InstAttr::*)(IndexType)) & InstAttr::setModelId)
          .def("placeStatus", &InstAttr::placeStatus)
          .def("setPlaceStatus", (void(InstAttr::*)(PlaceStatus const &)) & InstAttr::setPlaceStatus)
          .def("loc", &InstAttr::loc)
          .def("setLoc", (void(InstAttr::*)(InstAttr::PointType const &)) & InstAttr::setLoc)
          .def("name", &InstAttr::name)
          .def("setName", (void(InstAttr::*)(std::string const &)) & InstAttr::setName)
          .def("memory", &InstAttr::memory)
          .def("__repr__", [](InstAttr const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<NetAttr, BaseAttr>(m, "NetAttr")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("weight", &NetAttr::weight)
          .def("setWeight", (void(NetAttr::*)(NetAttr::WeightType)) & NetAttr::setWeight)
          .def("name", &NetAttr::name)
          .def("setName", (void(NetAttr::*)(std::string const &)) & NetAttr::setName)
          .def("memory", &NetAttr::memory)
          .def("__repr__", [](NetAttr const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<PinAttr, BaseAttr>(m, "PinAttr")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("signalDirect", &PinAttr::signalDirect)
          .def("setSignalDirect", (void(PinAttr::*)(SignalDirection const &)) & PinAttr::setSignalDirect)
          .def("memory", &PinAttr::memory)
          .def("__repr__", [](PinAttr const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Design>(m, "Design")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("numModels", &Design::numModels)
          .def("model", (Model & (Design::*) (IndexType)) & Design::model, py::return_value_policy::reference_internal)
          .def("models", &Design::models, py::return_value_policy::reference_internal)
          .def("modelId", (IndexType(Design::*)(std::string const &)) & Design::modelId)
          .def("netlist", &Design::netlist, py::return_value_policy::reference_internal)
          .def("VddVssNetId", &Design::VddVssNetId)
          .def("setNetlist", (void(Design::*)(Design::NetlistType const &)) & Design::setNetlist)
          .def("numModuleInsts", &Design::numModuleInsts)
          .def(
                  "moduleInst", [](Design &rhs, IndexType i) { return rhs.moduleInst(i); },
                  py::return_value_policy::reference_internal)
          .def("moduleInsts", &Design::moduleInsts, py::return_value_policy::reference_internal)
          .def("topModuleInstId", &Design::topModuleInstId)
          .def("setTopModuleInstId", (void(Design::*)(IndexType)) & Design::setTopModuleInstId)
          .def(
                  "topModuleInst", [](Design &rhs) { return rhs.topModuleInst().get(); },
                  py::return_value_policy::reference_internal)
          .def("numShapeConstrs", &Design::numShapeConstrs)
          .def("shapeConstr", (ShapeConstr & (Design::*) (IndexType)) & Design::shapeConstr,
                  py::return_value_policy::reference_internal)
          .def("addShapeConstr", (ShapeConstr & (Design::*) (ShapeConstrType const &) ) & Design::addShapeConstr,
                  py::return_value_policy::reference_internal)
          .def("uniquify", (IndexType(Design::*)(IndexType)) & Design::uniquify)
          .def("memory", &Design::memory)
          .def("__repr__", [](Design const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });

  // bind layout related classes
  py::class_<Resource, Object>(m, "Resource")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("name", &Resource::name)
          .def("setName", (void(Resource::*)(std::string const &)) & Resource::setName)
          .def("numModels", &Resource::numModels)
          .def("modelId", (IndexType(Resource::*)(IndexType)) & Resource::modelId)
          .def("modelIds", &Resource::modelIds)
          .def("memory", &Resource::memory)
          .def("__repr__", [](Resource const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<ResourceMap, Object>(m, "ResourceMap")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("numResources", &ResourceMap::numResources)
          .def("tryAddResource", (Resource & (ResourceMap::*) (std::string const &) ) & ResourceMap::tryAddResource,
                  py::return_value_policy::reference_internal)
          .def("tryAddModelResource",
                  (Resource & (ResourceMap::*) (IndexType, std::string const &) ) & ResourceMap::tryAddModelResource,
                  py::return_value_policy::reference_internal)
          .def("numModels", &ResourceMap::numModels)
          .def("setNumModels", (void(ResourceMap::*)(IndexType)) & ResourceMap::setNumModels)
          .def("resourceId", (IndexType(ResourceMap::*)(std::string const &)) & ResourceMap::resourceId)
          .def(
                  "resource", [](ResourceMap &rhs, IndexType i) { return rhs.resource(i); },
                  py::return_value_policy::reference_internal)
          .def(
                  "resource", [](ResourceMap &rhs, std::string const &name) { return rhs.resource(name).get(); },
                  py::return_value_policy::reference_internal)
          .def("modelResourceIds",
                  (std::vector<IndexType> const &(ResourceMap::*) (IndexType)) & ResourceMap::modelResourceIds,
                  py::return_value_policy::reference_internal)
          .def(
                  "__iter__", [](ResourceMap &rhs) { return py::make_iterator(rhs.begin(), rhs.end()); },
                  py::keep_alive<0, 1>())
          .def("memory", &ResourceMap::memory)
          .def("__repr__", [](ResourceMap const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<SiteType, Object>(m, "SiteType")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("name", &SiteType::name)
          .def("setName", (void(SiteType::*)(std::string const &)) & SiteType::setName)
          .def("numResources", &SiteType::numResources)
          .def("setNumResources", (void(SiteType::*)(IndexType)) & SiteType::setNumResources)
          .def("resourceCapacity", (IndexType(SiteType::*)(IndexType)) & SiteType::resourceCapacity)
          .def("setResourceCapacity", (void(SiteType::*)(IndexType, IndexType)) & SiteType::setResourceCapacity)
          .def("memory", &SiteType::memory)
          .def("__repr__", [](SiteType const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<SiteTypeMap, Object>(m, "SiteTypeMap")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("numSiteTypes", &SiteTypeMap::numSiteTypes)
          .def("tryAddSiteType", (SiteType & (SiteTypeMap::*) (std::string const &) ) & SiteTypeMap::tryAddSiteType,
                  py::return_value_policy::reference_internal)
          .def("siteTypeId", (IndexType(SiteTypeMap::*)(std::string const &)) & SiteTypeMap::siteTypeId)
          .def(
                  "siteType", [](SiteTypeMap &rhs, IndexType i) { return rhs.siteType(i); },
                  py::return_value_policy::reference_internal)
          .def(
                  "siteType", [](SiteTypeMap &rhs, std::string const &name) { return rhs.siteType(name).get(); },
                  py::return_value_policy::reference_internal)
          .def(
                  "__iter__", [](SiteTypeMap &rhs) { return py::make_iterator(rhs.begin(), rhs.end()); },
                  py::keep_alive<0, 1>())
          .def("memory", &SiteTypeMap::memory)
          .def("__repr__", [](SiteTypeMap const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Site, Object>(m, "Site")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("siteMapId", &Site::siteMapId)
          .def("setSiteMapId", (void(Site::*)(Site::PointType const &)) & Site::setSiteMapId)
          .def("bbox", &Site::bbox)
          .def("setBbox", (void(Site::*)(Site::BoxType const &)) & Site::setBbox)
          .def("clockRegionId", &Site::clockRegionId)
          .def("setClockRegionId", (void(Site::*)(IndexType)) & Site::setClockRegionId)
          .def("halfColumnRegionId", &Site::halfColumnRegionId)
          .def("setHalfColumnRegionId", (void(Site::*)(IndexType)) & Site::setHalfColumnRegionId)
          .def("siteTypeId", &Site::siteTypeId)
          .def("setSiteTypeId", (void(Site::*)(IndexType)) & Site::setSiteTypeId)
          .def("memory", &Site::memory)
          .def("__repr__", [](Site const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<SiteMap, Object>(m, "SiteMap")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("width", &SiteMap::width)
          .def("height", &SiteMap::height)
          .def("reshape", (void(SiteMap::*)(IndexType, IndexType)) & SiteMap::reshape)
          .def("at", py::overload_cast<IndexType>(&SiteMap::at))
          .def("at", py::overload_cast<IndexType, IndexType>(&SiteMap::at))
          .def("index1D", (IndexType(SiteMap::*)(IndexType, IndexType)) & SiteMap::index1D)
          .def(
                  "__call__", [](SiteMap &rhs, IndexType i, IndexType j) { return rhs(i, j); },
                  py::return_value_policy::reference_internal)
          .def(
                  "__iter__", [](SiteMap &rhs) { return py::make_iterator(rhs.begin(), rhs.end()); },
                  py::keep_alive<0, 1>())
          .def("tryAddSite", (Site & (SiteMap::*) (IndexType, IndexType)) & SiteMap::tryAddSite,
                  py::return_value_policy::reference_internal)
          .def("size", &SiteMap::size)
          .def("memory", &SiteMap::memory)
          .def("__repr__", [](SiteMap const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<ClockRegion, Object>(m, "ClockRegion")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("name", &ClockRegion::name)
          .def("setName", (void(ClockRegion::*)(std::string const &)) & ClockRegion::setName)
          .def("bbox", &ClockRegion::bbox)
          .def("setBbox", (void(ClockRegion::*)(ClockRegion::BoxType const &)) & ClockRegion::setBbox)
          .def("numHalfColumnRegions", &ClockRegion::numHalfColumnRegions)
          .def("halfColumnRegionId", (IndexType(ClockRegion::*)(IndexType)) & ClockRegion::halfColumnRegionId)
          .def("halfColumnRegionIds", &ClockRegion::halfColumnRegionIds)
          .def("addHalfColumnRegion", (void(ClockRegion::*)(IndexType)) & ClockRegion::addHalfColumnRegion)
          .def("numSites", (IndexType(ClockRegion::*)(SiteType)) & ClockRegion::numSites)
          .def("setNumSites", (void(ClockRegion::*)(SiteType, IndexType)) & ClockRegion::setNumSites)
          .def("incrNumSites", (IndexType(ClockRegion::*)(SiteType)) & ClockRegion::incrNumSites)
          .def("memory", &ClockRegion::memory)
          .def("__repr__", [](ClockRegion const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<ClockRegionMap, Object>(m, "ClockRegionMap")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("width", &ClockRegionMap::width)
          .def("height", &ClockRegionMap::height)
          .def("reshape", (void(ClockRegionMap::*)(IndexType, IndexType)) & ClockRegionMap::reshape)
          .def("at", py::overload_cast<IndexType>(&ClockRegionMap::at), py::return_value_policy::reference_internal)
          .def("at", py::overload_cast<IndexType, IndexType>(&ClockRegionMap::at),
                  py::return_value_policy::reference_internal)
          .def("__call__", (ClockRegion & (ClockRegionMap::*) (IndexType, IndexType)) & ClockRegionMap::operator())
          .def(
                  "__iter__", [](ClockRegionMap &rhs) { return py::make_iterator(rhs.begin(), rhs.end()); },
                  py::keep_alive<0, 1>())
          .def("size", &ClockRegionMap::size)
          .def("memory", &ClockRegionMap::memory)
          .def("__repr__", [](ClockRegionMap const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<HalfColumnRegion, Object>(m, "HalfColumnRegion")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("clockRegionId", &HalfColumnRegion::clockRegionId)
          .def("setClockRegionId", (void(HalfColumnRegion::*)(IndexType)) & HalfColumnRegion::setClockRegionId)
          .def("bbox", &HalfColumnRegion::bbox)
          .def("setBbox", (void(HalfColumnRegion::*)(HalfColumnRegion::BoxType const &)) & HalfColumnRegion::setBbox)
          .def("memory", &HalfColumnRegion::memory)
          .def("__repr__", [](HalfColumnRegion const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<ShapeConstr, Object> shape_constr(m, "ShapeConstr");
  shape_constr.def(py::init<>())
          .def(py::init<IndexType>())
          .def("type", &ShapeConstr::type)
          .def("setType", (void(ShapeConstr::*)(ShapeConstrType const &)) & ShapeConstr::setType)
          .def("numShapeElements", &ShapeConstr::numShapeElements)
          .def("shapeElement", (ShapeConstr::ShapeElement & (ShapeConstr::*) (IndexType)) & ShapeConstr::shapeElement,
                  py::return_value_policy::reference_internal)
          .def("shapeElements", &ShapeConstr::shapeElements, py::return_value_policy::reference_internal)
          .def("addShapeElement",
                  (ShapeConstr::ShapeElement & (ShapeConstr::*) (ShapeConstr::PointType const &, IndexType)) &
                          ShapeConstr::addShapeElement,
                  py::return_value_policy::reference_internal)
          .def("memory", &ShapeConstr::memory)
          .def("__repr__", [](ShapeConstr const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<ShapeConstr::ShapeElement, Object>(shape_constr, "ShapeElement")
          .def(py::init<>())
          .def("offset",
                  (ShapeConstr::PointType const &(ShapeConstr::ShapeElement::*) ()) & ShapeConstr::ShapeElement::offset)
          .def("setOffset", (void(ShapeConstr::ShapeElement::*)(ShapeConstr::PointType const &)) &
                                    ShapeConstr::ShapeElement::setOffset)
          .def("instId", &ShapeConstr::ShapeElement::instId)
          .def("setInstId", (void(ShapeConstr::ShapeElement::*)(IndexType)) & ShapeConstr::ShapeElement::setInstId)
          .def("__repr__", [](ShapeConstr::ShapeElement const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Layout>(m, "Layout")
          .def(py::init<>())
          .def(py::init<IndexType>())
          .def("siteMap", (SiteMap & (Layout::*) ()) & Layout::siteMap, py::return_value_policy::reference_internal)
          .def("addSite", (Site & (Layout::*) (IndexType, IndexType, SiteType)) & Layout::addSite,
                  py::return_value_policy::reference_internal)
          .def("clockRegionMap", (ClockRegionMap & (Layout::*) ()) & Layout::clockRegionMap,
                  py::return_value_policy::reference_internal)
          .def("numHalfColumnRegions", &Layout::numHalfColumnRegions)
          .def("halfColumnRegion", (HalfColumnRegion const &(Layout::*) (IndexType)) & Layout::halfColumnRegion,
                  py::return_value_policy::reference_internal)
          .def("halfColumnRegions", &Layout::halfColumnRegions, py::return_value_policy::reference_internal)
          .def("addHalfColumnRegion", (HalfColumnRegion & (Layout::*) (IndexType)) & Layout::addHalfColumnRegion,
                  py::return_value_policy::reference_internal)
          .def("siteTypeMap", (SiteTypeMap & (Layout::*) ()) & Layout::siteTypeMap,
                  py::return_value_policy::reference_internal)
          .def("siteType", (SiteType & (Layout::*) (Site const &) ) & Layout::siteType,
                  py::return_value_policy::reference_internal)
          .def("resourceMap", (ResourceMap & (Layout::*) ()) & Layout::resourceMap,
                  py::return_value_policy::reference_internal)
          .def("numSites", (IndexType(Layout::*)(IndexType)) & Layout::numSites)
          .def("setNumSites", (void(Layout::*)(IndexType, IndexType)) & Layout::setNumSites)
          .def("memory", &Layout::memory)
          .def("__repr__", [](Layout const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });

  // bind Database
  py::class_<Database> database(m, "Database");
  database.def(py::init<>())
          .def(py::init<IndexType>())
          .def("design", (Design & (Database::*) ()) & Database::design, py::return_value_policy::reference_internal)
          .def("layout", (Layout & (Database::*) ()) & Database::layout, py::return_value_policy::reference_internal)
          .def("params", (Database::Params & (Database::*) ()) & Database::params,
                  py::return_value_policy::reference_internal)
          .def("readBookshelf", (void(Database::*)(std::string const &)) & Database::readBookshelf)
          .def("readXML", (void(Database::*)(std::string const &)) & Database::readXML)
          .def("readVerilog", (void(Database::*)(std::string const &)) & Database::readVerilog)
          .def("readFlexshelf", (void(Database::*)(std::string const &, std::string const &, std::string const &)) &
                                        Database::readFlexshelf)
          .def("readFlexshelfLayout", (void(Database::*)(std::string const &)) & Database::readFlexshelfLayout)
          .def("readFlexshelfDesign",
                  (void(Database::*)(std::string const &, std::string const &)) & Database::readFlexshelfDesign)
          .def("memory", &Database::memory)
          .def("__repr__", [](Database const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });
  py::class_<Database::Params, Object>(database, "Params")
          .def(py::init<>())
          .def_readwrite("arch_half_column_region_width", &Database::Params::arch_half_column_region_width_)
          .def("__repr__", [](Database::Params const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });

  py::class_<ModelAreaType>(m, "ModelAreaType")
          .def(py::init<>())
          .def_readwrite("model_id", &ModelAreaType::model_id_)
          .def_readwrite("area_type_id", &ModelAreaType::area_type_id_)
          .def_readwrite("model_size", &ModelAreaType::model_size_)
          .def("__repr__", [](ModelAreaType const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });

  py::class_<PlaceParams>(m, "PlaceParams")
          .def(py::init<>())
          .def_readwrite("benchmark_name", &PlaceParams::benchmark_name_)
          .def_readwrite("area_type_names", &PlaceParams::area_type_names_)
          .def_readwrite("at_name2id_map", &PlaceParams::at_name2id_map_)
          .def_readwrite("model_area_types", &PlaceParams::model_area_types_)
          .def_readwrite("model_is_lut", &PlaceParams::model_is_lut_)
          .def_readwrite("model_is_ff", &PlaceParams::model_is_ff_)
          .def_readwrite("model_is_clocksource", &PlaceParams::model_is_clocksource_)
          .def_readwrite("resource2area_types", &PlaceParams::resource2area_types_)
          .def_readwrite("resource_categories", &PlaceParams::resource_categories_)
          .def_readwrite("adjust_area_at_ids", &PlaceParams::adjust_area_at_ids_)
          .def_readwrite("CLB_capacity", &PlaceParams::CLB_capacity_)
          .def_readwrite("BLE_capacity", &PlaceParams::BLE_capacity_)
          .def_readwrite("num_ControlSets_per_CLB", &PlaceParams::num_ControlSets_per_CLB_)
          .def_readwrite("x_wirelength_weight", &PlaceParams::x_wirelength_weight_)
          .def_readwrite("y_wirelength_weight", &PlaceParams::y_wirelength_weight_)
          .def_readwrite("x_cnp_bin_size", &PlaceParams::x_cnp_bin_size_)
          .def_readwrite("y_cnp_bin_size", &PlaceParams::y_cnp_bin_size_)
          .def_readwrite("honor_clock_region_constraints", &PlaceParams::honor_clock_region_constraints)
          .def_readwrite("honor_half_column_constraints", &PlaceParams::honor_half_column_constraints)
          .def_readwrite("cnp_utplace2_enable_packing", &PlaceParams::cnp_utplace2_enable_packing_)
          .def_readwrite("cnp_utplace2_max_num_clock_pruning_per_iteration",
                  &PlaceParams::cnp_utplace2_max_num_clock_pruning_per_iteration_)
          .def_readwrite("clock_region_capacity", &PlaceParams::clock_region_capacity_)
          .def_readwrite("enable_carry_chain_packing", &PlaceParams::enable_carry_chain_packing_)
          .def_readwrite("carry_chain_module_name", &PlaceParams::carry_chain_module_name_)
          .def_readwrite("ssr_chain_module_name", &PlaceParams::ssr_chain_module_name_)
          .def("__repr__", [](PlaceParams const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });

  py::class_<PlaceDB>(m, "PlaceDB")
          .def(py::init<Database &, PlaceParams const &>())
          .def(
                  "db", [](PlaceDB const &rhs) { return rhs.db().get(); }, py::return_value_policy::reference_internal)
          .def("numInsts", &PlaceDB::numInsts)
          .def("numNets", &PlaceDB::numNets)
          .def("numPins", &PlaceDB::numPins)
          .def("instPins", &PlaceDB::instPins, py::return_value_policy::reference_internal)
          .def("pin2Inst", py::overload_cast<IndexType>(&PlaceDB::pin2Inst, py::const_))
          .def("pin2Inst", py::overload_cast<>(&PlaceDB::pin2Inst, py::const_),
                  py::return_value_policy::reference_internal)
          .def("netPins", &PlaceDB::netPins, py::return_value_policy::reference_internal)
          .def("pin2Net", py::overload_cast<IndexType>(&PlaceDB::pin2Net, py::const_))
          .def("pin2Net", py::overload_cast<>(&PlaceDB::pin2Net, py::const_),
                  py::return_value_policy::reference_internal)
          .def("instSize", (PlaceDB::SizeType const &(PlaceDB::*) (IndexType)) & PlaceDB::instSize)
          .def("instSizes", &PlaceDB::instSizes, py::return_value_policy::reference_internal)
          .def("isInstLUT", (uint8_t(PlaceDB::*)(IndexType)) & PlaceDB::isInstLUT)
          .def("isInstLUTs", &PlaceDB::isInstLUTs)
          .def("getInstModelId", (IndexType(PlaceDB::*)(IndexType)) & PlaceDB::getInstModelId)
          .def("getInstModelIds", &PlaceDB::getInstModelIds)
          .def("getAreaTypeIndexFromName",
                  (IndexType(PlaceDB::*)(const std::string &)) & PlaceDB::getAreaTypeIndexFromName)
          .def("isInstFF", (bool(PlaceDB::*)(IndexType)) & PlaceDB::isInstFF)
          .def("isInstFFs", &PlaceDB::isInstFFs)
          .def("isResourceLUT", (bool(PlaceDB::*)(IndexType)) & PlaceDB::isResourceLUT)
          .def("isResourceFF", (bool(PlaceDB::*)(IndexType)) & PlaceDB::isResourceFF)
          .def("isInstClockSource", (bool(PlaceDB::*)(IndexType)) & PlaceDB::isInstClockSource)
          .def("instName", (std::string(PlaceDB::*)(IndexType)) & PlaceDB::instName)
          .def("instModelName", (std::string(PlaceDB::*)(IndexType)) & PlaceDB::instModelName)
          .def("nameToInst", (IndexType(PlaceDB::*)(std::string)) & PlaceDB::nameToInst)
          .def("netName", (std::string(PlaceDB::*)(IndexType)) & PlaceDB::netName)
          .def("nameToNet", (IndexType(PlaceDB::*)(std::string)) & PlaceDB::nameToNet)
          .def("pinNameToPinModelId", (IndexType(PlaceDB::*)(IndexType, std::string)) & PlaceDB::pinNameToPinModelId)
          .def("pinidToPinModelId", (IndexType(PlaceDB::*)(IndexType)) & PlaceDB::pinidToPinModelId)
          .def("pinidToPinModelName", (std::string(PlaceDB::*)(IndexType)) & PlaceDB::pinidToPinModelName)
          .def("instResourceCategory", (std::vector<uint8_t>(PlaceDB::*)(IndexType)) & PlaceDB::instResourceCategory)
          .def("instResourceCategories", &PlaceDB::instResourceCategories)
          .def("getClockNetIndex", (std::vector<IndexType>(PlaceDB::*)()) & PlaceDB::getClockNetIndex)
          .def("netIdToClockId", (IndexType(PlaceDB::*)(IndexType)) & PlaceDB::netIdToClockId)
          .def("clockIdToNetId", (IndexType(PlaceDB::*)(IndexType)) & PlaceDB::clockIdToNetId)
          .def("instToClocksCP", (std::vector<std::vector<IndexType>>(PlaceDB::*)()) & PlaceDB::instToClocksCP)
          .def("clockRegionMapSize", &PlaceDB::clockRegionMapSize)
          .def("numClockNets", (IndexType(PlaceDB::*)()) & PlaceDB::numClockNets)
          .def("numHalfColumnRegions", (IndexType(PlaceDB::*)()) & PlaceDB::numHalfColumnRegions)
          .def("resourceCategory", (uint8_t(PlaceDB::*)(IndexType)) & PlaceDB::resourceCategory)
          .def("xyToCrIndex",
                  (IndexType(PlaceDB::*)(const PlaceDB::CoordinateType &, const PlaceDB::CoordinateType &)) &
                          PlaceDB::XyToCrIndex)
          .def("xyToHcIndex",
                  (IndexType(PlaceDB::*)(const PlaceDB::CoordinateType &, const PlaceDB::CoordinateType &)) &
                          PlaceDB::XyToHcIndex)
          .def("resourceCategories", &PlaceDB::resourceCategories)
          .def("CLBCapacity", &PlaceDB::CLBCapacity)
          .def("BLECapacity", &PlaceDB::BLECapacity)
          .def("numControlSetsPerCLB", &PlaceDB::numControlSetsPerCLB)
          .def("halfCLBCapacity", &PlaceDB::halfCLBCapacity)
          .def("numBLEsPerCLB", &PlaceDB::numBLEsPerCLB)
          .def("numBLEsPerHalfCLB", &PlaceDB::numBLEsPerHalfCLB)
          .def("instAreaType", (std::vector<uint8_t> const &(PlaceDB::*) (IndexType)) & PlaceDB::instAreaType,
                  py::return_value_policy::reference_internal)
          .def("instAreaTypes", &PlaceDB::instAreaTypes, py::return_value_policy::reference_internal)
          .def("areaTypeInstGroup",
                  (std::vector<IndexType> const &(PlaceDB::*) (IndexType)) & PlaceDB::areaTypeInstGroup,
                  py::return_value_policy::reference_internal)
          .def("areaTypeInstGroups", &PlaceDB::areaTypeInstGroups, py::return_value_policy::reference_internal)
          .def("instPlaceStatus", py::overload_cast<IndexType>(&PlaceDB::instPlaceStatus, py::const_))
          .def("instPlaceStatus", py::overload_cast<>(&PlaceDB::instPlaceStatus, py::const_),
                  py::return_value_policy::reference_internal)
          .def("instLocs", &PlaceDB::instLocs, py::return_value_policy::reference_internal)
          .def("pinOffset", (PlaceDB::Point2DType(PlaceDB::*)(IndexType)) & PlaceDB::pinOffset)
          .def("pinOffsets", &PlaceDB::pinOffsets, py::return_value_policy::reference_internal)
          .def("pinSignalDirect", (SignalDirection(PlaceDB::*)(IndexType)) & PlaceDB::pinSignalDirect)
          .def("pinSignalDirects", &PlaceDB::pinSignalDirects, py::return_value_policy::reference_internal)
          .def("pinSignalType", (SignalType(PlaceDB::*)(IndexType)) & PlaceDB::pinSignalType)
          .def("pinSignalTypes", &PlaceDB::pinSignalTypes, py::return_value_policy::reference_internal)
          .def("netWeights", &PlaceDB::netWeights, py::return_value_policy::reference_internal)
          .def("diearea", &PlaceDB::diearea, py::return_value_policy::reference_internal)
          .def("binMapDims", &PlaceDB::binMapDims, py::return_value_policy::reference_internal)
          .def("binMapDim", (PlaceDB::DimType const &(PlaceDB::*) (IndexType)) & PlaceDB::binMapDim,
                  py::return_value_policy::reference_internal)
          .def("binMapSizes", &PlaceDB::binMapSizes, py::return_value_policy::reference_internal)
          .def("binMapSize", (PlaceDB::SizeType const &(PlaceDB::*) (IndexType)) & PlaceDB::binMapSize,
                  py::return_value_policy::reference_internal)
          .def("binCapacityMaps", &PlaceDB::binCapacityMaps, py::return_value_policy::reference_internal)
          .def("binCapacityMap",
                  (std::vector<PlaceDB::CoordinateType> const &(PlaceDB::*) (IndexType)) & PlaceDB::binCapacityMap,
                  py::return_value_policy::reference_internal)
          .def("numResources", &PlaceDB::numResources)
          .def("numAreaTypes", &PlaceDB::numAreaTypes)
          .def("resourceAreaTypes", (std::vector<uint8_t>(PlaceDB::*)(Resource const &)) & PlaceDB::resourceAreaTypes)
          .def("modelAreaTypes", py::overload_cast<std::string const &>(&PlaceDB::modelAreaTypes, py::const_))
          .def("modelAreaTypes", py::overload_cast<IndexType>(&PlaceDB::modelAreaTypes, py::const_))
          .def("totalMovableInstAreas",
                  (std::vector<PlaceDB::CoordinateType> const &(PlaceDB::*) ()) & PlaceDB::totalMovableInstAreas,
                  py::return_value_policy::reference_internal)
          .def("totalFixedInstAreas",
                  (std::vector<PlaceDB::CoordinateType> const &(PlaceDB::*) ()) & PlaceDB::totalFixedInstAreas,
                  py::return_value_policy::reference_internal)
          .def("fixedRange", &PlaceDB::fixedRange)
          .def("movableRange", &PlaceDB::movableRange)
          .def("newInstId", (IndexType(PlaceDB::*)(IndexType)) & PlaceDB::newInstId)
          .def("oldInstId", (IndexType(PlaceDB::*)(IndexType)) & PlaceDB::oldInstId)
          .def("collectInstIds", (std::vector<IndexType>(PlaceDB::*)(IndexType)) & PlaceDB::collectInstIds)
          .def("collectSiteBoxes", py::overload_cast<IndexType>(&PlaceDB::collectSiteBoxes, py::const_))
          .def("collectSiteBoxes", py::overload_cast<>(&PlaceDB::collectSiteBoxes, py::const_))
          .def("collectFlattenSiteBoxes",
                  (std::vector<PlaceDB::CoordinateType>(PlaceDB::*)()) & PlaceDB::collectFlattenSiteBoxes)
          .def("collectSiteCapacities", (std::vector<int32_t>(PlaceDB::*)(IndexType)) & PlaceDB::collectSiteCapacities)
          .def("siteMapDim", &PlaceDB::siteMapDim)
          .def("siteBBox", (PlaceDB::BoxType(PlaceDB::*)(IndexType, IndexType)) & PlaceDB::siteBBox)
          .def("chainInsts", &PlaceDB::chainInsts, py::return_value_policy::reference_internal)
          .def("apply",
                  [](PlaceDB &rhs, py::list locs_xyz) {
                    // locs_xyz must be flat
                    std::vector<PlaceDB::Point3DType> locs(locs_xyz.size() / 3);
                    for (std::size_t i = 0; i < locs.size(); ++i) {
                      locs[i].set(locs_xyz[i * 3].cast<PlaceDB::CoordinateType>(),
                              locs_xyz[i * 3 + 1].cast<PlaceDB::CoordinateType>(),
                              locs_xyz[i * 3 + 2].cast<PlaceDB::CoordinateType>());
                    }
                    rhs.apply(locs);
                  })
          .def("writeBookshelfPl", (bool(PlaceDB::*)(std::string const &)) & PlaceDB::writeBookshelfPl)
          .def("writeBookshelfNodes", (bool(PlaceDB::*)(std::string const &)) & PlaceDB::writeBookshelfNodes)
          .def("writeBookshelfNets", (bool(PlaceDB::*)(std::string const &)) & PlaceDB::writeBookshelfNets)
          .def("memory", &PlaceDB::memory)
          .def("__repr__", [](PlaceDB const &rhs) {
            std::stringstream os;
            os << rhs;
            return os.str();
          });

  py::bind_vector<VectorModelPin>(m, "VectorModelPin");
  py::bind_vector<VectorPin>(m, "VectorPin");
  py::bind_vector<VectorPinPtr>(m, "VectorPinPtr");
  py::bind_vector<VectorInst>(m, "VectorInst");
  py::bind_vector<VectorInstPtr>(m, "VectorInstPtr");
  py::bind_vector<VectorModuleInst>(m, "VectorModuleInst");
  py::bind_vector<VectorNet>(m, "VectorNet");
  py::bind_vector<VectorNetPtr>(m, "VectorNetPtr");
  py::bind_vector<VectorModel>(m, "VectorModel");
  py::bind_vector<VectorInstAttr>(m, "VectorInstAttr");
  py::bind_vector<VectorNetAttr>(m, "VectorNetAttr");
  py::bind_vector<VectorPinAttr>(m, "VectorPinAttr");
  py::bind_vector<VectorResource>(m, "VectorResource");
  py::bind_vector<VectorSiteType>(m, "VectorSiteType");
  py::bind_vector<VectorSite>(m, "VectorSite");
  py::bind_vector<VectorClockRegion>(m, "VectorClockRegion");
  py::bind_vector<VectorHalfColumnRegion>(m, "VectorHalfColumnRegion");
  py::bind_vector<VectorShapeConstr>(m, "VectorShapeConstr");
  py::bind_vector<VectorShapeElement>(m, "VectorShapeElement");

  BIND_VECTOR_TOLIST(VectorModelAreaType);
  BIND_VECTOR_TOLIST_LEVELS(VectorModelAreaTypeL2, 2);

#undef BIND_VECTOR_TOLIST
#undef BIND_VECTOR_TOLIST_LEVELS

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "develop";
#endif
}
