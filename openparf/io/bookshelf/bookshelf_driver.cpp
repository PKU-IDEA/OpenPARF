// $Id: driver.cc 39 2008-08-03 10:07:15Z tb $
/** \file driver.cc Implementation of the example::Driver class. */

#include "bookshelf_driver.h"
#include "bookshelf_scanner.h"

namespace bookshelfparser {

Driver::Driver(BookshelfDatabase &db)
    : trace_scanning(false), trace_parsing(false), parser(NULL), scanner(NULL),
      db_(db) {}

Driver::~Driver() {
  delete scanner;
  scanner = NULL;
  delete parser;
  parser = NULL;
}

bool Driver::parse_stream(std::istream &in, std::string const &sname) {
  streamname = sname;

  delete scanner;
  scanner = new Scanner(&in);
  scanner->set_debug(trace_scanning);

  delete parser;
  parser = new Parser(*this);
  parser->set_debug_level(trace_parsing);
  return (parser->parse() == 0);
}

bool Driver::parse_file(std::string const &filename) {
  std::ifstream in(filename.c_str());
  if (!in.good())
    return false;
  return parse_stream(in, filename);
}

bool Driver::parse_string(std::string const &input, std::string const &sname) {
  std::istringstream iss(input);
  return parse_stream(iss, sname);
}

void Driver::error(const class location &l, std::string const &m) {
  std::cerr << l << ": " << m << std::endl;
}

void Driver::error(std::string const &m) { std::cerr << m << std::endl; }

//////////////////////////////////
//                              //
//  Parser Call Back Functions  //
//                              //
//////////////////////////////////

void Driver::setLibFileCbk(std::string const &str) { db_.setLibFileCbk(str); }
void Driver::setSclFileCbk(std::string const &str) { db_.setSclFileCbk(str); }
void Driver::setNodeFileCbk(std::string const &str) { db_.setNodeFileCbk(str); }
void Driver::setNetFileCbk(std::string const &str) { db_.setNetFileCbk(str); }
void Driver::setPlFileCbk(std::string const &str) { db_.setPlFileCbk(str); }
void Driver::setWtFileCbk(std::string const &str) { db_.setWtFileCbk(str); }
void Driver::setShapeFileCbk(std::string const &str) {
  db_.setShapeFileCbk(str);
}
void Driver::addCellCbk(std::string const &name) { db_.addCellCbk(name); }
void Driver::addCellInputPinCbk(std::string const &pinName) {
  db_.addCellInputPinCbk(pinName);
}
void Driver::addCellOutputPinCbk(std::string const &pinName) {
  db_.addCellOutputPinCbk(pinName);
}
void Driver::addCellClockPinCbk(std::string const &pinName) {
  db_.addCellClockPinCbk(pinName);
}
void Driver::addCellCtrlPinCbk(std::string const &pinName) {
  db_.addCellCtrlPinCbk(pinName);
}
void Driver::addCellCtrlSRPinCbk(std::string const &pinName) {
  db_.addCellCtrlSRPinCbk(pinName);
}
void Driver::addCellCtrlCEPinCbk(std::string const &pinName) {
  db_.addCellCtrlCEPinCbk(pinName);
}
void Driver::addCellInputCasPinCbk(std::string const &pinName) {
  db_.addCellInputCasPinCbk(pinName);
}
void Driver::addCellOutputCasPinCbk(std::string const &pinName) {
  db_.addCellOutputCasPinCbk(pinName);
}
void Driver::addCellParameterCbk(std::string const &param) {
  db_.addCellParameterCbk(param);
}
void Driver::setSiteTypeCbk(std::string const &str) { site_type_ = str; }
void Driver::setSiteNumResourcesCbk(unsigned n) {
  site_resources_.emplace_back(resource_type_, n);
}
void Driver::endSiteBlockCbk() {
  db_.setSiteResources(site_type_, site_resources_);
  site_resources_.clear();
}
void Driver::setResourceTypeCbk(std::string const &str) {
  resource_type_ = str;
}
void Driver::addToCellNameListCbk(std::string const &str) {
  cell_name_list_.push_back(str);
}
void Driver::addResourceTypeCbk() {
  db_.addResourceTypeCbk(resource_type_, cell_name_list_);
  resource_type_.clear();
  cell_name_list_.clear();
}
void Driver::endResourceTypeBlockCbk() { db_.endResourceTypeBlockCbk(); }
void Driver::initSiteMapCbk(unsigned w, unsigned h) {
  db_.initSiteMapCbk(w, h);
}
void Driver::endSiteMapCbk() { db_.endSiteMapCbk(); }
void Driver::setSiteMapEntryCbk(unsigned x, unsigned y,
                                std::string const &site_type) {
  db_.setSiteMapEntryCbk(x, y, site_type);
}
void Driver::initClockRegionsCbk(unsigned w, unsigned h) {
  db_.initClockRegionsCbk(w, h);
}
void Driver::addNodeCbk(std::string const &node_name,
                        std::string const &cell_name) {
  db_.addNodeCbk(node_name, cell_name);
}
void Driver::setFixedNodeCbk(std::string const &node_name, unsigned x,
                             unsigned y, unsigned z) {
  db_.setFixedNodeCbk(node_name, x, y, z);
}
void Driver::addNetCbk(std::string const &net_name, unsigned degree) {
  db_.addNetCbk(net_name, degree);
}
void Driver::addPinCbk(std::string const &node_name,
                       std::string const &cell_pin_name) {
  db_.addPinCbk(node_name, cell_pin_name);
}

void Driver::addClockRegionCbk(std::string const &name, unsigned xlo,
                               unsigned ylo, unsigned xhi, unsigned yhi,
                               unsigned ymid, unsigned hcxmin) {
  db_.addClockRegionCbk(name, xlo, ylo, xhi, yhi, ymid, hcxmin);
}

void Driver::addShapeCbk(std::string const &shape_type) {
  db_.addShapeCbk(shape_type);
}
void Driver::addShapeNodeCbk(unsigned x, unsigned y,
                             std::string const &node_name) {
  db_.addShapeNodeCbk(x, y, node_name);
}

} // namespace bookshelfparser
