#ifndef BOOKSHELFPARSER_DATABASE_H_
#define BOOKSHELFPARSER_DATABASE_H_

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace bookshelfparser {

class BookshelfDatabase {
public:
  /* parsing aux file */
  virtual void setLibFileCbk(const std::string &str) = 0;
  virtual void setSclFileCbk(const std::string &str) = 0;
  virtual void setNodeFileCbk(const std::string &str) = 0;
  virtual void setNetFileCbk(const std::string &str) = 0;
  virtual void setPlFileCbk(const std::string &str) = 0;
  virtual void setWtFileCbk(const std::string &str) = 0;
  virtual void setShapeFileCbk(const std::string &str) = 0;

  /* parsing lib file */
  virtual void addCellCbk(const std::string &str) = 0;
  virtual void addCellInputPinCbk(const std::string &str) = 0;
  virtual void addCellOutputPinCbk(const std::string &str) = 0;
  virtual void addCellClockPinCbk(const std::string &str) = 0;
  virtual void addCellCtrlPinCbk(const std::string &str) = 0;
  virtual void addCellCtrlSRPinCbk(const std::string &str) = 0;
  virtual void addCellCtrlCEPinCbk(const std::string &str) = 0;
  virtual void addCellInputCasPinCbk(const std::string &str) = 0;
  virtual void addCellOutputCasPinCbk(const std::string &str) = 0;
  virtual void addCellParameterCbk(const std::string &str) = 0;

  /* parsing scl file */
  virtual void setSiteResources(
      std::string const &site_type,
      std::vector<std::pair<std::string, unsigned>> const &site_resources) = 0;
  virtual void
  addResourceTypeCbk(std::string const &resource_type,
                     std::vector<std::string> const &cell_list) = 0;
  virtual void endResourceTypeBlockCbk() = 0;
  virtual void initSiteMapCbk(unsigned w, unsigned h) = 0;
  virtual void endSiteMapCbk() = 0;
  virtual void setSiteMapEntryCbk(unsigned x, unsigned y,
                                  std::string const &site_type) = 0;
  virtual void initClockRegionsCbk(unsigned width, unsigned height) = 0;
  virtual void addClockRegionCbk(const std::string &name, unsigned xlo,
                                 unsigned ylo, unsigned xhi, unsigned yhi,
                                 unsigned ymid, unsigned hcxmin) = 0;

  /* parsing nodes file */
  virtual void addNodeCbk(const std::string &node_name,
                          const std::string &cell_name) = 0;

  /* parsing pl file */
  virtual void setFixedNodeCbk(const std::string &node_name, unsigned x,
                               unsigned y, unsigned z) = 0;

  /* parsing nets file */
  virtual void addNetCbk(const std::string &, unsigned degree) = 0;
  virtual void addPinCbk(const std::string &node_name,
                         const std::string &cell_pin_name) = 0;

  /* parsing shape file */
  virtual void addShapeCbk(const std::string &shapeType) = 0;
  virtual void addShapeNodeCbk(unsigned x, unsigned y,
                               const std::string &node_name) = 0;
};

} // namespace bookshelfparser

#endif
