#ifndef BOOKSHELFPARSER_DRIVER_H_
#define BOOKSHELFPARSER_DRIVER_H_

#include "bookshelf_database.h"
#include <string>

namespace bookshelfparser {

class Driver {
public:
  Driver(BookshelfDatabase &db);

  /// enable debug output in the flex scanner
  bool trace_scanning;

  /// enable debug output in the bison parser
  bool trace_parsing;

  /// stream name (file or input stream) used for error messages.
  std::string streamname;

  /** Invoke the scanner and parser for a stream.
   * @param in	input stream
   * @param sname	stream name for error messages
   * @return		true if successfully parsed
   */
  bool parse_stream(std::istream &in,
                    std::string const &sname = "stream input");

  /** Invoke the scanner and parser on an input string.
   * @param input	input string
   * @param sname	stream name for error messages
   * @return		true if successfully parsed
   */
  bool parse_string(std::string const &input,
                    std::string const &sname = "string stream");

  /** Invoke the scanner and parser on a file. Use parse_stream with a
   * std::ifstream if detection of file reading errors is required.
   * @param filename	input file name
   * @return		true if successfully parsed
   */
  bool parse_file(std::string const &filename);

  /** Error handling with associated line number. This can be modified to
   * output the error e.g. to a dialog box. */
  void error(const class location &l, std::string const &m);

  /** General error handling. This can be modified to output the error
   * e.g. to a dialog box. */
  void error(std::string const &m);

  virtual ~Driver();

  /* parsing aux file */
  void setLibFileCbk(std::string const &str);
  void setSclFileCbk(std::string const &str);
  void setNodeFileCbk(std::string const &str);
  void setNetFileCbk(std::string const &str);
  void setPlFileCbk(std::string const &str);
  void setWtFileCbk(std::string const &str);
  void setShapeFileCbk(std::string const &str);

  /* parsing lib file */
  void addCellCbk(std::string const &str);
  void addCellInputPinCbk(std::string const &str);
  void addCellOutputPinCbk(std::string const &str);
  void addCellClockPinCbk(std::string const &str);
  void addCellCtrlPinCbk(std::string const &str);
  void addCellCtrlSRPinCbk(std::string const &str);
  void addCellCtrlCEPinCbk(std::string const &str);
  void addCellInputCasPinCbk(std::string const &str);
  void addCellOutputCasPinCbk(std::string const &str);
  void addCellParameterCbk(std::string const &str);

  /* parsing scl file */
  void setSiteTypeCbk(std::string const &str);
  void setSiteNumResourcesCbk(unsigned n);
  void endSiteBlockCbk();
  void setResourceTypeCbk(std::string const &str);
  void addToCellNameListCbk(std::string const &str);
  void addResourceTypeCbk();
  void endResourceTypeBlockCbk();
  void initSiteMapCbk(unsigned w, unsigned h);
  void endSiteMapCbk();
  void setSiteMapEntryCbk(unsigned x, unsigned y, std::string const &site_type);
  void initClockRegionsCbk(unsigned width, unsigned height);
  void addClockRegionCbk(std::string const &name, unsigned xlo, unsigned ylo,
                         unsigned xhi, unsigned yhi, unsigned ymid,
                         unsigned hcxmin);

  /* parsing nodes file */
  void addNodeCbk(std::string const &node_name, std::string const &cell_name);

  /* parsing pl file */
  void setFixedNodeCbk(std::string const &node_name, unsigned x, unsigned y,
                       unsigned z);

  /* parsing nets file */
  void addNetCbk(std::string const &net_name, unsigned degree);
  void addPinCbk(std::string const &node_name,
                 std::string const &cell_pin_name);

  /* parsing shape file */
  void addShapeCbk(std::string const &shape_type);
  void addShapeNodeCbk(unsigned x, unsigned y, std::string const &node_name);

  class Parser *parser;
  class Scanner *scanner;

private:
  BookshelfDatabase &db_;

  std::string site_type_;     ///< temporary storage for site type name
  std::string resource_type_; ///< temporary storage for a resource type name
  std::vector<std::string>
      cell_name_list_; ///< temporary storage for a resource type entry
  std::vector<std::pair<std::string, unsigned>>
      site_resources_; ///< temporary storage for one site block of resources
};

} // End of namespace bookshelfparser

#endif // __BOOKSHELFPARSER_DRIVER_H__
