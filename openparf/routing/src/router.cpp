/*
 * @file          : router.cpp
 * @project       : OpenPARF
 * @author        : Jing Mai (jingmai@pku.edu.cn)
 * @created date  : April 25 2023, 16:48:48, Tuesday
 * @brief         :
 * -----
 * Last Modified: May 05 2023, 19:54:05, Friday
 * Modified By: Jing Mai <jingmai@pku.edu.cn>
 * -----
 * @history :
 * ====================================================================================
 * Date         	By     	(version)	Comments
 * -------------	-------	---------	--------------------------------------------------
 * ====================================================================================
 * Copyright (c) 2020 - 2023 All Right Reserved, PKU-IDEA Group
 */

#include "router/router.h"

#include <util/torch.h>
#include <util/util.h>

#include <pugixml/pugixml.hpp>

#include "clipp/clipp.h"
#include "database/builder.h"
#include "database/builder_template.h"
#include "router/inst.h"
#include "router/localrouter.h"
#include "router/predictmap.h"
#include "router/routegraphbuilder.h"
#include "test/netgenerator.h"
#include "utils/globalrouteresult.h"
#include "utils/ispd/parser.h"
#include "utils/printer.h"
#include "utils/vpr/vprparser.h"
#include "utils/xarch/xarchparser.h"

namespace router {

int routerForward(const std::string &routing_architecture_input,
        const std::string           &pl_path,
        const std::string           &net_path,
        const std::string           &node_path,
        const std::string           &routing_output_path) {
  using namespace clipp;

  // options
  bool        generateNet            = false;
  bool        inputNetlist           = false;
  bool        runVPR                 = false;
  bool        loadGR                 = false;
  bool        printGR                = false;
  bool        has_period             = false;
  bool        timing_driven          = false;
  bool        setGRWidth             = false;
  bool        setMaxRipupIter        = false;
  bool        setPrintCongestMapIter = false;
  int         mttype                 = 0;

  // parameters
  std::string xmlDoc;                // XML Architecture File
  std::string plDoc;                 // Placement Result
  std::string nodeDoc;               // .node Format File from ISPD 16/17 Contest
  std::string netDoc;                // .net Format File from ISPD 16/17 Contest
  std::string outDoc;                // output XML format file
  std::string outNetDoc;             // Output File of random generate netlist
  std::string inNetDoc;              // Input Netlist file
  std::string vprRouteDoc;           // VPR Route File
  std::string vprRRGDoc;             // VPR RRG File
  std::string printGRDoc;            // print GR File
  std::string loadGRDoc;             // load GR File
  std::string periodString;          // time period
  std::string GRWidthString;         // global routing max edge width
  std::string maxRipupIterString;    // max rrr iter
  std::string printCongestMapIter;   // print congest map iter

  xmlDoc  = routing_architecture_input;
  plDoc   = pl_path;
  netDoc  = net_path;
  nodeDoc = node_path;
  outDoc  = routing_output_path;

  std::cout << "xmlDoc: " << xmlDoc << std::endl;
  std::cout << "plDoc: " << plDoc << std::endl;
  std::cout << "netDoc: " << netDoc << std::endl;
  std::cout << "nodeDoc: " << nodeDoc << std::endl;
  std::cout << "outDoc: " << outDoc << std::endl;

  if (has_period) {
    router::InstList::period = std::stof(periodString.c_str());
  }

  if (setGRWidth) {
    router::RouteGraphBuilder::globalGraphMaxWidth = stoi(GRWidthString);
  } else {
    router::RouteGraphBuilder::globalGraphMaxWidth = 8;
  }

  Pathfinder::isTimingDriven = timing_driven;

  if (setMaxRipupIter) {
    router::Router::maxRipupIter = std::stoi(maxRipupIterString);
  } else {
    router::Router::maxRipupIter = 311;
  }

  if (setPrintCongestMapIter) {
    router::Router::printCongestMapIter = std::stoi(printCongestMapIter);
  } else {
    router::Router::printCongestMapIter = router::Router::maxRipupIter - 1;
  }

  if (runVPR) {
    // auto graph = parseRRGraph(vprRRGDoc.c_str());
    // auto netlist = parseRouteFile(vprRouteDoc.c_str(), graph);
    // router::Router fpgaRouter(graph, mttype);
    // for (auto net : netlist) fpgaRouter.addNet(net);
    // netlist.clear();
    // fpgaRouter.run();
    // printVPRWirelength(fpgaRouter.getNetlist(), graph);
    std::cerr << "VPR routing is not supported yet." << std::endl;
    return 0;
  }

  pugi::xml_document     doc;
  pugi::xml_parse_result result = doc.load_file(xmlDoc.c_str());
  std::cout << result.description() << std::endl;
  std::shared_ptr<database::GridLayout> gridLayout = database::buildGridLayout(doc.child("arch"));
  std::cout << "Mem Peak: " << get_memory_peak() << "M" << std::endl;
  std::cout << "Mem Curr: " << get_memory_current() << "M" << std::endl;

  router::RouteGraphBuilder           builder(gridLayout);
  std::shared_ptr<router::RouteGraph> graph = builder.run();
  std::cout << "VertexNum: " << graph->getVertexNum() << " EdgeNum: " << graph->getEdgeNum() << std::endl;
  std::cout << "Mem Peak: " << get_memory_peak() << "M" << std::endl;
  std::cout << "Mem Curr: " << get_memory_current() << "M" << std::endl;

  // router::PredictMap predictMap(graph);
  // predictMap.initPredictMap(gridLayout);
  // std::cout << "Mem Peak: " << get_memory_peak() << "M" << std::endl;
  // std::cout << "Mem Curr: " << get_memory_current() << "M" << std::endl;
  // #if 1
  // exit(-1);
  // #endif

  if (generateNet) {
    std::vector<std::shared_ptr<Net>> netlist = buildNetlist(plDoc.c_str(), netDoc.c_str(), nodeDoc.c_str(),
            gridLayout->getModuleLibrary(), gridLayout->getModuleLayout(), graph);
    std::cout << "Generate Complete" << std::endl;
    router::printRouteResult(netlist, outNetDoc, graph);
  } else if (inputNetlist) {
    std::cout << "Start Building FPGA Router" << std::endl;
    router::Router fpgaRouter(graph, gridLayout, inNetDoc, mttype);
    std::cout << "FPGA Router Build Finish!" << std::endl;
    fpgaRouter.run();
    std::cout << "Start Printing Result" << std::endl;
    router::printRouteResult(fpgaRouter.getNetlist(), outDoc, graph);
  } else {
    std::vector<std::shared_ptr<Net>> netlist = buildNetlist(plDoc.c_str(), netDoc.c_str(), nodeDoc.c_str(),
            gridLayout->getModuleLibrary(), gridLayout->getModuleLayout(), graph);
    router::Router                    fpgaRouter(graph, gridLayout, mttype);
    for (auto net : netlist) fpgaRouter.addNet(net);
    if (loadGR) {
      router::loadGlobalRouteResult(fpgaRouter.getNetlist(), loadGRDoc);
    }
    if (printGR) {
      fpgaRouter.grFileName = outNetDoc;
    } else {
      fpgaRouter.grFileName = "";
    }
    netlist.clear();
    fpgaRouter.run();
    std::cout << "fpga route finished!" << std::endl;
    router::printRouteResult(fpgaRouter.getNetlist(), outDoc, graph);
    std::cout << "print finished" << std::endl;
    // LocalRouter localrouter(graph);
    // localrouter.loadRouteResultFromRouteTree(router::Pathfinder::routetree);
    // localrouter.testRun();
  }
  return 0;
}
}   // namespace router

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &router::routerForward, "Router forward");
}
