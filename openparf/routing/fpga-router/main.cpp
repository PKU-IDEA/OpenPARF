#include <pugixml/pugixml.hpp>
#include "database/builder.h"
#include "router/router.h"
// #include "router/net.h"
#include "utils/printer.h"
#include "utils/ispd/parser.h"
#include "test/netgenerator.h"
#include "database/builder_template.h"
#include "router/routegraphbuilder.h"
#include "router/predictmap.h"
#include "clipp/clipp.h"
#include "utils/vpr/vprparser.h"
#include "utils/globalrouteresult.h"
#include "utils/xarch/xarchparser.h"
#include "router/inst.h"
#include "router/localrouter.h"

#include <iostream>
#include <map>

int main(int argc, char **argv) {
    using namespace clipp;

    std::string xmlDoc;
    std::string plDoc;
    std::string nodeDoc;
    std::string netDoc;
    std::string outDoc;
    std::string outNetDoc;
    std::string inNetDoc;
    std::string vprRouteDoc;
    std::string vprRRGDoc;
    std::string printGRDoc;
    std::string loadGRDoc;
    std::string periodString;
    std::string GRWidthString;
    std::string maxRipupIterString;
    std::string printCongestMapIter;

    bool generateNet;
    bool inputNetlist;
    bool runVPR;
    bool loadGR, printGR;
    bool has_period;
    bool timing_driven;
    bool setGRWidth;
    bool setMaxRipupIter;
    bool setPrintCongestMapIter;
    int mttype = 0;
    auto cli = (option("-xml") & value("XML Architecture File", xmlDoc),
                option("-pl") & value("Placement Result", plDoc),
                option("-node") & value(".node Format File from ISPD 16/17 Contest", nodeDoc),
                option("-net") & value(".net Format File from ISPD 16/17 Contest", netDoc),
                option("-out") & value("output XML format file", outDoc),
                option("-g").set(generateNet) & value("Output File of random generate netlist", outNetDoc),
                option("-netlist").set(inputNetlist) & value("Input Netlist file", inNetDoc),
                option("-mt") & value("Multithread type", mttype),
                option("-route") & value("VPR Route File", vprRouteDoc),
                option("-rrg").set(runVPR) & value("VPR RRG File", vprRRGDoc),
                option("-loadGR").set(loadGR) & value("load GR File", loadGRDoc),
                option("-printGR").set(printGR) & value("print GR File", printGRDoc),
                option("-time_period").set(has_period) & value("time period", periodString),
                option("-timing_driven").set(timing_driven),
                option("-gr_max_width").set(setGRWidth) & value("global routing max edge width", GRWidthString),
                option("-reverse_sort_order").set(Pathfinder::reverseSortOrder),
                option("-max_ripup_iteration").set(setMaxRipupIter) & value("max rrr iter", maxRipupIterString),
                option("-print_congestion_map_iteration").set(setPrintCongestMapIter) & value("print congest map iter", printCongestMapIter));


    parse(argc, argv, cli);
    // std::cout << periodString << std::endl;
    if (has_period)
        router::InstList::period = std::stof(periodString.c_str());
    if (setGRWidth)
        router::RouteGraphBuilder::globalGraphMaxWidth = stoi(GRWidthString);
    else
        router::RouteGraphBuilder::globalGraphMaxWidth = 8;
    Pathfinder::isTimingDriven = timing_driven;

    if (setMaxRipupIter)
        router::Router::maxRipupIter = std::stoi(maxRipupIterString);
    else
        router::Router::maxRipupIter = 311;

    if (setPrintCongestMapIter)
        router::Router::printCongestMapIter = std::stoi(printCongestMapIter);
    else
        router::Router::printCongestMapIter = router::Router::maxRipupIter - 1;
    // std::cout << router::InstList::period << std::endl;
    // getchar();

    if (runVPR) {
        // auto graph = parseRRGraph(vprRRGDoc.c_str());
        // auto netlist = parseRouteFile(vprRouteDoc.c_str(), graph);
        // router::Router fpgaRouter(graph, mttype);
        // for (auto net : netlist) fpgaRouter.addNet(net);
        // netlist.clear();
        // fpgaRouter.run();
        // printVPRWirelength(fpgaRouter.getNetlist(), graph);
        return 0;
    }

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xmlDoc.c_str());
    std::cout << result.description() << std::endl;
    std::shared_ptr<database::GridLayout> gridLayout = database::buildGridLayout(doc.child("arch"));
    std::cout << "Mem Peak: " << get_memory_peak() << "M" << std::endl;
    std::cout << "Mem Curr: " << get_memory_current() << "M" << std::endl;

    router::RouteGraphBuilder builder(gridLayout);
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
        std::vector<std::shared_ptr<Net>> netlist = buildNetlist(plDoc.c_str(), netDoc.c_str(), nodeDoc.c_str(), gridLayout->getModuleLibrary(), gridLayout->getModuleLayout(), graph);
        std::cout << "Generate Complete" << std::endl;
        router::printRouteResult(netlist, outNetDoc, graph);
    }
    else if (inputNetlist) {
        std::cout << "Start Building FPGA Router" << std::endl;
        router::Router fpgaRouter(graph, gridLayout, inNetDoc, mttype);
        std::cout << "FPGA Router Build Finish!" << std::endl;
        fpgaRouter.run();
        std::cout << "Start Printing Result" << std::endl;
        router::printRouteResult(fpgaRouter.getNetlist(), outDoc, graph);
    }
    else {
        std::vector<std::shared_ptr<Net>> netlist = buildNetlist(plDoc.c_str(), netDoc.c_str(), nodeDoc.c_str(), gridLayout->getModuleLibrary(), gridLayout->getModuleLayout(), graph);
        router::Router fpgaRouter(graph, gridLayout, mttype);
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
