#include "router/router.h"
#include "utils/printer.h"
#include "thirdparty/clipp/clipp.h"
#include "utils/fpgaif/fpgaifparser.h"

#include <iostream>
#include <map>
#include <chrono>

#include <queue>

using namespace std::chrono;

int main(int argc, char **argv) {
    using namespace clipp;

    std::string ifDeviceDoc;
    std::string ifPhysNetDoc;
    std::string ifOutputDoc;
    std::string rrgFolder;


    bool generateNet = false;
    bool inputNetlist = false;
    bool runVPR = false;
    bool loadGR, printGR = false;
    bool runIF = false;
    bool loadRRG = false;
    bool printHelp = false;

    int runSplit = 0;
    int mttype = 0;
    int pruneN = 0;
    auto cli = (option("-rrg").set(loadRRG) & value("dumped rrg folder", rrgFolder),
                option("-device") & value("IF device", ifDeviceDoc),
                option("-phys") & value("IF phys", ifPhysNetDoc),
                option("-ifout") & value("IF Output phys", ifOutputDoc));


    parse(argc, argv, cli);
    
    if (argc == 1) {
        std::cout << "[Usage] ./fpgarouter -rrg /path/to/rrg_folder -phys /path/to/unrouted.phys -ifout /path/to/output.phys" << std::endl;
        std::cout << "[Usage] ./fpgarouter -device /path/to/input.device -phys /path/to/unrouted.phys -ifout /path/to/output.phys" << std::endl; 
        return 0;
    }

    FPGAIFParser parser;
    high_resolution_clock::time_point t0, t1, t2, t3;
    t0 = high_resolution_clock::now();
    if (!loadRRG)
        parser.loadDevice(ifDeviceDoc);
    else parser.loadDeviceFromBinary(rrgFolder + "/rrg");
    parser.loadNetlist(ifPhysNetDoc);
    t1 = high_resolution_clock::now();
    parser.run();
    t2 = high_resolution_clock::now();
    parser.printIFResult(ifOutputDoc);
    t3 = high_resolution_clock::now();
    
    duration<double, std::ratio<1, 1> > duration_load(t1 - t0);
    duration<double, std::ratio<1, 1> > duration_route(t2 - t1);
    duration<double, std::ratio<1, 1> > duration_write(t3 - t2);
    duration<double, std::ratio<1, 1> > duration_total(t3 - t0);
    std::cout << "Total: " << duration_total.count() << " s\n"
            << "Load Data: " << duration_load.count() << " s\n"
            << "Route: " << duration_route.count() << " s\n"
            << "Write Data: " << duration_write.count() << " s" << std::endl;

    return 0;
    
}
