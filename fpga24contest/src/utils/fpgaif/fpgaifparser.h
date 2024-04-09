#ifndef FPGAIFPARSER_H
#define FPGAIFPARSER_H

#include "interchange/DeviceResources.capnp.h"
#include "interchange/LogicalNetlist.capnp.h"
#include "interchange/PhysicalNetlist.capnp.h"
#include "interchange/References.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize-packed.h>

// #include <kj/std/iostream.h>
#include <kj/io.h>
#include <kj/filesystem.h>

#include "database/pin.h"
#include "router/routegraph.h"
#include "router/net.h"
#include "router/router.h"

#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "../../thirdparty/parallel_hashmap/phmap.h"

using namespace router;

class FPGAIFParser {
    public:
        FPGAIFParser() {}
        void loadDevice(std::string fileName);
        void outputDevice2Binary(std::string prefix);
        void loadDeviceFromBinary(std::string prefix);
        void loadNetlist(std::string fileName);

        struct IFTile {
            int nameIdx;
            int row;
            int col;
            bool isInt = 0;

            IFTile() {}
            IFTile(int x, int y, int nameIdx_, bool flag = false) : row(x), col(y), nameIdx(nameIdx_), isInt(flag) {}
        };

        void run();

        struct IFSite {
            std::string name;
            int tileIdx;
            int siteIdx;
        };

        struct IFPip {
            // std::string tileName;
            // std::string wire0Name;
            // std::string wire1Name;
            int tileNameIdx, wire0NameIdx, wire1NameIdx;
            bool forward;
            bool isFixed;
            IFPip() {}
            IFPip(int v0, int v1, int v2, bool v3, bool v4) : tileNameIdx(v0), wire0NameIdx(v1), wire1NameIdx(v2), forward(v3), isFixed(v4){}
            // IFPip(std::string const& v0, std::string const& v1, std::string const& v2, bool v3, bool v4) : tileName(v0), wire0Name(v1), wire1Name(v2), forward(v3), isFixed(v4) {}
        };

        struct edge_meta_data{
            int vertex0;
            int vertex1;
            int wire0NameIdx;
            int wire1NameIdx;
            int tileNameIdx;
        };

        int getVertexFromSitePin(std::string& siteName, std::string& pinName) {
            auto siteNameIdx = deviceStringsMap[siteName];
            auto pinNameIdx = deviceStringsMap[pinName];
            return ifTileWireToVertex[ifSiteToTile[siteNameIdx]][ifSitetypePinToWire[std::make_pair(ifSiteToType[siteNameIdx], pinNameIdx)]];
        }
        // int getVertexFromSitePin(int siteNameIdx, int pinNameIdx){
        //     return ifTileWireToVertex[ifSiteToTile[siteNameIdx]][ifSitetypePinToWire[ifSiteToType[siteNameIdx]][pinNameIdx]];
        // }

        void setXmlResFile(std::string const& fileName) { xmlResFileName = fileName; }
        void setIfResFile(std::string const& fileName) { ifResFileName = fileName; }

        int computeWireLengthScore(INDEX_T x0, INDEX_T x1, INDEX_T y0, INDEX_T y1) {            
            bool isHorizontal = y0 == y1;
            bool isVertical = x0 == x1;

            int xRange = abs(x0 - x1);
            int yRange = abs(y0 - y1);

            return xRange + yRange;
        }

        void printIFResult(std::string const& fileName);

        // int projectSinkVertex(int vertexIdx);

    private:
        std::shared_ptr<RouteGraph> routegraph;
        std::vector<std::shared_ptr<Net>> netlist;
        std::string xmlResFileName;
        std::string ifResFileName;

        std::vector<IFTile> ifTiles;
        phmap::flat_hash_map<int, int> ifTileMap;
        // std::unordered_map<int, std::unordered_map<int, int>> ifTileWireToVertex;
        std::vector<phmap::flat_hash_map<int, int>> ifTileWireToVertex;
        // std::unordered_map<int, std::unordered_map<int, int>> ifSitetypePinToWire;
        phmap::flat_hash_map<std::pair<int, int>, int> ifSitetypePinToWire;
        phmap::flat_hash_map<int, int> ifSiteToTile;
        phmap::flat_hash_map<int, int> ifSiteToType;
        ::capnp::MallocMessageBuilder message;
        std::vector<std::string> deviceStrings;
        phmap::flat_hash_map<std::string, int> deviceStringsMap;
        // std::unordered_map<int, std::unordered_map<int, IFPip>> ifPips;
        std::vector<phmap::flat_hash_map<int, IFPip>> ifPips;
    
        std::vector<std::string> newStrList;
        std::vector<int> childVertex;
        std::vector<int> deviceStringToNewString;
};

#endif //FPGAIFPARSER_H
