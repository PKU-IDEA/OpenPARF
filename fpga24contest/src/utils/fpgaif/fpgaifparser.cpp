#include "fpgaifparser.h"

#include "capnp/message.h"
#include "router/pathfinder.h"
#include "utils/printer.h"
#include "utils/utils.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <zlib.h>

static constexpr int numThreads = 32;

void gzipCompress(const std::vector<char> &data,
                  std::vector<char> &compressedData) {
    z_stream deflateStream;
    deflateStream.zalloc = Z_NULL;
    deflateStream.zfree = Z_NULL;
    deflateStream.opaque = Z_NULL;
    deflateStream.avail_in = data.size();
    deflateStream.next_in = (Bytef *)data.data();

    // 初始化用于压缩的流
    if (deflateInit2(&deflateStream, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                     MAX_WBITS + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("Failed to initialize zlib deflate stream.");
    }

    // std::vector<char> compressedData;
    char buffer[4096];
    do {
        deflateStream.avail_out = sizeof(buffer);
        deflateStream.next_out = reinterpret_cast<Bytef *>(buffer);
        deflate(&deflateStream, Z_FINISH);
        size_t have = sizeof(buffer) - deflateStream.avail_out;
        compressedData.insert(compressedData.end(), buffer, buffer + have);
    } while (deflateStream.avail_out == 0);

    // 清理
    deflateEnd(&deflateStream);

    // return compressedData;
}

void FPGAIFParser::loadDevice(std::string fileName) {
    // std::ifstream ifs(fileName, std::ios::binary | std::ios::ate);
    // std::streamsize size = ifs.tellg();
    // ifs.seekg(0, std::ios::beg);

    // std::vector<char> buffer(size);
    // ifs.read(buffer.data(), size);
    // kj::ArrayPtr<const capnp::word> words(reinterpret_cast<const
    // capnp::word*>(buffer.data()), size / sizeof(capnp::word));
    // // kj::BufferedInputStream is;
    // // capnp::PackedMessageReader reader(is);
    // // auto fs = kj::newDiskFilesystem();
    // // auto file = fs->getRoot().openFile(kj::Path::parse(fileName.c_str()));
    // // auto mmap = file->mmap(0, file->stat().size);
    // // auto wordPtr = reinterpret_cast<const capnp::word*>(mmap.begin());
    // // kj::ArrayPtr<const capnp::word> wordArray(wordPtr, mmap.size() /
    // sizeof(capnp::word)); capnp::FlatArrayMessageReader reader(words); int fd
    // = open(fileName.c_str(), O_RDONLY); kj::FdInputStream fdStream(fd);
    auto dev_st = std::chrono::high_resolution_clock::now();
    gzFile gz = gzopen(fileName.c_str(), "rb");
    if (!gz) {
        std::cerr << "failed to open file" << std::endl;
        exit(-1);
    }
    std::vector<char> decompressedData;
    char buffer[4096];

    int bytesRead;
    while ((bytesRead = gzread(gz, buffer, sizeof(buffer))) > 0) {
        decompressedData.insert(decompressedData.end(), buffer,
                                buffer + bytesRead);
    }

    gzclose(gz);

    if (decompressedData.size() % sizeof(capnp::word) != 0) {
        throw std::runtime_error(
            "Decompressed data size is not a multiple of word size");
    }

    capnp::ReaderOptions option;
    option.traversalLimitInWords = std::numeric_limits<uint64_t>::max();
    option.nestingLimit = 64;

    capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(
            reinterpret_cast<const capnp::word *>(decompressedData.data()),
            decompressedData.size() / sizeof(capnp::word)),
        option);

    // auto deviceResource = reader.getRoot<DeviceResources>();
    auto device = reader.getRoot<DeviceResources::Device>();
    auto tiles = device.getTileList();
    auto tiletypes = device.getTileTypeList();
    auto nodes = device.getNodes();
    auto nodeTimings = device.getNodeTimings();
    auto strings = device.getStrList();
    auto wires = device.getWires();
    auto wireTypes = device.getWireTypes();

    for (int i = 0; i < strings.size(); i++) {
        deviceStrings.push_back(std::string(strings[i].cStr()));
        deviceStringsMap[deviceStrings.back()] = deviceStrings.size() - 1;
    }

    ifTileWireToVertex.resize(strings.size());

    int width = 0, height = 0;
    for (auto tile : tiles) {
        auto nameIdx = tile.getName();
        auto name = std::string(strings[nameIdx].cStr());
        int x, y;
        const char *s = strrchr(name.c_str(), 'X');
        sscanf(s, "X%dY%d", &x, &y);
        // char s[100];
        // sscanf(name.c_str(), "%[^0-9]%dY%d", s, &x, &y);
        // if (name.find("NULL") != std::string::npos)
        //     continue;
        ifTiles.emplace_back(x, y, nameIdx,
                             (name.find("INT_X") != std::string::npos));
        width = std::max(width, (int)x);
        height = std::max(height, (int)y);
        ifTileMap[nameIdx] = ifTiles.size() - 1;
    }

    std::vector<std::thread> thread_pools(numThreads);
    std::vector<std::mutex> mtx_0(nodes.size()), mtx_1(nodes.size()),
        mtx_2((width + 1) * (height + 1));

    std::cout << "device width: " << width + 1 << " height: " << height + 1
              << std::endl;

    routegraph =
        std::make_shared<RouteGraph>(width + 1, height + 1, nodes.size());
    routegraph->setVertexNum(nodes.size());

    {
        auto add_node_func = [&](int nodeIdx) {
            auto node = nodes[nodeIdx];
            auto nodeWires = node.getWires();
            auto wire = wires[nodeWires[0]];
            auto tileNameIdx = wire.getTile(), nodeNameIdx = wire.getWire();
            auto nodeType = wire.getType();
            auto nodeName = std::string(strings[nodeNameIdx].cStr());
            auto nodeTypeName = std::string(
                strings[wireTypes[wire.getType()].getName()].cStr());
            if (ifTileMap.find(tileNameIdx) == ifTileMap.end())
                return;
            auto &tile = ifTiles[ifTileMap[tileNameIdx]];
            std::shared_ptr<database::Pin> pin =
                std::make_shared<database::Pin>(nodeName, tile.row, tile.col,
                                                nullptr);
            int graphIdx =
                routegraph->addVertex(nodeIdx, tile.row, tile.col, tile.row,
                                      tile.col, pin, 1, (IntentCode)nodeType);
            if (tile.isInt)
                routegraph->setVertexType(graphIdx, INTTILE);
            for (auto wireIdx : nodeWires) {
                auto wire = wires[wireIdx];
                auto wireTileNameIdx = wire.getTile(),
                     wireNameIdx = wire.getWire();
                auto wireTileName = std::string(strings[wire.getTile()].cStr());
                auto wireName = std::string(strings[wire.getWire()].cStr());
                auto nodeType = wire.getType();
                std::string nodeTypeName = std::string(
                    strings[wireTypes[wire.getType()].getName()].cStr());
                if (nodeTypeName == "NODE_PINFEED" && tile.isInt)
                    routegraph->setVertexType(graphIdx, NETSINK);
                if (numThreads > 1) {
                    mtx_0[wireTileNameIdx].lock();
                }
                ifTileWireToVertex[wireTileNameIdx][wireNameIdx] = graphIdx;
                if (numThreads > 1) {
                    mtx_0[wireTileNameIdx].unlock();
                }
            }
        };

        auto thread_add_node_func = [&](int threadIdx) {
            for (int nodeIdx = threadIdx; nodeIdx < nodes.size();
                 nodeIdx += numThreads) {
                add_node_func(nodeIdx);
            }
        };

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
            thread_pools[threadIdx] =
                std::thread(thread_add_node_func, threadIdx);
        }

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
            thread_pools[threadIdx].join();
        }
    }

    std::cout << "routegraph vertex num: " << routegraph->getVertexNum()
              << std::endl;
    ifPips.resize(routegraph->getVertexNum());

    auto siteTypeList = device.getSiteTypeList();

    std::vector<uint32_t> tileEdgeOffsets;
    tileEdgeOffsets.push_back(0);
    for (auto tile : tiles) {
        auto tileNameIdx = tile.getName();
        if (ifTileMap.find(tileNameIdx) == ifTileMap.end()) {
            tileEdgeOffsets.push_back(0);
            continue;
        }
        tileEdgeOffsets.push_back(tiletypes[tile.getType()].getPips().size());
    }

    for (int tileIdx = 1; tileIdx < tileEdgeOffsets.size(); tileIdx++) {
        tileEdgeOffsets[tileIdx] += tileEdgeOffsets[tileIdx - 1];
    }

    routegraph->setEdgeNum(tileEdgeOffsets[tiles.size()]);

    {
        std::vector<std::vector<edge_meta_data>> bi_edges(numThreads);
        auto add_edge_func = [&](DeviceResources::Device::Tile::Reader tile,
                                 int offset, int threadIdx) {
            auto tileNameIdx = tile.getName();
            auto tileIdx = ifTileMap.find(tileNameIdx)->second;
            auto tileType = tiletypes[tile.getType()];
            auto tileWires = tileType.getWires();
            auto tilePips = tileType.getPips();
            for (int pipIdx = 0; pipIdx < tilePips.size(); pipIdx++) {
                auto pip = tilePips[pipIdx];
                auto wire0NameIdx = pip.getWire0();
                wire0NameIdx = tileWires[wire0NameIdx];
                auto wire0Name = std::string(strings[wire0NameIdx].cStr());
                int vertex0 = ifTileWireToVertex[tileNameIdx][wire0NameIdx];
                auto wire1NameIdx = pip.getWire1();
                wire1NameIdx = tileWires[wire1NameIdx];
                auto wire1Name = std::string(strings[wire1NameIdx].cStr());
                int vertex1 = ifTileWireToVertex[tileNameIdx][wire1NameIdx];
                if (numThreads > 1) {
                    mtx_0[vertex0].lock();
                }
                ifPips[vertex0][vertex1] =
                    IFPip(tileNameIdx, wire0NameIdx, wire1NameIdx, 1, 1);
                routegraph->addEdge(
                    pipIdx + offset, vertex0, vertex1,
                    computeWireLengthScore(routegraph->getPos(vertex0).X(),
                                           routegraph->getPos(vertex1).X(),
                                           routegraph->getPos(vertex0).Y(),
                                           routegraph->getPos(vertex1).Y()));
                // routegraph->setVertexLengthAndBaseCost(
                //     vertex0,
                //     computeWireLengthScore(routegraph->getPos(vertex0).X(),
                //                            routegraph->getPos(vertex1).X(),
                //                            routegraph->getPos(vertex0).Y(),
                //                            routegraph->getPos(vertex1).Y()));
                // routegraph->edgeOut(pipIdx + offset);
                if (numThreads > 1) {
                    mtx_0[vertex0].unlock();
                    //     mtx_1[vertex1].lock();
                }
                // routegraph->edgeIn(pipIdx + offset);
                // if (numThreads > 1) {
                //     mtx_1[vertex1].unlock();
                //     mtx_2[routegraph->getPos(vertex0).X() * height +
                //           routegraph->getPos(vertex0).Y()]
                //         .lock();
                // }
                // globalroutegraph->addEdge(
                //     routegraph->getPos(vertex0).X(),
                //     routegraph->getPos(vertex0).Y(),
                //     routegraph->getPos(vertex1).X(),
                //     routegraph->getPos(vertex1).Y(),
                //     computeWireLengthScore(routegraph->getPos(vertex0).X(),
                //                            routegraph->getPos(vertex1).X(),
                //                            routegraph->getPos(vertex0).Y(),
                //                            routegraph->getPos(vertex1).Y()));
                // if (numThreads > 1) {
                //     mtx_2[routegraph->getPos(vertex0).X() * height +
                //           routegraph->getPos(vertex0).Y()]
                //         .unlock();
                // }
                if (!pip.getDirectional()) {
                    edge_meta_data emd;
                    emd.vertex0 = vertex0;
                    emd.vertex1 = vertex1;
                    emd.tileNameIdx = tileNameIdx;
                    emd.wire0NameIdx = wire0NameIdx;
                    emd.wire1NameIdx = wire1NameIdx;
                    bi_edges[threadIdx].push_back(emd);
                }
            }
        };

        auto thread_func = [&](int threadIdx) {
            for (int i = threadIdx; i < tiles.size(); i += numThreads) {
                add_edge_func(tiles[i], tileEdgeOffsets[i], threadIdx);
            }
        };

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
            thread_pools[threadIdx] = std::thread(thread_func, threadIdx);
        }

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
            thread_pools[threadIdx].join();
        }
        for (int i = 0; i < routegraph->getEdgeNum(); i++) {
            auto &edge = routegraph->getEdge(i);
            routegraph->setVertexLengthAndBaseCost(
                edge.from,
                computeWireLengthScore(routegraph->getPos(edge.from).X(),
                                       routegraph->getPos(edge.to).X(),
                                       routegraph->getPos(edge.from).Y(),
                                       routegraph->getPos(edge.to).Y()));
            routegraph->edgeIn(i);
            routegraph->edgeOut(i);
        }

        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
            for (auto &emd : bi_edges[threadIdx]) {
                int vertex0 = emd.vertex0, vertex1 = emd.vertex1;
                routegraph->addEdge(
                    vertex1, vertex0,
                    computeWireLengthScore(routegraph->getPos(vertex0).X(),
                                           routegraph->getPos(vertex1).X(),
                                           routegraph->getPos(vertex0).Y(),
                                           routegraph->getPos(vertex1).Y()));
                ifPips[vertex1][vertex0] = IFPip(
                    emd.tileNameIdx, emd.wire0NameIdx, emd.wire1NameIdx, 0, 1);
            }
        }
    }

    for (auto tile : tiles) {
        auto tileNameIdx = tile.getName();
        auto tileName = std::string(strings[tileNameIdx].cStr());
        if (ifTileMap.find(tileNameIdx) == ifTileMap.end())
            continue;
        auto tileType = tiletypes[tile.getType()];
        auto tileWires = tileType.getWires();
        auto tilePiPs = tileType.getPips();
        if (!tile.hasSites())
            continue;
        auto tileSites = tile.getSites();
        auto tileSiteTypes = tileType.getSiteTypes();
        for (auto tileSite : tileSites) {
            auto siteNameIdx = tileSite.getName();
            auto siteName = std::string(strings[siteNameIdx].cStr());
            auto siteType = std::string(
                strings[siteTypeList[tileSiteTypes[tileSite.getType()]
                                         .getPrimaryType()]
                            .getName()]
                    .cStr());
            ifSiteToType[siteNameIdx] =
                tile.getType() * 1000 + tileSite.getType();
            ifSiteToTile[siteNameIdx] = tileNameIdx;
        }
    }
    int tileId = 0;
    for (auto tileType : tiletypes) {
        auto siteTypes = tileType.getSiteTypes();
        int id = 0;
        for (auto siteType : siteTypes) {
            auto pinToWire = siteType.getPrimaryPinsToTileWires();
            auto pins = siteTypeList[siteType.getPrimaryType()].getPins();
            assert(pins.size() == pinToWire.size());
            auto siteName = std::string(
                strings[siteTypeList[siteType.getPrimaryType()].getName()]
                    .cStr());
            // std::cout << "----" << siteName << std::endl;
            int sz = pins.size();
            for (int i = 0; i < sz; i++) {
                auto pinNameIdx = pins[i].getName();
                auto pinName = std::string(strings[pins[i].getName()].cStr());
                auto WireName = std::string(strings[pinToWire[i]].cStr());
                // if (siteName == "HPIOB_M" && pinName == "I") std::cout <<
                // siteName << ' ' << pinName << ' ' << pins[i].getName() <<  '
                // ' << WireName << std::endl;
                ifSitetypePinToWire[std::make_pair(tileId * 1000 + id,
                                                   pinNameIdx)] = pinToWire[i];
            }
            id++;
        }
        tileId++;
    }
    std::cout << "routegraph edge num: " << routegraph->getEdgeNum()
              << std::endl;

    auto dev_ed = std::chrono::high_resolution_clock::now();

    std::cout << "load device : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(dev_ed -
                                                                       dev_st)
                     .count()
              << " ms " << std::endl;
    // // dump global route graph
    // globalroutegraph->dumpGraph();
    // std::cout << "check connection: " <<
    // globalroutegraph->checkConnect(14055, 14059) << std::endl; std::cout <<
    // "14055 -> degree: " <<  globalroutegraph->getDegree(14055) << std::endl;
    // std::cout << "14059 -> degree: " <<  globalroutegraph->getDegree(14059)
    // << std::endl; ifs.close();
}

void FPGAIFParser::outputDevice2Binary(std::string prefix) {
    FILE *fp = fopen((prefix + ".strings").c_str(), "w");
    int width = routegraph->getWidth();
    int height = routegraph->getHeight();
    int vertexNum = routegraph->getVertexNum();
    fprintf(fp, "%d\n", width);
    fprintf(fp, "%d\n", height);
    fprintf(fp, "%d\n", vertexNum);
    fprintf(fp, "%d\n", (int)deviceStrings.size());
    for (auto &s : deviceStrings) {
        fprintf(fp, "%s\n", s.c_str());
    }
    fclose(fp);
    fp = fopen((prefix + ".vertices").c_str(), "wb");
    auto &vertices = routegraph->getVertices();
    for (auto &v : vertices) {
        v.pin = nullptr;
    }
    fwrite(vertices.data(), sizeof(RouteGraphVertex), vertexNum, fp);
    fclose(fp);
    fp = fopen((prefix + ".edges").c_str(), "wb");
    auto &edges = routegraph->getEdges();
    int edgeNum = edges.size();
    fwrite(&edgeNum, sizeof(edgeNum), 1, fp);
    fwrite(edges.data(), sizeof(RouteGraphEdge), edgeNum, fp);
    fclose(fp);
    fp = fopen((prefix + ".tw2v").c_str(), "wb");
    int tw2v_size = ifTileWireToVertex.size();
    std::vector<std::pair<std::pair<int, int>, int>> tw2v_vec;
    for (int i = 0; i < tw2v_size; i++) {
        for (auto it : ifTileWireToVertex[i]) {
            tw2v_vec.push_back(
                std::make_pair(std::make_pair(i, it.first), it.second));
        }
    }
    tw2v_size = tw2v_vec.size();
    fwrite(&tw2v_size, sizeof(int), 1, fp);
    fwrite(tw2v_vec.data(), sizeof(std::pair<std::pair<int, int>, int>),
           tw2v_vec.size(), fp);
    fclose(fp);
    fp = fopen((prefix + ".stp2w").c_str(), "wb");
    std::vector<std::pair<std::pair<int, int>, int>> stp2w_vec;
    for (auto it : ifSitetypePinToWire) {
        stp2w_vec.push_back(std::make_pair(it.first, it.second));
    }
    int stp2w_size = stp2w_vec.size();
    fwrite(&stp2w_size, sizeof(int), 1, fp);
    fwrite(stp2w_vec.data(), sizeof(std::pair<std::pair<int, int>, int>),
           stp2w_vec.size(), fp);
    fclose(fp);
    fp = fopen((prefix + ".s2ti").c_str(), "wb");
    std::vector<std::pair<int, int>> s2ti_vec;
    for (auto it : ifSiteToTile) {
        s2ti_vec.push_back(it);
    }
    int s2ti_size = s2ti_vec.size();
    fwrite(&s2ti_size, sizeof(int), 1, fp);
    fwrite(s2ti_vec.data(), sizeof(std::pair<int, int>), s2ti_size, fp);
    fclose(fp);
    fp = fopen((prefix + ".s2ty").c_str(), "wb");
    std::vector<std::pair<int, int>> s2ty_vec;
    for (auto it : ifSiteToType) {
        s2ty_vec.push_back(it);
    }
    int s2ty_size = s2ty_vec.size();
    fwrite(&s2ty_size, sizeof(int), 1, fp);
    fwrite(s2ty_vec.data(), sizeof(std::pair<int, int>), s2ty_size, fp);
    fclose(fp);
    for (int o = 0; o < 4; o++) {
        fp = fopen((prefix + ".pips_" + std::to_string(o)).c_str(), "wb");
        auto length = (ifPips.size() + 3) / 4;
        std::vector<std::pair<std::pair<int, int>, IFPip>> pip_vec;
        for (int i = o * length; i < ifPips.size() && i < o * length + length;
             i++) {
            for (auto it : ifPips[i]) {
                pip_vec.push_back(
                    std::make_pair(std::make_pair(i, it.first), it.second));
            }
        }
        int pip_size = pip_vec.size();
        fwrite(&pip_size, sizeof(int), 1, fp);
        fwrite(pip_vec.data(), sizeof(std::pair<std::pair<int, int>, IFPip>),
               pip_size, fp);
        fclose(fp);
    }
}

void FPGAIFParser::loadDeviceFromBinary(std::string prefix) {
    int stringSize = 0;
    int width = 0;
    int height = 0;
    int vertexNum = 0;
    {
        FILE *fp = fopen((prefix + ".strings").c_str(), "r");
        if (fp == nullptr) {
            std::cout << " [ERROR] " << __FILE__ << " " << __LINE__
                      << std::endl;
            exit(1);
        }
        size_t ret;
        ret = fscanf(fp, "%d", &width);
        ret = fscanf(fp, "%d", &height);
        ret = fscanf(fp, "%d", &vertexNum);
        routegraph =
            std::make_shared<RouteGraph>(width + 1, height + 1, vertexNum);
        ret = fscanf(fp, "%d", &stringSize);
        deviceStrings.clear();
        deviceStringsMap.clear();
        deviceStrings.resize(stringSize);
        for (int i = 0; i < stringSize; i++) {
            char buf[5000];
            ret = fscanf(fp, "%s", buf);
            deviceStrings[i] = std::string(buf);
            deviceStringsMap[deviceStrings[i]] = i;
        }
        fclose(fp);
        ifPips.clear();
        ifPips.resize(vertexNum);
    }
    std::array<std::function<void(void)>, 10> funcs;
    funcs[0] = [&]() {
        FILE *fp = fopen((prefix + ".vertices").c_str(), "rb");
        if (fp == nullptr) {
            std::cout << " [ERROR] " << __FILE__ << " " << __LINE__
                      << std::endl;
            exit(1);
        }
        routegraph->setVertexNum(vertexNum);
        routegraph->setWidth(width);
        routegraph->setHeight(height);
        auto &vertices = routegraph->getVertices();
        auto ret =
            fread(vertices.data(), sizeof(RouteGraphVertex), vertexNum, fp);
        for (auto &v : vertices) {
            v.pin = std::make_shared<database::Pin>(
                deviceStrings[v.pinNodeName], v.tileRow, v.tileCol, nullptr);
        }
        fclose(fp);
    };

    funcs[1] = [&]() {
        auto &edges = routegraph->getEdges();
        edges.clear();
        FILE *fp = fopen((prefix + ".edges").c_str(), "rb");
        if (fp == nullptr) {
            std::cout << " [ERROR] " << __FILE__ << " " << __LINE__
                      << std::endl;
            exit(1);
        }
        int edgeNum = 0;
        auto ret = fread(&edgeNum, sizeof(edgeNum), 1, fp);
        routegraph->setEdgeNum(edgeNum);
        ret = fread(edges.data(), sizeof(RouteGraphEdge), edgeNum, fp);
        fclose(fp);
    };

    funcs[2] = [&]() {
        FILE *fp = fopen((prefix + ".tw2v").c_str(), "rb");
        if (fp == nullptr) {
            std::cout << " [ERROR] " << __FILE__ << " " << __LINE__
                      << std::endl;
            exit(1);
        }
        ifTileWireToVertex.clear();
        ifTileWireToVertex.resize(deviceStrings.size());
        std::vector<std::pair<std::pair<int, int>, int>> tw2v_vec;
        int tw2v_size = 0;
        auto ret = fread(&tw2v_size, sizeof(int), 1, fp);
        tw2v_vec.resize(tw2v_size);
        ret = fread(tw2v_vec.data(),
                    sizeof(std::pair<std::pair<int, int>, int>), tw2v_size, fp);
        for (auto &it : tw2v_vec) {
            ifTileWireToVertex[it.first.first][it.first.second] = it.second;
        }
        fclose(fp);
    };

    funcs[3] = [&]() {
        FILE *fp = fopen((prefix + ".stp2w").c_str(), "rb");
        if (fp == nullptr) {
            std::cout << " [ERROR] " << __FILE__ << " " << __LINE__
                      << std::endl;
            exit(1);
        }
        std::vector<std::pair<std::pair<int, int>, int>> stp2w_vec;
        int stp2w_size = 0;
        auto ret = fread(&stp2w_size, sizeof(int), 1, fp);
        stp2w_vec.resize(stp2w_size);
        ret =
            fread(stp2w_vec.data(), sizeof(std::pair<std::pair<int, int>, int>),
                  stp2w_size, fp);
        ifSitetypePinToWire.clear();
        for (auto it : stp2w_vec) {
            ifSitetypePinToWire[it.first] = it.second;
        }
        fclose(fp);
    };

    funcs[4] = [&]() {
        FILE *fp = fopen((prefix + ".s2ti").c_str(), "rb");
        if (fp == nullptr) {
            std::cout << " [ERROR] " << __FILE__ << " " << __LINE__
                      << std::endl;
            exit(1);
        }
        std::vector<std::pair<int, int>> s2ti_vec;
        int s2ti_size = 0;
        auto ret = fread(&s2ti_size, sizeof(int), 1, fp);
        s2ti_vec.resize(s2ti_size);
        ret =
            fread(s2ti_vec.data(), sizeof(std::pair<int, int>), s2ti_size, fp);
        ifSiteToTile.clear();
        for (auto it : s2ti_vec) {
            ifSiteToTile[it.first] = it.second;
        }
        fclose(fp);
    };

    funcs[5] = [&]() {
        FILE *fp = fopen((prefix + ".s2ty").c_str(), "rb");
        if (fp == nullptr) {
            std::cout << " [ERROR] " << __FILE__ << " " << __LINE__
                      << std::endl;
            exit(1);
        }
        std::vector<std::pair<int, int>> s2ty_vec;
        int s2ty_size = 0;
        auto ret = fread(&s2ty_size, sizeof(int), 1, fp);
        s2ty_vec.resize(s2ty_size);
        ret =
            fread(s2ty_vec.data(), sizeof(std::pair<int, int>), s2ty_size, fp);
        ifSiteToType.clear();
        for (auto it : s2ty_vec) {
            ifSiteToType[it.first] = it.second;
        }
        fclose(fp);
    };

    funcs[6] = [&]() {
        FILE *fp = fopen((prefix + ".pips_0").c_str(), "rb");
        std::vector<std::pair<std::pair<int, int>, IFPip>> pip_vec;
        int pip_size = 0;
        auto ret = fread(&pip_size, sizeof(int), 1, fp);
        pip_vec.resize(pip_size);
        ret =
            fread(pip_vec.data(), sizeof(std::pair<std::pair<int, int>, IFPip>),
                  pip_size, fp);
        for (auto it : pip_vec) {
            ifPips[it.first.first][it.first.second] = it.second;
        }
        fclose(fp);
    };

    funcs[7] = [&]() {
        FILE *fp = fopen((prefix + ".pips_1").c_str(), "rb");
        std::vector<std::pair<std::pair<int, int>, IFPip>> pip_vec;
        int pip_size = 0;
        auto ret = fread(&pip_size, sizeof(int), 1, fp);
        pip_vec.resize(pip_size);
        ret =
            fread(pip_vec.data(), sizeof(std::pair<std::pair<int, int>, IFPip>),
                  pip_size, fp);
        for (auto it : pip_vec) {
            ifPips[it.first.first][it.first.second] = it.second;
        }
        fclose(fp);
    };

    funcs[8] = [&]() {
        FILE *fp = fopen((prefix + ".pips_2").c_str(), "rb");
        std::vector<std::pair<std::pair<int, int>, IFPip>> pip_vec;
        int pip_size = 0;
        auto ret = fread(&pip_size, sizeof(int), 1, fp);
        pip_vec.resize(pip_size);
        ret =
            fread(pip_vec.data(), sizeof(std::pair<std::pair<int, int>, IFPip>),
                  pip_size, fp);
        for (auto it : pip_vec) {
            ifPips[it.first.first][it.first.second] = it.second;
        }
        fclose(fp);
    };

    funcs[9] = [&]() {
        FILE *fp = fopen((prefix + ".pips_3").c_str(), "rb");
        std::vector<std::pair<std::pair<int, int>, IFPip>> pip_vec;
        int pip_size = 0;
        auto ret = fread(&pip_size, sizeof(int), 1, fp);
        pip_vec.resize(pip_size);
        ret =
            fread(pip_vec.data(), sizeof(std::pair<std::pair<int, int>, IFPip>),
                  pip_size, fp);
        for (auto it : pip_vec) {
            ifPips[it.first.first][it.first.second] = it.second;
        }
        fclose(fp);
    };

    auto thread_func = [&](int threadIdx) {
        for (int func_idx = threadIdx; func_idx < 10; func_idx += numThreads) {
            funcs[func_idx]();
        }
    };

    std::vector<std::thread> thread_pools(numThreads);
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
        thread_pools[threadIdx] = std::thread(thread_func, threadIdx);
    }
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
        thread_pools[threadIdx].join();
    }
}

void FPGAIFParser::loadNetlist(std::string fileName) {
    // std::ifstream ifs(fileName, std::ios::binary | std::ios::ate);
    // std::streamsize size = ifs.tellg();
    // ifs.seekg(0, std::ios::beg);

    // std::vector<char> buffer(size);
    // ifs.read(buffer.data(), size);
    // kj::ArrayPtr<const capnp::word> words(reinterpret_cast<const
    // capnp::word*>(buffer.data()), size / sizeof(capnp::word));
    // // kj::BufferedInputStream is;
    // // capnp::PackedMessageReader reader(is);
    // // auto fs = kj::newDiskFilesystem();
    // // auto file = fs->getRoot().openFile(kj::Path::parse(fileName.c_str()));
    // // auto mmap = file->mmap(0, file->stat().size);
    // // auto wordPtr = reinterpret_cast<const capnp::word*>(mmap.begin());
    // // kj::ArrayPtr<const capnp::word> wordArray(wordPtr, mmap.size() /
    // sizeof(capnp::word)); capnp::FlatArrayMessageReader reader(words);
    auto net_st = std::chrono::high_resolution_clock::now();
    childVertex.resize(routegraph->getVertexNum(), -1);

    gzFile gz = gzopen(fileName.c_str(), "rb");
    if (!gz) {
        std::cerr << "failed to open file" << std::endl;
        exit(-1);
    }
    std::vector<char> decompressedData;
    char buffer[4096];

    int bytesRead;
    while ((bytesRead = gzread(gz, buffer, sizeof(buffer))) > 0) {
        decompressedData.insert(decompressedData.end(), buffer,
                                buffer + bytesRead);
    }

    gzclose(gz);

    if (decompressedData.size() % sizeof(capnp::word) != 0) {
        throw std::runtime_error(
            "Decompressed data size is not a multiple of word size");
    }
    capnp::ReaderOptions option;
    option.traversalLimitInWords = std::numeric_limits<uint64_t>::max();
    option.nestingLimit = 64;

    capnp::FlatArrayMessageReader reader(
        kj::arrayPtr(
            reinterpret_cast<const capnp::word *>(decompressedData.data()),
            decompressedData.size() / sizeof(capnp::word)),
        option);
    auto physNetlist = reader.getRoot<PhysicalNetlist::PhysNetlist>();
    auto builder = message.initRoot<PhysicalNetlist::PhysNetlist>();
    // message.
    // auto builder =
    // message.setRoot<PhysicalNetlist::PhysNetlist::Reader>(physNetlist);
    // physNetlist.
    builder.setPart(physNetlist.getPart());
    builder.setPlacements(physNetlist.getPlacements());
    builder.setPhysNets(physNetlist.getPhysNets());
    builder.setPhysCells(physNetlist.getPhysCells());
    builder.setStrList(physNetlist.getStrList());
    builder.setSiteInsts(physNetlist.getSiteInsts());
    builder.setProperties(physNetlist.getProperties());
    builder.setNullNet(physNetlist.getNullNet());
    std::cout << "phys netlist size: " << builder.getPhysNets().size()
              << std::endl;
    // builder.copyFrom()
    // builder.

    auto physNets = builder.getPhysNets();
    auto strings = builder.getStrList();
    newStrList.reserve(strings.size() + deviceStrings.size());
    for (auto string : strings) {
        newStrList.push_back(std::string(string.cStr()));
    }
    deviceStringToNewString.resize(deviceStrings.size(), -1);
    std::cout << "phys netlist size: " << physNets.size() << std::endl;
    // vertexToSitePin.resize(routegraph->getVertexNum());
    std::vector<int> fatherVertex(routegraph->getVertexNum(), -1);
    for (auto physNet : physNets) {
        std::string netName = std::string(strings[physNet.getName()].cStr());
        std::shared_ptr<Net> net = std::make_shared<Net>(netName);

        auto sources = physNet.getSources();
        auto sinks = physNet.getStubs();
        if (physNet.getType() !=
                PhysicalNetlist::PhysNetlist::NetType::SIGNAL ||
            sources.size() == 0 || sinks.size() == 0) {
            auto sources = physNet.getSources();
            std::queue<PhysicalNetlist::PhysNetlist::RouteBranch::Builder> q;
            // std::cout << physNet.getStubNodes().size() << ' ' <<
            // physNet.getStubs().size() << ' ' << physNet.getSources().size()
            // << std::endl;
            for (auto source : sources)
                q.push(source);
            while (!q.empty()) {
                auto now = q.front();
                q.pop();
                if (now.getRouteSegment().which() ==
                    PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::
                        Which::SITE_PIN) {
                    // std::cout <<
                    // strings[now.getRouteSegment().getSitePin().getSite()].cStr()
                    // << ' ' <<
                    // strings[now.getRouteSegment().getSitePin().getPin()].cStr()
                    // << std::endl;
                    std::string siteName =
                        strings[now.getRouteSegment().getSitePin().getSite()]
                            .cStr();
                    std::string pinName =
                        strings[now.getRouteSegment().getSitePin().getPin()]
                            .cStr();
                    int vertexIdx = getVertexFromSitePin(siteName, pinName);
                    if (routegraph->getVertexCap(vertexIdx) > 0)
                        routegraph->addVertexCap(vertexIdx, -1);
                }
                if (now.getRouteSegment().which() ==
                    PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::
                        Which::PIP) {
                    // std::cout <<
                    // strings[now.getRouteSegment().getPip().getTile()].cStr()
                    // << ' ' <<
                    // strings[now.getRouteSegment().getPip().getWire0()].cStr()
                    // << ' ' <<
                    // strings[now.getRouteSegment().getPip().getWire1()].cStr()
                    // << std::endl;
                    std::string tileName =
                        strings[now.getRouteSegment().getPip().getTile()]
                            .cStr();
                    std::string wire0Name =
                        strings[now.getRouteSegment().getPip().getWire0()]
                            .cStr();
                    std::string wire1Name =
                        strings[now.getRouteSegment().getPip().getWire1()]
                            .cStr();
                    auto tileNameIdx = deviceStringsMap[tileName],
                         wire0NameIdx = deviceStringsMap[wire0Name],
                         wire1NameIdx = deviceStringsMap[wire1Name];
                    int vertexIdx0 =
                        ifTileWireToVertex[tileNameIdx][wire0NameIdx];
                    int vertexIdx1 =
                        ifTileWireToVertex[tileNameIdx][wire1NameIdx];
                    if (routegraph->getVertexCap(vertexIdx0) > 0)
                        routegraph->addVertexCap(vertexIdx0, -1);
                    if (routegraph->getVertexCap(vertexIdx1) > 0)
                        routegraph->addVertexCap(vertexIdx1, -1);
                }
                if (!now.hasBranches())
                    continue;
                auto branchs = now.getBranches();
                for (auto branch : branchs) {
                    q.push(branch);
                }
            }
            auto stubs = physNet.getStubs();
            for (auto stub : stubs)
                q.push(stub);
            while (!q.empty()) {
                auto now = q.front();
                q.pop();
                // std::cout << now.getRouteSegment().which() << std::endl;
                if (now.getRouteSegment().which() ==
                    PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::
                        Which::SITE_PIN) {
                    // std::cout <<
                    // strings[now.getRouteSegment().getSitePin().getSite()].cStr()
                    // << ' ' <<
                    // strings[now.getRouteSegment().getSitePin().getPin()].cStr()
                    // << std::endl;
                    std::string siteName =
                        strings[now.getRouteSegment().getSitePin().getSite()]
                            .cStr();
                    std::string pinName =
                        strings[now.getRouteSegment().getSitePin().getPin()]
                            .cStr();
                    int vertexIdx = getVertexFromSitePin(siteName, pinName);
                    if (routegraph->getVertexCap(vertexIdx) > 0)
                        routegraph->addVertexCap(vertexIdx, -1);
                }
                if (now.getRouteSegment().which() ==
                    PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::
                        Which::PIP) {
                    // std::cout <<
                    // strings[now.getRouteSegment().getPip().getTile()].cStr()
                    // << ' ' <<
                    // strings[now.getRouteSegment().getPip().getWire0()].cStr()
                    // << ' ' <<
                    // strings[now.getRouteSegment().getPip().getWire1()].cStr()
                    // << std::endl;
                    std::string tileName =
                        strings[now.getRouteSegment().getPip().getTile()]
                            .cStr();
                    std::string wire0Name =
                        strings[now.getRouteSegment().getPip().getWire0()]
                            .cStr();
                    std::string wire1Name =
                        strings[now.getRouteSegment().getPip().getWire1()]
                            .cStr();
                    auto tileNameIdx = deviceStringsMap[tileName],
                         wire0NameIdx = deviceStringsMap[wire0Name],
                         wire1NameIdx = deviceStringsMap[wire1Name];
                    int vertexIdx0 =
                        ifTileWireToVertex[tileNameIdx][wire0NameIdx];
                    int vertexIdx1 =
                        ifTileWireToVertex[tileNameIdx][wire1NameIdx];
                    if (routegraph->getVertexCap(vertexIdx0) > 0)
                        routegraph->addVertexCap(vertexIdx0, -1);
                    if (routegraph->getVertexCap(vertexIdx1) > 0)
                        routegraph->addVertexCap(vertexIdx1, -1);
                }
                if (!now.hasBranches())
                    continue;
                auto branchs = now.getBranches();
                for (auto branch : branchs) {
                    q.push(branch);
                }
            }
            continue;
        }

        // if (netName.find("control") != std::string::npos) continue;
        // if (netName.find("BUFGP") != std::string::npos) continue;
        // if (sources.size() > 1) {
        //     std::cout << "[Warning] sources num more than 1" << std::endl;
        // }
        // if (sources.size() == 0) {
        //     std::cout << "[Warning] source size is zero" << std::endl;
        //     continue;
        // }
        std::queue<PhysicalNetlist::PhysNetlist::RouteBranch::Builder> q;
        std::vector<int> initSources;
        for (auto source : sources)
            q.push(source);
        int sourceNum = 0;
        std::vector<int> sourceVs;
        while (!q.empty()) {
            auto now = q.front();
            q.pop();
            auto segment = now.getRouteSegment();
            if (segment.which() == PhysicalNetlist::PhysNetlist::RouteBranch::
                                       RouteSegment::Which::SITE_PIN) {
                auto sitePin = segment.getSitePin();
                auto siteName = std::string(strings[sitePin.getSite()].cStr());
                auto pinName = std::string(strings[sitePin.getPin()].cStr());
                int vertexIdx = getVertexFromSitePin(siteName, pinName);
                assert(vertexIdx > 0);
                if (sinks.size() != 0) {
                    // net->setSource(vertexIdx);
                    // std::cout << net->getName() << " : " << siteName << '_'
                    // << pinName << "->";
                    auto branch = now;
                    int wd = 1000;
                    std::queue<int> q;
                    q.push(vertexIdx);
                    initSources.push_back(vertexIdx);
                    int intVertex = -1;
                    while (!q.empty() && wd) {
                        int now = q.front();
                        wd--;
                        q.pop();
                        for (int i = routegraph->getHeadOutEdgeIdx(now);
                             i != -1; i = routegraph->getEdge(i).preFromEdge) {
                            int frontVertex = routegraph->getEdge(i).to;
                            if (routegraph->getVertexType(frontVertex) ==
                                INTTILE) {
                                intVertex = now;
                                break;
                            }
                            if (fatherVertex[frontVertex] != -1)
                                continue;
                            fatherVertex[frontVertex] = now;
                            q.push(frontVertex);
                        }
                    }

                    if (intVertex != -1) {
                        std::stack<int> st;
                        int now = intVertex;
                        while (now != vertexIdx) {
                            st.push(now);
                            now = fatherVertex[now];
                        }
                        while (!st.empty()) {
                            int nexVertex = st.top();
                            st.pop();
                            // if (ifPips.find(fatherVertex[nexVertex]) ==
                            //     ifPips.end())
                            //     std::cout << "[error] vertex "
                            //               << fatherVertex[nexVertex]
                            //               << " not found!" << std::endl;
                            // if (ifPips[fatherVertex[nexVertex]].find(
                            //         nexVertex) ==
                            //     ifPips[fatherVertex[nexVertex]].end())
                            //     std::cout << "[error] vertex " << nexVertex
                            //               << " not found!" << std::endl;
                            auto &ifPip =
                                ifPips[fatherVertex[nexVertex]][nexVertex];
                            branch.initBranches(1);
                            branch = branch.getBranches()[0];
                            branch.initRouteSegment();
                            auto pip = branch.getRouteSegment().initPip();
                            if (deviceStringToNewString[ifPip.tileNameIdx] == -1) {
                                deviceStringToNewString[ifPip.tileNameIdx] = newStrList.size();
                                newStrList.push_back(std::string(deviceStrings[ifPip.tileNameIdx]));
                            }
                            pip.setTile(deviceStringToNewString[ifPip.tileNameIdx]);
                            if (deviceStringToNewString[ifPip.wire0NameIdx] == -1) {
                                deviceStringToNewString[ifPip.wire0NameIdx] = newStrList.size();
                                newStrList.push_back(std::string(deviceStrings[ifPip.wire0NameIdx]));
                            }
                            pip.setWire0(deviceStringToNewString[ifPip.wire0NameIdx]);
                            if (deviceStringToNewString[ifPip.wire1NameIdx] == -1) {
                                deviceStringToNewString[ifPip.wire1NameIdx] = newStrList.size();
                                newStrList.push_back(std::string(deviceStrings[ifPip.wire1NameIdx]));
                            }
                            pip.setWire1(deviceStringToNewString[ifPip.wire1NameIdx]);
                            pip.setForward(ifPip.forward);
                        }
                        vertexIdx = intVertex;
                        net->setIndirect();
                    }
                    sourceVs.push_back(vertexIdx);
                    routegraph->setVertexType(vertexIdx, VertexType::SOURCE);
                    // if (net->getName() ==
                    // "grp_processImage_fu_310/grp_int_sqrt_fu_10448/icmp_ln3409_12_fu_1348_p2_carry_n_3")
                    // std::cout <<
                    // routegraph->getVertexByIdx(vertexIdx)->getName() << ' '
                    // << routegraph->getPos(vertexIdx).X() << ' ' <<
                    // routegraph->getPos(vertexIdx).Y() << std::endl;
                    net->addGuideNode(routegraph->getPos(vertexIdx).X() - 5,
                                      routegraph->getPos(vertexIdx).Y() - 5);
                    net->addGuideNode(routegraph->getPos(vertexIdx).X() + 5,
                                      routegraph->getPos(vertexIdx).Y() + 5);
                    // if (netName == "net_78668") {
                    //     // std::cout << sitePin.
                    //     std::cout << siteName << ' ' << pinName << ' ' <<
                    //     sitePin.getPin() << std::endl; std::cout <<
                    //     ifSiteToTile[siteName] << ' ' <<
                    //     ifSiteToType[siteName] << ' ' <<
                    //     ifSitetypePinToWire[ifSiteToType[siteName]][pinName]
                    //     << std::endl; std::cout <<
                    //     routegraph->getPos(vertexIdx).X() << ' ' <<
                    //     routegraph->getPos(vertexIdx).Y() << std::endl;
                    // }
                    sourceNum++;
                } else {
                    if (routegraph->getVertexCap(vertexIdx) > 0)
                        routegraph->addVertexCap(vertexIdx, -1);
                    // std::cout <<
                    // routegraph->getVertexByIdx(vertexIdx)->getName() << ' '
                    // << netName << std::endl;
                }
            }
            if (!now.hasBranches())
                continue;
            auto branchs = now.getBranches();
            for (auto branch : branchs) {
                q.push(branch);
            }
        }
        // auto source = sources[0];
        // auto routeSegment = source.getRouteSegment();
        // if (routeSegment.which() !=
        // PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::Which::SITE_PIN)
        // {
        //     std::cout << "[warning] source is not SITE_PIN but " <<
        //     routeSegment.which() << std::endl; getchar(); continue;
        // }
        for (auto sink : sinks) {

            while (!q.empty())
                q.pop();
            q.push(sink);
            while (!q.empty()) {
                auto now = q.front();
                q.pop();
                auto segment = now.getRouteSegment();
                if (segment.which() ==
                    PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::
                        Which::SITE_PIN) {
                    auto sitePin = segment.getSitePin();
                    auto siteName =
                        std::string(strings[sitePin.getSite()].cStr());
                    auto pinName =
                        std::string(strings[sitePin.getPin()].cStr());
                    int vertexIdx = getVertexFromSitePin(siteName, pinName);

                    if (net->isIndirect()) {
                        int wd = 1000;
                        std::queue<int> q;
                        q.push(vertexIdx);
                        int intVertex = -1;
                        while (!q.empty() && wd) {
                            int now = q.front();
                            q.pop();
                            // if (net->getName() == "ibufds/O") std::cout <<
                            // routegraph->getVertexByIdx(now)->getName() <<
                            // std::endl;
                            wd--;
                            if (routegraph->getVertexType(now) == NETSINK ||
                                routegraph->getVertexType(now) == INTTILE) {
                                intVertex = now;
                                break;
                            }
                            for (int i = routegraph->getHeadInEdgeIdx(now);
                                 i != -1;
                                 i = routegraph->getEdge(i).preToEdge) {
                                auto &edge = routegraph->getEdge(i);
                                if (childVertex[edge.from] != -1)
                                    continue;
                                // if (net->getName() == "net_524074")
                                //     std::cout <<
                                //     routegraph->getVertexByIdx(edge.to)->getName()
                                //     << "<-" <<
                                //     routegraph->getVertexByIdx(edge.from)->getName()
                                //     << std::endl;
                                childVertex[edge.from] = now;
                                q.push(edge.from);
                            }
                        }
                        // std::cout << net->getName() << " : " << siteName <<
                        // '_'
                        // << pinName << "->"; if (intVertex != -1)
                        //     std::cout <<
                        //     routegraph->getVertexByIdx(intVertex)->getName()
                        //     << '
                        //     '
                        //     <<routegraph->getVertexByIdx(intVertex)->getPos().X()
                        //     << ' ' <<
                        //     routegraph->getVertexByIdx(intVertex)->getPos().Y()
                        //     << std::endl;
                        // else
                        //     std::cout <<
                        //     routegraph->getVertexByIdx(vertexIdx)->getName()
                        //     << std::endl;
                        if (intVertex != -1)
                            vertexIdx = intVertex;
                        else
                            net->setIndirect(false);
                    }
                    // if (net->getName() ==
                    // "grp_face_detect_Pipeline_imageScalerL1_imageScalerL1_1_fu_299/mul_9ns_27s_32_1_1_U5/ap_CS_fsm_reg[40][4]")
                    // std::cout <<
                    // routegraph->getVertexByIdx(vertexIdx)->getName() << ' '
                    // << routegraph->getPos(vertexIdx).X() << ' ' <<
                    // routegraph->getPos(vertexIdx).Y() << std::endl;
                    assert(vertexIdx != 0);
                    routegraph->setVertexType(vertexIdx,
                                              router::VertexType::NETSINK);
                    net->addGuideNode(routegraph->getPos(vertexIdx).X() - 5,
                                      routegraph->getPos(vertexIdx).Y() - 5);
                    net->addGuideNode(routegraph->getPos(vertexIdx).X() + 5,
                                      routegraph->getPos(vertexIdx).Y() + 5);
                    net->addSink(vertexIdx);
                    // if (netName == "net_78668") {
                    //     // std::cout << sitePin.
                    //     std::cout << siteName << ' ' << pinName << ' ' <<
                    //     sitePin.getPin() << std::endl; std::cout <<
                    //     ifSiteToTile[siteName] << ' ' <<
                    //     ifSitetypePinToWire[ifSiteToType[siteName]][pinName]
                    //     << std::endl; std::cout <<
                    //     routegraph->getPos(vertexIdx).X() << ' ' <<
                    //     routegraph->getPos(vertexIdx).Y() << std::endl;
                    //     std::cout << vertexIdx << ' ' <<
                    //     routegraph->getVertexByIdx(vertexIdx)->getName() <<
                    //     std::endl;
                    // }
                    // vertexToSitePin[vertexIdx] =
                    // std::make_pair(sitePin.getSite(), sitePin.getPin());
                    break;
                }
                if (!now.hasBranches())
                    continue;
                auto branchs = now.getBranches();
                for (auto branch : branchs) {
                    q.push(branch);
                }
            }
        }
        if (!net->isIndirect())
            for (int i = 0; i < sourceVs.size(); i++)
                sourceVs[i] = initSources[i];
        if (sourceNum == 0)
            continue;
        if (sourceNum == 1) {
            net->setSource(sourceVs[0]);
        }
        if (sourceNum > 1) {
            // std::cout << "[Warning] net " << netName << " have " << sourceNum
            // << "sources" << std::endl;
            std::shared_ptr<database::Pin> pin =
                std::make_shared<database::Pin>(
                    netName + "_virtual_source",
                    routegraph->getPos(sourceVs[0]).X(),
                    routegraph->getPos(sourceVs[0]).Y(), nullptr);
            int vSource = routegraph->addVertex(
                routegraph->getPos(sourceVs[0]).X(),
                routegraph->getPos(sourceVs[0]).Y(),
                routegraph->getPos(sourceVs[0]).X(),
                routegraph->getPos(sourceVs[0]).Y(), pin, 1);
            for (auto source : sourceVs) {
                routegraph->addEdge(vSource, source, baseCost);
            }
            net->setSource(vSource);
            routegraph->setVertexType(vSource, VertexType::SOURCE);
        }
        if (sinks.size() > 0) {
            // if (net->getName() ==
            // "grp_processImage_fu_310/grp_int_sqrt_fu_10448/icmp_ln3409_12_fu_1348_p2_carry_n_3")
            //     std::cout <<
            //     routegraph->getVertexByIdx(net->getSource())->getName() << '
            //     ' << routegraph->getVertexByIdx(initSource)->getName() << ' '
            //     << net->isIndirect() << std::endl;
            // if (!net->isIndirect()) {
            //     net->setSource(initSource);
            //     // net->addGuideNode(0, 0);
            //     // net->addGuideNode(routegraph->getWidth(),
            //     routegraph->getHeight());
            // }
            // net->useGlobalResult(false);
            netlist.push_back(net);
        }
    }
    std::cout << "netlist size: " << netlist.size() << std::endl;
    auto net_ed = std::chrono::high_resolution_clock::now();
    std::cout << "load net : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(net_ed -
                                                                       net_st)
                     .count()
              << " ms" << std::endl;
    // exit(0);
}

void FPGAIFParser::run() {
    Router router(routegraph);
    int totalNum = 100;
    int i = 0;
    for (auto net : netlist) {
        router.addNet(net);

        // if (net->getSinkSize() < 2 && i < totalNum) {
        //     router.addNet(net);
        //     i++;
        // }

        // if (net->getName() == "grp_processImage_fu_310/p_II_1140_fu_4260[7]")
        // {
        //     router.addNet(net);
        // }
    }
    router.runTaskflow();
    // printRouteResult(router.getNetlist(), xmlResFileName, routegraph);
}

void FPGAIFParser::printIFResult(std::string const &fileName) {
    std::cout << "Printing Results into Interchange Format...\n";
    auto physNetlist = message.getRoot<PhysicalNetlist::PhysNetlist>();
    auto physNets = physNetlist.getPhysNets();
    auto strings = physNetlist.getStrList();
    auto &routetree = router::Pathfinder::routetree;
    auto &netRoot = routetree.getNetRoot();
    std::unordered_map<std::string, std::shared_ptr<router::Net>> nets;
    std::unordered_map<int, PhysicalNetlist::PhysNetlist::RouteBranch::Reader>
        vertexToSitePin;
    for (auto &it : netRoot) {
        nets[it.first->getName()] = it.first;
    }
    // std::cout << newStrList.size() << std::endl;
    std::cout << "Modificationing PhysNets\n";
    for (auto physNet : physNets) {
        std::string netName = std::string(strings[physNet.getName()].cStr());
        // if (netName.find("DSP") != std::string::npos) continue;
        if (nets.find(netName) == nets.end())
            continue;
        auto net = nets[netName];
        if (net->getRouteStatus() != SUCCESS)
            continue;
        int checkSink = 0;
        auto sinks = physNet.disownStubs();
        for (auto sink : sinks.get()) {
            assert(sink.getRouteSegment().which() ==
                   PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::
                       Which::SITE_PIN);
            auto sitePin = sink.getRouteSegment().getSitePin();
            auto siteName = std::string(strings[sitePin.getSite()].cStr());
            auto pinName = std::string(strings[sitePin.getPin()].cStr());
            int vertexIdx = getVertexFromSitePin(siteName, pinName);
            vertexToSitePin[vertexIdx] = sink;
        }
        physNet.initStubs(0);

        std::queue<PhysicalNetlist::PhysNetlist::RouteBranch::Builder>
            branchQueue;
        for (auto source : physNet.getSources()) {
            branchQueue.push(source);
        }
        while (!branchQueue.empty()) {
            auto nowBranch = branchQueue.front();
            branchQueue.pop();
            // if (net->getName() ==
            // "processing_element_inst_43/dot_product_16_8_30_8_inst_2/dsp_block_16_8_false_inst_0/resulta_tmp0/P[0]")
            //     std::cout << nowBranch.getRouteSegment().which() <<
            //     std::endl;
            // if (nowBranch.getRouteSegment().which() ==
            // PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::Which::PIP)
            // {
            //     if (net->getName() ==
            //     "processing_element_inst_43/dot_product_16_8_30_8_inst_2/dsp_block_16_8_false_inst_0/resulta_tmp0/P[0]")
            //     std::cout <<
            //     newStrList[nowBranch.getRouteSegment().getPip().getTile()] <<
            //     ' ' <<
            //     newStrList[nowBranch.getRouteSegment().getPip().getWire1()]
            //     << ' ' << nowBranch.getBranches().size() << std::endl;
            // }
            if (nowBranch.getRouteSegment().which() ==
                PhysicalNetlist::PhysNetlist::RouteBranch::RouteSegment::Which::
                    SITE_PIN) {
                auto sitePin = nowBranch.getRouteSegment().getSitePin();
                auto siteName = std::string(strings[sitePin.getSite()].cStr());
                auto pinName = std::string(strings[sitePin.getPin()].cStr());
                int vertexIdx = getVertexFromSitePin(siteName, pinName);
                assert(vertexIdx > 0);
                auto treeNode = routetree.getTreeNodeByIdx(vertexIdx);
                // if (net->getName() ==
                // "core_inst/core_inst/core_pcie_inst/core_inst/iface[1].interface_inst/desc_fetch_inst/dma_psdpram_inst/dma_ram_rd_resp_data_int[366]")
                //     std::cout << vertexIdx << ' ' << treeNode << std::endl;
                if (treeNode != nullptr && treeNode->net == net) {
                    std::queue<std::pair<
                        std::shared_ptr<router::TreeNode>,
                        PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>
                        q;
                    q.push(std::make_pair(treeNode, nowBranch));
                    while (!q.empty()) {
                        auto nowTreeNode = q.front().first;
                        auto prevBranch = q.front().second;
                        q.pop();
                        int nowVertex = nowTreeNode->nodeId;
                        // if (net->getName() == "net_524074")
                        //     std::cout <<
                        //     routegraph->getVertexByIdx(childVertex[nowVertex])->getName()
                        //     << ' ' <<
                        //     routegraph->getVertexByIdx(nowVertex)->getName()
                        //     << std::endl;
                        // if (net->getName() == "net_524074") {
                        //     std::cout << ' ' <<  nowVertex << ' ' <<
                        //     childVertex[nowVertex] << ' ' <<
                        //     nowTreeNode->firstChild << std::endl;
                        // }
                        if (childVertex[nowVertex] != -1 &&
                            nowTreeNode->firstChild == nullptr) {
                            routetree.addNode(nowTreeNode,
                                              childVertex[nowVertex], net);
                        }
                        int totalNodes = 0;
                        for (auto nex = nowTreeNode->firstChild; nex != nullptr;
                             nex = nex->right)
                            totalNodes++;
                        if (totalNodes == 0) {
                            auto newBranch = prevBranch.initBranches(1);
                            newBranch.setWithCaveats(
                                0, vertexToSitePin[nowTreeNode->nodeId]);
                            checkSink++;
                            // auto sitePin =
                            // newBranch[0].getRouteSegment().initSitePin();
                            // sitePin.setSite(vertexToSitePin[nowTreeNode->nodeId].first);
                            // sitePin.setPin(vertexToSitePin[nowTreeNode->nodeId].second);
                        } else {
                            auto newBranch =
                                prevBranch.initBranches(totalNodes);
                            int id = 0;
                            for (auto nex = nowTreeNode->firstChild;
                                 nex != nullptr; nex = nex->right) {
                                newBranch[id].initRouteSegment();
                                auto pip =
                                    newBranch[id].getRouteSegment().initPip();
                                auto &ifPip =
                                    ifPips[nowTreeNode->nodeId][nex->nodeId];
                                if (deviceStringToNewString[ifPip.tileNameIdx] == -1) {
                                    deviceStringToNewString[ifPip.tileNameIdx] = newStrList.size();
                                    newStrList.push_back(std::string(deviceStrings[ifPip.tileNameIdx]));
                                }
                                pip.setTile(deviceStringToNewString[ifPip.tileNameIdx]);
                                if (deviceStringToNewString[ifPip.wire0NameIdx] == -1) {
                                    deviceStringToNewString[ifPip.wire0NameIdx] = newStrList.size();
                                    newStrList.push_back(std::string(deviceStrings[ifPip.wire0NameIdx]));
                                }
                                pip.setWire0(deviceStringToNewString[ifPip.wire0NameIdx]);
                                if (deviceStringToNewString[ifPip.wire1NameIdx] == -1) {
                                    deviceStringToNewString[ifPip.wire1NameIdx] = newStrList.size();
                                    newStrList.push_back(std::string(deviceStrings[ifPip.wire1NameIdx]));
                                }
                                pip.setWire1(deviceStringToNewString[ifPip.wire1NameIdx]);
                                pip.setForward(ifPip.forward);
                                q.push(std::make_pair(nex, newBranch[id]));
                                id++;
                            }
                        }
                    }
                    continue;
                }
            } else if (nowBranch.getRouteSegment().which() ==
                           PhysicalNetlist::PhysNetlist::RouteBranch::
                               RouteSegment::PIP &&
                       nowBranch.getBranches().size() == 0) {
                auto segment = nowBranch.getRouteSegment();
                std::string tileName = newStrList[segment.getPip().getTile()];
                std::string wireName = newStrList[segment.getPip().getWire1()];
                auto tileNameIdx = deviceStringsMap[tileName],
                     wireNameIdx = deviceStringsMap[wireName];
                // if (net->getName() ==
                // "core_inst/core_inst/core_pcie_inst/core_inst/iface[1].interface_inst/desc_fetch_inst/dma_psdpram_inst/dma_ram_rd_resp_data_int[366]")
                //     std::cout << tileName << ' ' << wireName << std::endl;
                int vertexIdx = ifTileWireToVertex[tileNameIdx][wireNameIdx];
                assert(vertexIdx > 0);
                auto treeNode = routetree.getTreeNodeByIdx(vertexIdx);
                if (treeNode != nullptr && treeNode->net == net) {
                    std::queue<std::pair<
                        std::shared_ptr<router::TreeNode>,
                        PhysicalNetlist::PhysNetlist::RouteBranch::Builder>>
                        q;
                    q.push(std::make_pair(treeNode, nowBranch));
                    while (!q.empty()) {
                        auto nowTreeNode = q.front().first;
                        auto prevBranch = q.front().second;
                        q.pop();
                        int nowVertex = nowTreeNode->nodeId;
                        // if (net->getName() == "net_524074")
                        //     std::cout <<
                        //     routegraph->getVertexByIdx(childVertex[nowVertex])->getName()
                        //     << ' ' <<
                        //     routegraph->getVertexByIdx(nowVertex)->getName()
                        //     << std::endl;
                        // if (net->getName() == "net_524074") {
                        //     std::cout << ' ' <<  nowVertex << ' ' <<
                        //     childVertex[nowVertex] << ' ' <<
                        //     nowTreeNode->firstChild << std::endl;
                        // }
                        if (childVertex[nowVertex] != -1 &&
                            nowTreeNode->firstChild == nullptr) {
                            routetree.addNode(nowTreeNode,
                                              childVertex[nowVertex], net);
                        }
                        int totalNodes = 0;
                        for (auto nex = nowTreeNode->firstChild; nex != nullptr;
                             nex = nex->right)
                            totalNodes++;
                        if (totalNodes == 0) {
                            auto newBranch = prevBranch.initBranches(1);
                            newBranch.setWithCaveats(
                                0, vertexToSitePin[nowTreeNode->nodeId]);
                            checkSink++;
                            // auto sitePin =
                            // newBranch[0].getRouteSegment().initSitePin();
                            // sitePin.setSite(vertexToSitePin[nowTreeNode->nodeId].first);
                            // sitePin.setPin(vertexToSitePin[nowTreeNode->nodeId].second);
                        } else {
                            auto newBranch =
                                prevBranch.initBranches(totalNodes);
                            int id = 0;
                            for (auto nex = nowTreeNode->firstChild;
                                 nex != nullptr; nex = nex->right) {
                                newBranch[id].initRouteSegment();
                                auto pip =
                                    newBranch[id].getRouteSegment().initPip();
                                if (ifPips[nowTreeNode->nodeId].find(
                                        nex->nodeId) ==
                                    ifPips[nowTreeNode->nodeId].end())
                                    std::cout << "[error] vertex "
                                              << nex->nodeId << " not found!"
                                              << std::endl;
                                auto &ifPip =
                                    ifPips[nowTreeNode->nodeId][nex->nodeId];
                                if (deviceStringToNewString[ifPip.tileNameIdx] == -1) {
                                    deviceStringToNewString[ifPip.tileNameIdx] = newStrList.size();
                                    newStrList.push_back(std::string(deviceStrings[ifPip.tileNameIdx]));
                                }
                                pip.setTile(deviceStringToNewString[ifPip.tileNameIdx]);
                                if (deviceStringToNewString[ifPip.wire0NameIdx] == -1) {
                                    deviceStringToNewString[ifPip.wire0NameIdx] = newStrList.size();
                                    newStrList.push_back(std::string(deviceStrings[ifPip.wire0NameIdx]));
                                }
                                pip.setWire0(deviceStringToNewString[ifPip.wire0NameIdx]);
                                if (deviceStringToNewString[ifPip.wire1NameIdx] == -1) {
                                    deviceStringToNewString[ifPip.wire1NameIdx] = newStrList.size();
                                    newStrList.push_back(std::string(deviceStrings[ifPip.wire1NameIdx]));
                                }
                                pip.setWire1(deviceStringToNewString[ifPip.wire1NameIdx]);
                                pip.setForward(ifPip.forward);
                                q.push(std::make_pair(nex, newBranch[id]));
                                id++;
                            }
                        }
                    }
                    continue;
                }
            }
            auto branchs = nowBranch.getBranches();
            for (auto branch : branchs) {
                branchQueue.push(branch);
            }
        }
        if (checkSink != net->getSinkSize()) {
            std::cout << "[Error] Net " << net->getName() << " has stub sinks"
                      << std::endl;
            std::cout << checkSink << ' ' << net->getSinkSize() << std::endl;
        }
    }
    std::cout << "Modificationing StrList\n";
    int oldSize = strings.size();
    auto newStrSize = newStrList.size();
    auto newStrings = physNetlist.initStrList(newStrSize);
    std::cout << newStrSize << ' ' << oldSize << std::endl;
    for (int i = 0; i < newStrSize; i++) {
        newStrings.set(i, newStrList[i].c_str());
        if (newStrList[i] == "")
            std::cout << "[error] find empty string id:" << i << std::endl;
        // std::cout << newStrings[i].cStr() << std::endl;
        // getchar();
    }
    // // std::cout << newStrList[301589] << ' ' << newStrList[301591] <<
    // std::endl; std::cout << newStrList[301589].c_str() << ' ' <<
    // newStrList[301591].c_str() << std::endl; std::cout <<
    // physNetlist.getStrList()[301589].cStr() << ' ' <<
    // physNetlist.getStrList()[301591].cStr() << std::endl;

    std::cout << "Outputing Files\n";
    kj::VectorOutputStream output;
    capnp::writeMessage(output, message);
    auto array = output.getArray();
    std::vector<char> unzipData(array.begin(), array.end());
    // uLongf compressedSize = compressBound(array.size() *
    // sizeof(capnp::word)); std::vector<Bytef> compressedData(compressedSize);
    std::cout << "output size before gzip: " << unzipData.size() << std::endl;
    // int f = open("ungzip.phys", O_WRONLY | O_CREAT, 0666);
    // capnp::writeMessageToFd(f, message);
    // close(f);

    // gzFile file = gzopen(fileName.c_str(), "wb");
    // if (!file) {
    //     std::cout << "file open failed!" << std::endl;
    //     exit(-1);
    // }

    // int result = gzwrite(file, array.begin(), array.size());
    std::vector<char> gzipData;
    gzipCompress(unzipData, gzipData);
    std::cout << "output size after gzip: " << gzipData.size() << std::endl;
    std::ofstream outputFile(fileName, std::ios::binary);
    outputFile.write(gzipData.data(), gzipData.size());

    // gzclose(file);
    // int result = compress2(compressedData.data(), &compressedSize,
    // reinterpret_cast<const Bytef*>(array.begin()), array.size(), 6); if
    // (result != Z_OK) {
    //     std::cerr << "[FATAL ERROR] GZIP FAILED!" << std::endl;
    //     exit(-1);
    // }
    // std::ofstream ofs(fileName, std::ios::binary);
    // ofs.write(reinterpret_cast<const char*>(compressedData.data()),
    // compressedSize); ofs.close();
}