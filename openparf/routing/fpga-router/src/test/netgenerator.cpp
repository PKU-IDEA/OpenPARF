#include "netgenerator.h"

#include <cstdlib>
#include <ctime>

std::vector<std::shared_ptr<router::Net>> generateNetlistRandomly(std::shared_ptr<router::RouteGraph> graph) {
    // srand(time(0));    
    std::vector<std::shared_ptr<router::Net>> netlist;
    int n = 1000;
    for (int i = 0; i < n; i++) {
        netlist.push_back(generateNetRandomly(graph));
    }
    return std::move(netlist);
}

std::shared_ptr<router::Net> generateNetRandomly(std::shared_ptr<router::RouteGraph> graph) {
    std::shared_ptr<router::Net> net(new router::Net());
    int vertexNum = graph->getVertexNum();
    int source = rand() % vertexNum;
    while(graph->getVertexDegree(source) == 0) 
        source = rand() % vertexNum;
    int sinkNum = rand() % 100 + 2;
    net->setSource(source);
    for (int i = 0; i < sinkNum; i++) {
        int sink = source;
        while (true) {
            if (rand() % 1939 == 0) break;
            if (graph->getVertexDegree(sink) == 0) break;
            int id = rand() % graph->getVertexDegree(sink);
            if (graph->getPos( graph->getEdge(sink, id)).X() == 0 && graph->getPos( graph->getEdge(sink, id)).Y() == 0) {
                std::cout << "SINK : " << sink << " Name : " << graph->getVertexByIdx(sink)->getName() << std::endl;
            } 
            sink = graph->getEdge(sink, id);
        }
        // if (graph->getPos(sink).X() == 0 && graph->getPos(sink).Y() == 0) 
        //     std::cout << " SINK : " << sink << " Name : " << graph->getVertexByIdx(sink)->getName() << std::endl;
        if (!net->isSink(sink) && sink != source)
            net->addSink(sink);
    }
    return net;
}