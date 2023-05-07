#include "tilerouter.h"
#include "router.h"
#include <chrono>
#include <future>
#include <queue>
#include "pathfinder.h"

namespace router
{
    void TileRouter::tileRoute()
    {
        auto &netlist = router->getNetlist();
        for (auto net : netlist)
        {
            int source = net->getSource();
            int sourceX = graph->getPos(source).X(), sourceY = graph->getPos(source).Y();
            gridNetlist[sourceX][sourceY].push_back(net);
        }
        std::string name = "FUAT";
        buildLocalGraph(name);
        std::cout << localgraph->getVertexNum() << ' ' << localgraph->getEdgeNum() << std::endl;
        GRBEnv env = GRBEnv();
        env.set(GRB_IntParam_OutputFlag, 0);
        env.set(GRB_IntParam_Threads, 1);
        //  env.set(GRB_IntParam_MIPFocus, 1);
        //  env.set(GRB_DoubleParam_Heuristics, 0.01); // Percentage time spent in feadsibility heuristics
        // env.set(GRB_IntParam_Cuts, 3); // Very Agressive cut generation
        // env.set(GRB_IntParam_Presolve, 2); // Agressive presolve
        // env.set(GRB_IntParam_VarBranch, 2); // Maximum Infeasibility Branching
        int width = layout->getwidth(), height = layout->getHeight();
        int totalTiles = 0, maxNets = 0;
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++)
            {
                // std::cout << i << ' ' << j << std::endl;
                if (layout->getContent(i, j).gridModule != nullptr && layout->getContent(i, j).gridModule->getName() == "FUAT" && gridNetlist[i][j].size())
                {
                    // std::cout << "tile Routing " << i << ' ' << j << std::endl;
                    totalTiles++;
                    maxNets = std::max(maxNets, (int)gridNetlist[i][j].size());
                    // getchar();
                }
            }
        std::cout << "total tiles: " << totalTiles << " maxNets: " << maxNets << std::endl;
        using namespace std::chrono;
        high_resolution_clock::time_point s, e;
        s = high_resolution_clock::now();
        int totalTile = width * height;
// #pragma omp parallel for num_threads(40)
        for (int id = 0; id < totalTile; id++)
            {
                int i = id / height, j = id % height;
                // std::cout << i << ' ' << j << std::endl;
                if (layout->getContent(i, j).gridModule != nullptr && layout->getContent(i, j).gridModule->getName() == "FUAT" && gridNetlist[i][j].size())
                {
                    // if (gridNetlist[i][j].size() != maxNets)
                    //     continue;
                    // std::cout << "tile Routing " << i << ' ' << j << std::endl;
                    ILPRoute(env, i, j);

                    // std::cout << "x: " << i << " y: " << j << " Netlist size " << gridNetlist[i][j].size() << " rt : " << duration_ms.count() << std::endl;
                    // getchar();
                    // getchar();
                }
            }
        e = high_resolution_clock::now();
        duration<double, std::ratio<1, 1>> duration_ms(e - s);
        std::cout << "tile Route finish! RT: " << duration_ms.count() << std::endl;
    }

    void TileRouter::ILPRoute(GRBEnv &env, int x, int y)
    {
        int n = localgraph->getVertexNum(), k = gridNetlist[x][y].size();
        // std::cout << "n: " << n << ' ' << "k: " << k << std::endl;
        GRBModel model(env);

        std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, GRBVar>>> R;
        std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, GRBVar>>>> S;
        for (int j = 0; j < k; j++)
            for (int i = 0; i < n; i++)
            {
                for (int l = 0; l < localgraph->getVertexDegree(i); l++)
                {
                    int t = localgraph->getEdge(i, l);
                    R[i][t][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "R_" + std::to_string(i) + "_" + std::to_string(l) + "_" + std::to_string(j));
                    for (int kk = 0; kk < gridNetlist[x][y][j]->getSinkSize(); kk++)
                    {
                        int sink = gridNetlist[x][y][j]->getSinkByIdx(kk);
                        int sinkX = graph->getPos(sink).X(), sinkY = graph->getPos(sink).Y();
                        if (sinkX != x || sinkY != y)
                            continue;
                        S[i][t][j][kk] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "S_" + std::to_string(i) + "_" + std::to_string(l) + "_" + std::to_string(j) + "_" + std::to_string(kk));
                        model.addConstr(S[i][t][j][kk] <= R[i][t][j]);
                    }
                }
            }

        GRBLinExpr obj;
        for (int i = 0; i < n; i++)
            for (int l = 0; l < localgraph->getVertexDegree(i); l++)
                for (int j = 0; j < k; j++)
                    obj += R[i][localgraph->getEdge(i, l)][j];
        model.setObjective(obj, GRB_MINIMIZE);

        // std::cout << "BUILD GRB obj finished!" << std::endl;

        for (int i = 0; i < n; i++)
        {
            // std::cout << i << std::endl;
            GRBLinExpr lhs;
            for (int l = 0; l < localgraph->getInputDegree(i); l++)
            {
                int f = localgraph->getInputEdge(i, l);
                for (int j = 0; j < k; j++)
                    lhs += R[f][i][j];
            }
            model.addConstr(lhs <= graph->getVertexCap(graph->getVertexId(x, y, localgraph->getOriginIdx(i))), "CAP Constr");
        }
        // std::cout << "BUILD GRB CAP Constraints finished!" << std::endl;

        for (int i = 0; i < n; i++)
        {
            int originIdx = graph->getVertexId(x, y, localgraph->getOriginIdx(i));
            for (int j = 0; j < k; j++)
            {
                for (int kk = 0; kk < gridNetlist[x][y][j]->getSinkSize(); kk++)
                {
                    int sink = gridNetlist[x][y][j]->getSinkByIdx(kk);
                    int sinkX = graph->getPos(sink).X(), sinkY = graph->getPos(sink).Y();
                    if (sinkX != x || sinkY != y)
                        continue;
                    GRBLinExpr lhs, rhs;
                    for (int l = 0; l < localgraph->getVertexDegree(i); l++)
                        rhs += S[i][localgraph->getEdge(i, l)][j][kk];
                    for (int l = 0; l < localgraph->getInputDegree(i); l++)
                        lhs += S[localgraph->getInputEdge(i, l)][i][j][kk];
                    if (gridNetlist[x][y][j]->getSource() == originIdx)
                    {
                        model.addConstr(rhs == 1, "Source Constr");
                    }
                    else if (gridNetlist[x][y][j]->getSinkByIdx(kk) == originIdx)
                    {
                        model.addConstr(lhs == 1, "Sink Constr");
                    }
                    else
                    {
                        model.addConstr(lhs == rhs, "fanin equal fanout constr");
                    }
                }
            }
        }

        // std::cout << "Finish adding Constraints" << std::endl;

        model.optimize();
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            auto& routetree = Pathfinder::routetree;
            for (int j = 0; j < k; j++) {
                std::queue<std::pair<int, std::shared_ptr<TreeNode>>> q;
                auto net = gridNetlist[x][y][j];
                std::cout << "net: " << net->getName() << std::endl;
                int source = net->getSource();
                int st = localgraphIdx[graph->getVertexByIdx(source)->getPinId()];
                q.push(std::make_pair(st, routetree.getNetRoot(net)));
                while (!q.empty()) {
                    int now = q.front().first;
                    auto father = q.front().second;
                    q.pop();
                    for (int l = 0; l < localgraph->getVertexDegree(now); l++) {
                        int t = localgraph->getEdge(now, l);
                        if (R[now][t][j].get(GRB_DoubleAttr_X) == 1) {
                            int t_originIdx = graph->getVertexId(x, y, localgraph->getOriginIdx(t));
                            int now_originIdx = graph->getVertexId(x, y, localgraph->getOriginIdx(now));
                            std::cout << graph->getVertexByIdx(now_originIdx)->getName() <<"->" << graph->getVertexByIdx(t_originIdx)->getName() << std::endl;
                            auto treeNode = routetree.addNode(father, t_originIdx, net);
                            q.push(std::make_pair(t, treeNode));
                        }
                    }
                }
            }
        }
    }

    void TileRouter::buildLocalGraph(std::string moduleName)
    {
        auto module = layout->getModuleLibrary()[moduleName];
        int maxVertex = layout->getModulePinNum(moduleName);
        localgraphIdx.resize(maxVertex, -1);
        localgraph = std::make_shared<LocalRouteGraph>();

        addLocalGraphVertex(module);
        addLocalGraphEdge(module);
    }

    void TileRouter::addLocalGraphVertex(std::shared_ptr<database::Module> module)
    {
        auto allPorts = module->allPorts();
        for (auto &it : allPorts)
        {
            if (it.first == "o_gsw_direct" || it.first == "i_gsw")
                continue;
            auto port = it.second;
            for (int i = 0; i < port->getPinNum(); i++)
            {
                auto pin = port->getPinByIdx(i);
                int localId = localgraph->addVertex(pin->getPinId(), 1.0);
                localgraphIdx[pin->getPinId()] = localId;
            }
        }
        auto allSubModules = module->allSubmodules();
        for (auto &it : allSubModules)
        {
            if (it.first.find("gsw") != std::string::npos)
                continue;
            addLocalGraphVertex(it.second);
        }
    }

    void TileRouter::addLocalGraphEdge(std::shared_ptr<database::Module> module)
    {
        auto allPorts = module->allPorts();
        for (auto &it : allPorts)
        {
            if (it.first == "o_gsw_direct" || it.first == "i_gsw")
                continue;
            auto port = it.second;
            for (int i = 0; i < port->getPinNum(); i++)
            {
                auto pin = port->getPinByIdx(i);
                for (int j = 0; j < pin->getConnectSize(); j++)
                {
                    auto toPin = pin->getConnectPinByIdx(j);

                    if (localgraphIdx[toPin->getPinId()] != -1)
                        localgraph->addEdge(localgraphIdx[pin->getPinId()], localgraphIdx[toPin->getPinId()], 1);
                }
            }
        }
        auto allSubModules = module->allSubmodules();
        for (auto &it : allSubModules)
        {
            if (it.first.find("gsw") != std::string::npos)
                continue;
            addLocalGraphEdge(it.second);
        }
    }
}