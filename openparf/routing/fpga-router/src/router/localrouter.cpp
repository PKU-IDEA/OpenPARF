#include "localrouter.h"
// #include "router.h"
#include "utils/printer.h"
#include <iostream>
#include <fstream>
#include <lemon/list_graph.h>
#include <chrono>
#include <future>
#include <queue>
#include <map>
#include "pathfinder.h"
#include "taskflow/taskflow.hpp"

namespace router {
    void LocalRouter::loadRouteResultFromRouteTree(RouteTree& routeTree) {
        for (auto& it : routeTree.getNetRoots()) {
            auto net = it.first;
            auto root = it.second;
            routeTreeDfs(root, nullptr);
        }
    }

    void LocalRouter::routeTreeDfs(std::shared_ptr<TreeNode> node, std::shared_ptr<GridNet> net) {
        int x = graph->getPos(node->nodeId).X(), y =  graph->getPos(node->nodeId).Y();
        if (graph->getVertexCap(node->nodeId) < 0) congestPos.insert(std::make_pair(x, y));
        // graph->addVertexCap(node->nodeId, 1);
        if (net == nullptr) {
            net = std::make_shared<GridNet>(node->net->getName() + "_" + std::to_string(x) + "_" + std::to_string(y), node->net);
            gridNetlist[x][y].push_back(net);
            // netlist.push_back(net);
            net->setSource(node->nodeId, node);
            net->addGuideNode(x, y);
        }
        for (auto child = node->firstChild; child != nullptr; child = child->right) {
            INDEX_T cx = graph->getPos(child->nodeId).X(), cy =  graph->getPos(child->nodeId).Y();
            if (cx != x || cy != y) {
                net->addSink(node->nodeId, node);
                routeTreeDfs(child, nullptr);
            }
            else routeTreeDfs(child, net);
        }
        if (node->firstChild == nullptr) net->addSink(node->nodeId, node);
    }

    void LocalRouter::buildGridNetDfs(std::shared_ptr<TreeNode> node, int x, int y, std::shared_ptr<GridNet> net, std::vector<std::shared_ptr<GridNet> >& netlist, std::unordered_map<int, int>& visited_vertex) {
        // graph->addVertexCap(node->nodeId, 1);
        if (net == nullptr) {
            net = std::make_shared<GridNet>(node->net->getName() + "_" + std::to_string(x) + "_" + std::to_string(y), node->net);
            netlist.push_back(net);
            // netlist.push_back(net);
            net->setSource(node->nodeId, node);
            net->addGuideNode(x, y);
        }
        int nx = graph->getPos(node->nodeId).X(), ny = graph->getPos(node->nodeId).Y();
        if (nx != x || ny != y)
            visited_vertex[node->nodeId]++;
        for (auto child = node->firstChild; child != nullptr; child = child->right) {
            INDEX_T cx = graph->getPos(child->nodeId).X(), cy =  graph->getPos(child->nodeId).Y();

            if (abs(x - cx) + abs(cy - y) > 1 || (layout->getContent(cx, cy).gridModule == nullptr || layout->getContent(cx, cy).gridModule->getName() != "FUAT"))  {
                net->addSink(node->nodeId, node);
            }
            else buildGridNetDfs(child, x, y, net, netlist, visited_vertex);
        }
        if (node->firstChild == nullptr) net->addSink(node->nodeId, node);
    }

    void LocalRouter::ripupDfs(std::shared_ptr<TreeNode> node, std::shared_ptr<GridNet> net) {
        // if (node->nodeId == 45874537) {
        //     std::cout << "DFS on Node " << node->nodeId << " net(origin) " << net->getName() << "(" << node->net->getName() << ") "  << net->isSinkTreeNode(node) << ' ' << net->getSourceTreeNode()->nodeId << std::endl;
        //     std::cout << Pathfinder::routetree.getTreeNodeByIdx(node->nodeId) << ' ' << node << std::endl;
        // }
        if (net->isSinkTreeNode(node)) return;
        for (auto child = node->firstChild; child != nullptr; child = child->right) {
            ripupDfs(child, net);
        }
        if (net->getSourceTreeNode() != node)
            Pathfinder::routetree.eraseNode(node);
    }

    void LocalRouter::testRun() {
        tf::Taskflow taskflow;
        int width = graph->getWidth(), height = graph->getHeight();
        std::map<std::pair<int, int>, tf::Task> tasks;
        for (auto& pos : congestPos) {
            tasks[pos] = taskflow.emplace(
                [&]() {
                //     std::cout << "Starting..." << std::endl;
                //     std::cout << pos.first << ' ' << pos.second << ' '  std::endl;
                    if (layout->getContent(pos.first, pos.second).gridModule != nullptr && layout->getContent(pos.first, pos.second).gridModule->getName() == "FUAT")
                    tryILP(pos.first, pos.second);
                }
            );
        }

        for (auto& it : tasks) {
            if (congestPos.find(std::make_pair(it.first.first - 1, it.first.second - 1)) != congestPos.end())
                it.second.succeed(tasks[std::make_pair(it.first.first - 1, it.first.second - 1)]);
            if (congestPos.find(std::make_pair(it.first.first - 1, it.first.second)) != congestPos.end())
                it.second.succeed(tasks[std::make_pair(it.first.first - 1, it.first.second)]);
            if (congestPos.find(std::make_pair(it.first.first - 1, it.first.second + 1)) != congestPos.end())
                it.second.succeed(tasks[std::make_pair(it.first.first - 1, it.first.second + 1)]);
            if (congestPos.find(std::make_pair(it.first.first, it.first.second - 1)) != congestPos.end())
                it.second.succeed(tasks[std::make_pair(it.first.first, it.first.second - 1)]);

        }
        tf::Executor executor(16);
        std::cout << "starting taskflow..." << std::endl;
        executor.run(taskflow).wait();
        std::cout << "taskflow finished!" << std::endl;
        return;
//         int n = congestPosVec.size();
// // #pragma omp parallel for num_threads(16)
//         for (int i = 0; i < n; i++) {
//             tryILP(congestPosVec[i].first, congestPosVec[i].second);
//         }
    }

    std::shared_ptr<LocalRouteGraph> LocalRouter::dumpRouteGraph(int x, int y) {
        std::unordered_map<int, int> newVertexId;
        std::shared_ptr<LocalRouteGraph> localgraph(new LocalRouteGraph());
        int height = graph->getHeight();
        for (auto it : graph->vertexIds()[x * height + y]) {
            newVertexId[it] = localgraph->addVertex(it, graph->getVertexCost(it));
        }
        for (auto it : newVertexId) {
            int degree = graph->getVertexDegree(it.first);
            for (int i = 0; i < degree; i++) {
                int sink = graph->getEdge(it.first, i);
                if (newVertexId.find(sink) != newVertexId.end()) {
                    // std::cout << graph->getVertexByIdx(it.first)->getName() << "->" << graph->getVertexByIdx(sink)->getName() << std::endl;
                    localgraph->addEdge(it.second, newVertexId[sink], graph->getEdgeCost(it.first, i));
                }
            }
        }
        return localgraph;
    }

    void LocalRouter::tryILP(int x, int y) {
        // std::cout << "try ILP at " << x << ' ' << y << std::endl;
        std::unordered_map<int, int> newVertexId;
        std::shared_ptr<LocalRouteGraph> localgraph(new LocalRouteGraph());
        int height = graph->getHeight(), width = graph->getWidth();
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                // std::cout << x + i << ' ' << y + j << std::endl;
                if ((i == 0 || j == 0) && x + i < width && x + i >= 0 && y + j < height && y + j >= 0) {
                    // std::cout << x + i << ' ' << y + j << ' ' << layout->getContent(x + i, y + j).gridModule << std::endl;
                    if (layout->getContent(x + i, y + j).gridModule == nullptr || layout->getContent(x + i, y + j).gridModule->getName() != "FUAT")
                        continue;
                    for (auto it : graph->vertexIds()[(x + i) * height + (y + j)]) {
                        newVertexId[it] = localgraph->addVertex(it, graph->getVertexCost(it));
                    }
                }
            }
        }
        // std::cout << "add vertex finished!" << std::endl;
        for (auto it : newVertexId) {
            int degree = graph->getVertexDegree(it.first);
            for (int i = 0; i < degree; i++) {
                int sink = graph->getEdge(it.first, i);
                if (newVertexId.find(sink) != newVertexId.end()) {
                    // std::cout << graph->getVertexByIdx(it.first)->getName() << "->" << graph->getVertexByIdx(sink)->getName() << std::endl;
                    localgraph->addEdge(it.second, newVertexId[sink], graph->getEdgeCost(it.first, i));
                }
            }
        }
        std::vector<std::shared_ptr<GridNet> > netlist;
        std::unordered_map<int, int> visited_vertex;
        for (auto net : gridNetlist[x][y]) {
            buildGridNetDfs(net->getSourceTreeNode(), x, y, nullptr, netlist, visited_vertex);
        }
        int n = localgraph->getVertexNum(), k = netlist.size();
        // std::cout << "n : " << n << " k: " << k << std::endl;

        // GRBEnv env = GRBEnv();
        //  env.set(GRB_IntParam_OutputFlag, 0);
        //  env.set(GRB_IntParam_Threads, 1);
        //  env.set(GRB_IntParam_MIPFocus, 1);


        GRBModel model = GRBModel(env);

        // std::cout << "BUILD GRB ENV Finished!" << std::endl;

        std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, GRBVar> > > R;
        std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, GRBVar> > > > S;
        for (int j = 0; j < k; j++)
            for (int i = 0; i < n; i++) {
                for (int l = 0; l < localgraph->getVertexDegree(i); l++) {
                    int t = localgraph->getEdge(i, l);
                    R[i][t][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "R_" + std::to_string(i) + "_" + std::to_string(l) + "_" + std::to_string(j));
                    for (int kk = 0; kk < netlist[j]->getSinkSize(); kk++) {
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

        for (int i = 0; i < n; i++) {
            // std::cout << i << std::endl;
            GRBLinExpr lhs;
            for (int l = 0; l < localgraph->getInputDegree(i); l++) {
                int f = localgraph->getInputEdge(i, l);
                for (int j = 0; j < k; j++)
                    lhs += R[f][i][j];
            }
            int originIdx = localgraph->getOriginIdx(i);
            int nx = graph->getPos(originIdx).X(), ny = graph->getPos(originIdx).Y();
            if (nx == x && ny == y)
                model.addConstr(lhs <= graph->getVertexMaxCap(localgraph->getOriginIdx(i)), "CAP Constr");
            else
                model.addConstr(lhs <= graph->getVertexCap(originIdx) + visited_vertex[originIdx], "CAP Constr");
        }
        // std::cout << "BUILD GRB CAP Constraints finished!" << std::endl;

        for (int i = 0; i < n; i++) {
            int originIdx = localgraph->getOriginIdx(i);
            for (int j = 0; j < k; j++) {
                for (int kk = 0; kk < netlist[j]->getSinkSize(); kk++) {
                    GRBLinExpr lhs, rhs;
                    for (int l = 0; l < localgraph->getVertexDegree(i); l++)
                        rhs += S[i][localgraph->getEdge(i, l)][j][kk];
                    for (int l = 0; l < localgraph->getInputDegree(i); l++)
                        lhs += S[localgraph->getInputEdge(i, l)][i][j][kk];
                    if (netlist[j]->getSource() == originIdx) {
                        model.addConstr(rhs == 1, "Source Constr");
                    }
                    else if (netlist[j]->getSinkByIdx(kk) == originIdx) {
                        model.addConstr(lhs == 1, "Sink Constr");
                    }
                    else {
                        model.addConstr(lhs == rhs, "fanin equal fanout constr");
                    }
                }
            }
        }

        // std::cout << "Finish adding Constraints" << std::endl;

        model.optimize();

        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            auto &routetree = Pathfinder::routetree;
            for (int j = 0; j < k; j++) {
                auto net = netlist[j];
                // std::cout << "working on " << net->getName() << std::endl;
                ripupDfs(net->getSourceTreeNode(), net);
                // std::cerr << "rip up dfs finished!" << std::endl;
                std::queue<std::pair<int, std::shared_ptr<TreeNode>>> q;
                int source = net->getSource();
                int st = newVertexId[source];
                q.push(std::make_pair(st, net->getSourceTreeNode()));
                while (!q.empty()) {
                    int now = q.front().first;
                    auto father = q.front().second;
                    // std::cout << now << ' ' << father->net->getName() << std::endl;
                    q.pop();
                    for (int l = 0; l < localgraph->getVertexDegree(now); l++) {
                        int t = localgraph->getEdge(now, l);
                        if (R[now][t][j].get(GRB_DoubleAttr_X) == 1) {
                            // std::cout << "used " << now << ' ' << t << std::endl;
                            int t_originIdx = localgraph->getOriginIdx(t);
                            if (net->isSink(t_originIdx)) {
                                // std::cout << "adding " << t_originIdx << std::endl;
                                auto node = net->getSinkTreeNodeByIdx(net->getSinkIdx(t_originIdx));
                                if (node == nullptr || node->net != net->getOriginNet()) {
                                    std::cout << "[error] sink not unique" << std::endl;
                                    exit(-1);
                                }
                                node->father = father;
                                if (father->firstChild != node) {
                                // std::cout << node << ' ' << node->left << ' ' << node->right << std::endl;
                                if (node->left != nullptr) node->left->right = node->right;
                                if (node->right != nullptr) node->right->left = node->left;
                                node->right = father->firstChild;
                                if (father->firstChild != nullptr) father->firstChild->left = node;
                                node->left = nullptr;
                                father->firstChild = node;
                                }
                                // std::cout << father << ' ' << node << ' ' << father->firstChild << ' ' << node->father << ' ' << node->left << ' ' << node->right << std::endl;
                            }
                            else {
                                // std::cout << "adding " << t_originIdx << std::endl;
                                auto node = routetree.addNode(father, t_originIdx, net->getOriginNet());
                                // std::cout << "added node " << node << ' ' << node->firstChild << std::endl;
                                q.push(std::make_pair(t, node));
                            }
                        }
                    }
                }
            }
        }
    }

    void LocalRouter::dumpLocalGraphAndPrint(int x, int y, std::string fileName) {
        std::unordered_map<int, int> newVertexId;
        std::shared_ptr<LocalRouteGraph> localgraph(new LocalRouteGraph());
        int height = graph->getHeight();
        for (auto it : graph->vertexIds()[x * height + y]) {
            newVertexId[it] = localgraph->addVertex(it, graph->getVertexCost(it));
        }
        for (auto it : newVertexId) {
            int degree = graph->getVertexDegree(it.first);
            for (int i = 0; i < degree; i++) {
                int sink = graph->getEdge(it.first, i);
                if (newVertexId.find(sink) != newVertexId.end()) {
                    // std::cout << graph->getVertexByIdx(it.first)->getName() << "->" << graph->getVertexByIdx(sink)->getName() << std::endl;
                    localgraph->addEdge(it.second, newVertexId[sink], graph->getEdgeCost(it.first, i));
                }
            }
        }
        // std::cout << "BUILD GRAPH Finished" << std::endl;
        int n = localgraph->getVertexNum(), k = gridNetlist[x][y].size();
        // std::cout << "n : " << n << " k: " << k << std::endl;

        GRBEnv env = GRBEnv();
         env.set(GRB_IntParam_OutputFlag, 0);
         env.set(GRB_IntParam_Threads, 1);
         env.set(GRB_IntParam_MIPFocus, 1);


        GRBModel model = GRBModel(env);

        // std::cout << "BUILD GRB ENV Finished!" << std::endl;

        std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, GRBVar> > > R;
        std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, GRBVar> > > > S;
        for (int j = 0; j < k; j++)
            for (int i = 0; i < n; i++) {
                for (int l = 0; l < localgraph->getVertexDegree(i); l++) {
                    int t = localgraph->getEdge(i, l);
                    R[i][t][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "R_" + std::to_string(i) + "_" + std::to_string(l) + "_" + std::to_string(j));
                    for (int kk = 0; kk < gridNetlist[x][y][j]->getSinkSize(); kk++) {
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

        for (int i = 0; i < n; i++) {
            // std::cout << i << std::endl;
            GRBLinExpr lhs;
            for (int l = 0; l < localgraph->getInputDegree(i); l++) {
                int f = localgraph->getInputEdge(i, l);
                for (int j = 0; j < k; j++)
                    lhs += R[f][i][j];
            }
            model.addConstr(lhs <= graph->getVertexMaxCap(localgraph->getOriginIdx(i)), "CAP Constr");
        }
        // std::cout << "BUILD GRB CAP Constraints finished!" << std::endl;

        for (int i = 0; i < n; i++) {
            int originIdx = localgraph->getOriginIdx(i);
            for (int j = 0; j < k; j++) {
                for (int kk = 0; kk < gridNetlist[x][y][j]->getSinkSize(); kk++) {
                    GRBLinExpr lhs, rhs;
                    for (int l = 0; l < localgraph->getVertexDegree(i); l++)
                        rhs += S[i][localgraph->getEdge(i, l)][j][kk];
                    for (int l = 0; l < localgraph->getInputDegree(i); l++)
                        lhs += S[localgraph->getInputEdge(i, l)][i][j][kk];
                    if (gridNetlist[x][y][j]->getSource() == originIdx) {
                        model.addConstr(rhs == 1, "Source Constr");
                    }
                    else if (gridNetlist[x][y][j]->getSinkByIdx(kk) == originIdx) {
                        model.addConstr(lhs == 1, "Sink Constr");
                    }
                    else {
                        model.addConstr(lhs == rhs, "fanin equal fanout constr");
                    }
                }
            }
        }

        // std::cout << "Finish adding Constraints" << std::endl;

        model.optimize();

        // ofstream ofs("ilp.out");
        // for (int j = 0; j < k; j++)
        //     for (int i = 0; i < n; i++)
        //         for (int l = 0; l < localgraph->getVertexDegree(i); l++){
        //         if (R[i][localgraph->getEdge(i, l)][j].get(GRB_DoubleAttr_X) == 1) {
        //             ofs << gridNetlist[x][y][j]->getName() << ' ' << graph->getVertexByIdx(localgraph->getOriginIdx(i))->getName() << "->" << graph->getVertexByIdx(localgraph->getOriginIdx(localgraph->getEdge(i, l)))->getName() << std::endl;
        //         }
        //     }
        // ofs.close();

                // model.computeIIS();
        // model.write("problem.ilp");

        // std::ofstream ofs(fileName);
        // ofs << localgraph->getVertexNum() * 2 << std::endl;
        // for (int i = 0; i < localgraph->getVertexNum(); i++) {
        //     ofs << i * 2 << ' ' << x << ' ' << y << std::endl;
        //     ofs << i * 2 + 1 << ' ' << x << ' ' << y << std::endl;

        // }
        // ofs << localgraph->getEdgeNum() + localgraph->getVertexNum() << std::endl;
        // int id = 0;
        // for (int i = 0; i < localgraph->getVertexNum(); i++) {
        //     ofs << id++ << ' ' << i * 2 << ' ' << i * 2 + 1 << ' ' << graph->getVertexMaxCap(localgraph->getOriginIdx(i)) << ' ' << 1 << std::endl;
        //     int degree = localgraph->getVertexDegree(i);
        //     for (int j = 0; j < degree; j++)
        //         ofs << id++ << ' ' << i * 2 + 1 << ' ' << localgraph->getEdge(i, j) * 2 << ' ' << 1 << ' ' << 1 << std::endl;
        // }

        // int totalSinks = 0;
        // for (auto net : gridNetlist[x][y]) {
        //     totalSinks += net->getSinkSize();
        // }
        // ofs << totalSinks << std::endl;
        // id = 0;
        // for (auto net : gridNetlist[x][y]) {
        //     for (auto sink : net->getSinks()) {
        //         ofs << id++ << ' ' <<  newVertexId[net->getSource()] * 2 << ' ' << newVertexId[sink] * 2 + 1 << ' ' << 1.0 / net->getSinkSize() << std::endl;
        //     }
        // }
        // ofs.close();

        // using namespace lemon;
        // typedef ListDigraph Digraph;

        // Digraph G;
        // Digraph::ArcMap<int> cost(G), cap(G);
        // Digraph::NodeMap<int> sup(G);
        // Digraph::NodeMap<std::string> name(G);
        // std::vector<Digraph::Node> nodes;
        // std::vector<Digraph::Arc> arcs;
        // int n = localgraph->getVertexNum();
        // for (int i = 0; i < n; i++) {
        //     nodes.push_back(G.addNode());
        //     name.set(nodes[i], graph->getVertexByIdx(localgraph->getOriginIdx(i))->getName());
        // }
        // for (int i = 0; i < n; i++) {
        //     int degree = localgraph->getVertexDegree(i);
        //     for (int j = 0; j < degree; j++) {
        //         auto arc = G.addArc(nodes[i], nodes[localgraph->getEdge(i, j)]);
        //         cost.set(arc, 120);
        //         cap.set(arc, 120);
        //         arcs.push_back(arc);
        //     }
        // }
        // NetworkSimplex<Digraph> ns(G);
        // ns.upperMap(cap);
        // ns.costMap(cost);
        // for (auto net : gridNetlist[x][y]) {
        //     sup.set(nodes[newVertexId[net->getSource()]], 120);
        //     for (auto sink : net->getSinks()) {
        //         sup.set(nodes[newVertexId[sink]], -120 / net->getSinkSize());
        //     }
        // }
        // ns.supplyMap(sup);

        // std::ofstream ofs2("lemon.graph");
        // DigraphWriter<Digraph> writer(G, ofs2);
        // writer.nodeMap("supply", sup).arcMap("cap", cap).arcMap("cost", cost).nodeMap("name", name).run();
        // ofs2.close();
        // ofs2.open("lemon.out");

        // using namespace std::chrono;
        // high_resolution_clock::time_point st, ed;
        // st = high_resolution_clock::now();
        // ofs2 << ns.run() << std::endl;
        // ed = high_resolution_clock::now();
        // duration<double, std::ratio<1, 1000>> duration_ms(ed - st);
        // std::cout << "MCF RT: " << duration_ms.count() << "ms" << std::endl;
        // for (int i = 0; i < arcs.size(); i++) {
        //     int source = G.id(G.source(arcs[i]));
        //     int sink = G.id(G.target(arcs[i]));
        //     source = localgraph->getOriginIdx(source);
        //     sink = localgraph->getOriginIdx(sink);
        //     ofs2 << graph->getVertexByIdx(source)->getName() << ' ' << graph->getVertexByIdx(sink)->getName() << ' ' << ns.flow(arcs[i]) << std::endl;
        // }
        // ofs2.close();


    }

    void LocalRouter::tryMCMCFRoute(int x, int y) {
        MCF mcf = MCF();
        std::unordered_map<int, int> newVertexId;
        std::shared_ptr<LocalRouteGraph> localgraph(new LocalRouteGraph());
        int height = graph->getHeight();
        for (auto it : graph->vertexIds()[x * height + y]) {
            newVertexId[it] = localgraph->addVertex(it, graph->getVertexCost(it));
        }
        for (auto it : newVertexId) {
            int degree = graph->getVertexDegree(it.first);
            for (int i = 0; i < degree; i++) {
                int sink = graph->getEdge(it.first, i);
                if (newVertexId.find(sink) != newVertexId.end()) {
                    // std::cout << graph->getVertexByIdx(it.first)->getName() << "->" << graph->getVertexByIdx(sink)->getName() << std::endl;
                    localgraph->addEdge(it.second, newVertexId[sink], graph->getEdgeCost(it.first, i));
                }
            }
        }
        int n = localgraph->getVertexNum() * 2, m = localgraph->getEdgeNum() + localgraph->getVertexNum();
        mcf.nodes = (NODE *)malloc(sizeof(NODE) * (n));
        mcf.no_node = n;
        for (int i = 0; i < n; i++) {
            mcf.nodes[i].id = i;
            mcf.nodes[i].x = x;
            mcf.nodes[i].y = y;
            mcf.nodes[i].pre = -1;
            mcf.nodes[i].dist = DBL_MAX;
            mcf.nodes[i].no_comm = 0;
            mcf.nodes[i].comms = NULL;
            mcf.nodes[i].no_edge = 0;
            mcf.nodes[i].dij_visited = 0;
            mcf.nodes[i].dij_updated = 0;
            mcf.nodes[i].min_visited = 0;
            if (i % 2 == 0)
                mcf.nodes[i].edges = (int *)malloc(sizeof(int) * 1);
            else
                mcf.nodes[i].edges = (int *)malloc(sizeof(int) * localgraph->getVertexDegree(i / 2));
        }

        mcf.edges = (EDGE *)malloc(sizeof(EDGE) * m);
        mcf.no_edge = m;
        mcf._temp_edge_flow = (double*)malloc(sizeof(double) * (n));
        int id = 0;
        for (int i = 0; i < localgraph->getVertexNum(); i++) {
            mcf.edges[id].id = id;
            mcf.edges[id].src = i * 2;
            mcf.edges[id].dest = i * 2 + 1;
            mcf.edges[id].capacity = graph->getVertexMaxCap(localgraph->getOriginIdx(i));
            mcf.edges[id].left_capacity = graph->getVertexMaxCap(localgraph->getOriginIdx(i));
            mcf.edges[id].latency = 1.0;
            mcf.edges[id].length = 0.0;

            mcf.edges[i].flow = 0.0;
            mcf.edges[i]._flows = NULL;
            mcf.nodes[mcf.edges[id].src].edges[mcf.nodes[mcf.edges[id].src].no_edge] = id;
            mcf.nodes[mcf.edges[id].src].no_edge++;
            id++;

            int degree = localgraph->getVertexDegree(i);
            for (int j = 0; j < degree; j++) {
                mcf.edges[id].id = id;
                mcf.edges[id].src = i * 2 + 1;
                mcf.edges[id].dest = localgraph->getEdge(i, j) * 2;
                mcf.edges[id].capacity = 1.0;
                mcf.edges[id].left_capacity = 1.0;
                mcf.edges[id].latency = 1.0;
                mcf.edges[id].length = 0.0;

                mcf.edges[i].flow = 0.0;
                mcf.edges[i]._flows = NULL;
                mcf.nodes[mcf.edges[id].src].edges[mcf.nodes[mcf.edges[id].src].no_edge] = id;
                mcf.nodes[mcf.edges[id].src].no_edge++;
                id++;
            }
        }
        int totalSinks = 0;
        for (auto net : gridNetlist[x][y]) {
            totalSinks += net->getSinkSize();
        }
        mcf._commodities = (COMMODITY *)malloc(sizeof(COMMODITY) * (totalSinks));
        mcf.no_commodity = totalSinks;
        id = 0;
        for (auto net : gridNetlist[x][y]) {
            for (auto sink : net->getSinks()) {
                mcf._commodities[id].id = id;
                mcf._commodities[id].src = newVertexId[net->getSource()] * 2;
                mcf._commodities[id].dest = newVertexId[sink] * 2 + 1;
                mcf._commodities[id].demand = 1.0 / net->getSinkSize();
                mcf._commodities[id].left_demand = 1.0 / net->getSinkSize();

                if (mcf.nodes[mcf._commodities[id].src].comms == NULL) {
                    mcf.nodes[mcf._commodities[id].src].comms = (int *)malloc(sizeof(int) * n);
                }
                mcf.nodes[mcf._commodities[id].src].comms[mcf.nodes[mcf._commodities[id].src].no_comm] = id;
                mcf.nodes[mcf._commodities[id].src].no_comm++;
                id++;
            }
        }

        for (int i = 0; i < m; i++) {
            mcf.edges[i]._flows = (double *)malloc(sizeof(double) * (totalSinks));
            for (int j = 0; j < totalSinks; j++)
                mcf.edges[i]._flows[j] = 0.0;
        }
        for (int i = 0; i < n; i++) {
            mcf.nodes[i]._preferred_path = (int *)malloc(sizeof(int) * totalSinks);
            for (int j = 0; j < totalSinks; j++)
                mcf.nodes[i]._preferred_path[j] = -1;
        }

        mcf.init_param();
        mcf.print_network_demands();
        exit(0);
        std::cout << "START SOLVE MCF" << std::endl;
        int max_iter = 10000;
        for (int iter = 0; iter < max_iter; iter++) {
            std::cout << "ITER #" << iter << endl;
            mcf.run_mcf_solver();
            mcf.do_randomized_rounding();
            std::vector<std::set<int>> used(n);
            for (int i = 0; i < totalSinks; i++) {
                int src_id = mcf._commodities[i].src;
                int src = src_id;
                while (src_id != mcf._commodities[i].dest) {
                    if (used[src_id].find(src) == used[src_id].end()) used[src_id].insert(src_id);
                    src_id = mcf.nodes[src_id]._preferred_path[i];
                }
                if (used[src_id].find(src) == used[src_id].end()) used[src_id].insert(src_id);
            }
            bool finished = true;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < totalSinks; j++) {
                    int nex = mcf.nodes[i]._preferred_path[j];
                    if (nex == -1) continue;
                    if (used[nex].size() > graph->getVertexMaxCap(localgraph->getOriginIdx(nex / 2))) {
                        for (int k = 0; k < mcf.nodes[i].no_edge; k++)
                            if (mcf.edges[mcf.nodes[i].edges[k]].dest == nex) {
                                mcf.edges[mcf.nodes[i].edges[k]].latency += 2;
                                finished = 0;
                            }
                    }
                }
            }
            mcf.print_routing_paths();
            if (!finished) {
                for (int i = 0; i < n; i++) {
                    mcf.nodes[i].pre = -1;
                    mcf.nodes[i].dist = DBL_MAX;
                    mcf.nodes[i].no_comm = 0;
                    mcf.nodes[i].comms = NULL;
                    mcf.nodes[i].no_edge = 0;
                    mcf.nodes[i].dij_visited = 0;
                    mcf.nodes[i].dij_updated = 0;
                    mcf.nodes[i].min_visited = 0;
                    for (int j = 0; j < totalSinks; j++) {
                        mcf.nodes[i]._preferred_path[j] = -1;
                    }
                }
                for (int i = 0; i < m; i++) {
                    mcf.edges[i].left_capacity = mcf.edges[i].capacity;
                    mcf.edges[i].length = 0.0;

                    mcf.edges[i].flow = 0.0;
                    mcf.edges[i]._flows = NULL;
                    for (int j = 0; j < totalSinks; j++) {
                       mcf.edges[i]._flows[j] = 0.0;
                    }
                }
                for (int i = 0; i < totalSinks; i++) {
                    mcf._commodities[i].left_demand = mcf._commodities[i].demand;

                }
            }
            else break;
        }
        mcf.free_topology();
    }

    void LocalRouter::dumpLemonDiGraph(std::shared_ptr<LocalRouteGraph> localgraph, int x, int y) {

    }
}