#ifndef GRID_H
#define GRID_H
#include <iostream>

#include <memory>
#include <string>
#include <vector>

namespace database {

const int NO_MODULE = -1;

class Module;

class Grid {

public:
    Grid() {}
    Grid(std::string _name, int _width, int _height);
    ~Grid(){
        // std::cout << "Deleting Grid " + name << std::endl;
        // for (int i = 0; i < width; i++) {
        //     gridModuleName[i].clear();
        //     gridPriority[i].clear();
        //     gridModule[i].clear();
        // }
        // gridModuleName.clear();
        // gridPriority.clear();
        // gridModule.clear();

    }
    
    std::string getName() {return name;}
    int getWidth() {return width;}
    int getHeight() {return height;}

    void setPriority(int prio, int startX, int endX, int startY, int endY); 
    int getPriority(int startX, int endX, int startY, int endY);
    void setGridModule(std::shared_ptr<Module> module, std::string moduleName, int x, int y, int moduleWidth, int moduleHeight);
    std::shared_ptr<Module> getModule(int x, int y);
    std::string getModuleName(int x, int y) { return gridModuleName[x][y]; }

    std::shared_ptr<Module> getGridGSW(int x, int y) { return gridGSW[x][y]; }
    void setGridGSW(int x, int y, std::shared_ptr<Module> gsw) { gridGSW[x][y] = gsw; }

private:
    std::string name;
    int width;
    int height;
    std::vector<std::vector<std::shared_ptr<Module> > > gridModule;
    std::vector<std::vector<int> > gridPriority;
    std::vector<std::vector<std::string> > gridModuleName;
    std::vector<std::vector<std::shared_ptr<Module> > > gridGSW;
    
};

} // namespace database

#endif