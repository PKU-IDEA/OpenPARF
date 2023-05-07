#include <algorithm>
#include <iostream>

#include "grid.h"
#include "module.h"

namespace database {

Grid::Grid(std::string _name, int _width, int _height) {
    name = _name;
    width = _width;
    height = _height;

    gridModule.resize(width);
    gridPriority.resize(width);
    gridModuleName.resize(width);
    gridGSW.resize(width);

    for (int i = 0; i < width; i++) {
        gridModule[i].resize(height);
        gridPriority[i].resize(height);
        gridModuleName[i].resize(height);
        gridGSW[i].resize(height, nullptr);
    }

    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++) {
            gridModule[i][j] = std::shared_ptr<Module>(NULL);
            gridPriority[i][j] = NO_MODULE;
        }
}

void Grid::setPriority(int prio, int startX, int endX, int startY, int endY) {
    // std::cout << gridPriority.size() << std::endl;
    for (int i = startX; i <= endX; i++) {
        // std::cout << i << ' ' << gridPriority[i].size() << std::endl;
        for (int j = startY; j <= endY; j++) {
            // std::cout << i << ' ' << j << ' ' << prio << ' ' << gridPriority[i][j] << std::endl;
            if (prio > gridPriority[i][j])
                gridPriority[i][j] = prio;
        }
    }
}

int Grid::getPriority(int startX, int endX, int startY, int endY) {
    int prio = NO_MODULE;
    for (int i = startX; i <= endX; i++)
        for (int j = startY; j <= endY; j++)
            prio = std::max(prio, gridPriority[i][j]);
    return prio;
}

void Grid::setGridModule(std::shared_ptr<Module> module, std::string moduleName, int x, int y, int moduleWidth, int moduleHeight) {
    for (int i = x; i < x + moduleWidth; i++)
        for (int j = y; j < y + moduleHeight; j++) {
            gridModule[i][j] = module;
            if (i == x && j == y) {
                gridModuleName[i][j] = moduleName;
                // std::cout <<i << ' ' << j << ' ' << moduleName << std::endl;
            }
            else {
                gridModuleName[i][j] = "<" + moduleName + ">";
            }
        }
}

std::shared_ptr<Module> Grid::getModule(int x, int y) {
    return gridModule[x][y];
}

} // namespace database