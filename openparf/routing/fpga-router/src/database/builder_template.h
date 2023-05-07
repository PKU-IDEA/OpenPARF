#ifndef BUILDER_TEMPLATE_H
#define BUILDER_TEMPLATE_H

#include "builder.h"

namespace database {

    class GridContent {
    public:
        GridContent() {}
        GridContent(std::string _name, std::shared_ptr<Module> _gridModule, int _width, int _height)
            : name(_name), gridModule(_gridModule), width(_width), height(_height) {}

        std::string name;
        std::shared_ptr<Module> gridModule;
        int width, height;
    };

    class GridLayout {
    public:
        GridLayout() {}
        GridLayout(int _width, int _height);
        void addModuleTemplate(std::string moduleName, pugi::xml_node moduleInfo, pugi::xml_node archInfo);
        
        void addContent(std::string name, std::string type, int width, int height, int x, int y) {
            layout[x][y] = GridContent(name, moduleLibrary[type], width, height);
            totalPins += modulePinNum[type];
        }
        GridContent getContent(int x, int y) {
            return layout[x][y];
        }

        void setPriority(int prio, int x, int y) { priority[x][y] = std::max(priority[x][y], prio); }
        int getPriority(int x, int y) { return priority[x][y]; }
        void setPriority(int prio, int startX, int endX, int startY, int endY); 
        int getPriority(int startX, int endX, int startY, int endY);

        int getwidth() {return width;}
        int getHeight() {return height;}

        std::unordered_map<std::string, std::shared_ptr<Module> > &getModuleLibrary() { return moduleLibrary; }
        std::vector<std::vector<GridContent> >& getModuleLayout() { return layout; }
        int getTotalPins() { return totalPins; }
        int getModulePinNum(std::string name) { return modulePinNum[name]; }

    private:
        std::unordered_map<std::string, std::shared_ptr<Module> > moduleLibrary;
        std::unordered_map<std::string, int> modulePinNum;
        std::vector<std::vector<GridContent> > layout;
        std::vector<std::vector<int> > priority;
        int width, height;
        int totalPins;
    };

    std::shared_ptr<GridLayout> buildGridLayout(pugi::xml_node archInfo);
} // namespace database

#endif // BUILDER_TEMPLATE_H