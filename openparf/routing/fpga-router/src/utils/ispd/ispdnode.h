#ifndef ISPDNODE_H
#define ISPDNODE_H

#include <string>

enum class NodeType {
    FDRE,
    LUT6,
    LUT5,
    LUT4,
    LUT3,
    LUT2,
    LUT1,
    CARRY8,
    DSP48E2,
    RAMB36E2,
    BUFGCE,
    IBUF,
    OBUF,
    UNDEFINE
};

class ISPDNode {
public:
    ISPDNode() {}
    ISPDNode(NodeType _type, int x, int y, int _bel) : type(_type), posx(x), posy(y), bel(_bel) {}
    ISPDNode(NodeType _type) : type(_type), posx(0), posy(0), bel(0) {}
    NodeType type;
    int posx;
    int posy;
    int bel;
};

#endif // ISPDNODE_H