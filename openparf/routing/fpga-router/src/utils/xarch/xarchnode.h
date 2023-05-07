#ifndef XArchNODE_H
#define XArchNODE_H

#include "utils/ispd/ispdnode.h"

enum class XArchNodeType {
    DUMMY,
    LUT5,
    LUT6,
    LRAM,
    SHIFT,
    DFF,
    CLA4,
    INPAD,
    OUTPAD,
    GCU0,
    BRAM36K,
    RAMB,
    UNDEFINED
};

class XArchNode {
public:
    XArchNode() {}
    XArchNode(XArchNodeType _type, int x, int y, int _bel) : type(_type), posx(x), posy(y), bel(_bel) {}
    XArchNode(XArchNodeType _type) : type(_type), posx(0), posy(0), bel(0) {}
    XArchNodeType type;
    int posx;
    int posy;
    int bel;
};

#endif //XArchNODE_H