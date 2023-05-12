#ifndef UTILS_H
#define UTILS_H


#include <sys/resource.h>
#include <unistd.h>

using COST_T = float;
using INDEX_T = int;

const COST_T baseCost = 2.6274e-11;

const int highFanoutThres = 64;
const INDEX_T maxHighFanoutAddDist = 5;

template<class T> 
class XY {
public:
    XY() {}
    XY(T _x, T _y):x(_x),y(_y) {}

    T X() { return x; }
    T Y() { return y; }

    bool operator ==(const XY<T>& a) const {
        return x == a.x && y == a.y;
    }

    
private:
    T x;
    T y;
}; 

struct BoundingBox {
    INDEX_T start_x;
    INDEX_T end_x;
    INDEX_T start_y;
    INDEX_T end_y;
};

double get_memory_peak();
double get_memory_current();
#endif //UTILS_H