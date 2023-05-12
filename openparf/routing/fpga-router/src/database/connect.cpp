#include "connect.h"

namespace database {

Connect::Connect(std::string _name, std::shared_ptr<Pin> _output, ConnectType _type) {
    name = _name;
    output = _output;
    type = _type;
    wireLength = 1.0;
    cost = 1.0;
}

Connect::Connect(std::string _name, std::shared_ptr<Pin> _output, ConnectType _type, COST_T wl) {
    name = _name;
    output = _output;
    type = _type;
    wireLength = wl;
    cost = 1.0;
}


} // namespace database