#include "teca_bad_cast.h"

#include <sstream>

// --------------------------------------------------------------------------
teca_bad_cast::teca_bad_cast(const std::string &from, const std::string &to)
{
    std::ostringstream oss;
    oss << "Failed to cast from " << from << " to " << to;
    m_what = oss.str();
}
