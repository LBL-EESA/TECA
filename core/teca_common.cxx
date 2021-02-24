#include "teca_common.h"

namespace std
{
// **************************************************************************
std::ostream &operator<<(std::ostream &os, const std::vector<std::string> &vec)
{
    if (!vec.empty())
    {
        os << "\"" << vec[0] << "\"";
        size_t n = vec.size();
        for (size_t i = 1; i < n; ++i)
            os << ", \"" << vec[i] << "\"";
    }
    return os;
}
}

// **************************************************************************
int have_tty()
{
    static int have = -1;
    if (have < 0)
        have = isatty(fileno(stderr));
    return have;
}
