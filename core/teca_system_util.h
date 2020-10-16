#ifndef teca_system_util_h
#define teca_system_util_h

#include "teca_common.h"
#include "teca_string_util.h"

#include <cstdlib>

namespace teca_system_util
{
// initialize val with the environment variable named by var converted to a
// numeric type. Only floating point and signed integers are implemented. For
// unsigned types, check that the return is greater or equal to zero.
//
// returns:
//    0  if the variable was found and val was initialized from it
//    1  if the varibale was not found
//   -1  if the variable was found but conversion from string failed
template <typename T>
int get_environment_variable(const char *var, T &val)
{
    const char *tmp = getenv(var);
    if (tmp)
    {
        if (teca_string_util::string_tt<T>::convert(tmp, val))
        {
            TECA_ERROR("Failed to convert " << var << " = \""
                << tmp << "\" to a number")
            return -1;
        }
        return 0;
    }
    return 1;
}
}

#endif
