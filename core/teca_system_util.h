#ifndef teca_system_util_h
#define teca_system_util_h


#include "teca_common.h"

#include <cstdlib>
#include <cstring>
#include <cerrno>

namespace teca_system_util
{

template <typename T>
struct string_tt {};

#define DECLARE_STR_CONVERSION_I(_CPP_T, _FUNC)                 \
template <>                                                     \
struct string_tt<_CPP_T>                                        \
{                                                               \
    static int convert(char *str, _CPP_T &val)                  \
    {                                                           \
        errno = 0;                                              \
        char *endp = nullptr;                                   \
        _CPP_T tmp = _FUNC(str, &endp, 0);                      \
        if (errno != 0)                                         \
        {                                                       \
            TECA_ERROR("Failed to convert string \""            \
                << str << "\" to a nunber." << strerror(errno)) \
            return  -1;                                         \
        }                                                       \
        else if (endp == str)                                   \
        {                                                       \
            TECA_ERROR("Failed to convert string \""            \
                << str << "\" to a nunber. Invalid string.")    \
            return  -1;                                         \
        }                                                       \
        val = tmp;                                              \
        return 0;                                               \
    }                                                           \
};

#define DECLARE_STR_CONVERSION_F(_CPP_T, _FUNC)                 \
template <>                                                     \
struct string_tt<_CPP_T>                                        \
{                                                               \
    static int convert(char *str, _CPP_T &val)                  \
    {                                                           \
        errno = 0;                                              \
        char *endp = nullptr;                                   \
        _CPP_T tmp = _FUNC(str, &endp);                         \
        if (errno != 0)                                         \
        {                                                       \
            TECA_ERROR("Failed to convert string \""            \
                << str << "\" to a nunber." << strerror(errno)) \
            return  -1;                                         \
        }                                                       \
        else if (endp == str)                                   \
        {                                                       \
            TECA_ERROR("Failed to convert string \""            \
                << str << "\" to a nunber. Invalid string.")    \
            return  -1;                                         \
        }                                                       \
        val = tmp;                                              \
        return 0;                                               \
    }                                                           \
};

DECLARE_STR_CONVERSION_F(float, strtof)
DECLARE_STR_CONVERSION_F(double, strtod)
DECLARE_STR_CONVERSION_I(char, strtol)
DECLARE_STR_CONVERSION_I(short, strtol)
DECLARE_STR_CONVERSION_I(int, strtol)
DECLARE_STR_CONVERSION_I(long, strtoll)
DECLARE_STR_CONVERSION_I(long long, strtoll)

// initialize val from an environment variable named by var. the valid values
// of the variable are case insensative: 0,1,true,false,on,off.
//
// returns:
//    0  if the variable was found and val was initialized from it
//    1  if the varibale was not found
//   -1  if the variable was found but conversion from string failed
int get_environment_variable(const char *var, bool &val);

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
        if (string_tt<T>::convert(tmp, val))
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
