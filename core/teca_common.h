#ifndef teca_common_h
#define teca_common_h

#include "teca_config.h"
#include "teca_parallel_id.h"

#include <iostream>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <vector>

// the operator<< overloads have to be namespace std in order for
// boost to find them. they are needed for mutitoken program options
namespace std
{
/// send a vector to a stream
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    if (!vec.empty())
    {
        os << vec[0];
        size_t n = vec.size();
        for (size_t i = 1; i < n; ++i)
            os << ", " << vec[i];
    }
    return os;
}

/// send a vector of strings to a stream
std::ostream &operator<<(std::ostream &os, const std::vector<std::string> &vec);
}

#ifndef SWIG
/// send a fixed length c-array to the stream
template <typename num_t, int len,
    typename = typename std::enable_if<!std::is_same<num_t,char>::value,bool>::type>
std::ostream &operator<<(std::ostream &os, const num_t (& data)[len])
{
    os << data[0];
    for (int i = 1; i < len; ++i)
        os << ", " << data[i];
    return os;
}
#endif

/** Return true if we are writing to a TTY. If we are not then we should not
 * use ansi color codes.
 */
int have_tty();

#define ANSI_RED "\033[1;31;40m"
#define ANSI_GREEN "\033[1;32;40m"
#define ANSI_YELLOW "\033[1;33;40m"
#define ANSI_WHITE "\033[1;37;40m"
#define ANSI_OFF "\033[0m"

#define BEGIN_HL(_color) (have_tty()?_color:"")
#define END_HL (have_tty()?ANSI_OFF:"")

#define TECA_MESSAGE(_strm, _head, _head_color, _msg)                   \
_strm                                                                   \
    << BEGIN_HL(_head_color) << _head << END_HL                         \
    << " " << teca_parallel_id() << " [" << __FILE__ << ":" << __LINE__ \
    << " " << TECA_VERSION_DESCR << "]" << std::endl                    \
    << BEGIN_HL(_head_color) << _head << END_HL << " "                  \
    << BEGIN_HL(ANSI_WHITE) << "" _msg << END_HL << std::endl;

#define TECA_ERROR(_msg) TECA_MESSAGE(std::cerr, "ERROR:", ANSI_RED, _msg)
#define TECA_WARNING(_msg) TECA_MESSAGE(std::cerr, "WARNING:", ANSI_YELLOW, _msg)
#define TECA_STATUS(_msg) TECA_MESSAGE(std::cerr, "STATUS:", ANSI_GREEN, _msg)

#endif
