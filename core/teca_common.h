#ifndef teca_common_h
#define teca_common_h

/// @file

#include "teca_config.h"
#include "teca_parallel_id.h"

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <vector>
#include <array>

/** The call signature for the error handler. The error handler will be passed
 * a string describing the error.
 */
using p_teca_error_handler = void (*) (const char*);

/// global error handling hooks
namespace teca_error
{
/// The global error handler instance.
extern p_teca_error_handler error_handler TECA_EXPORT;

/** An error handler that flushes stdout and stderr streams, and sends msg to
 * the stderr before returing. This implements the behavior up to and including
 * TECA 4.1.0
 */
TECA_EXPORT
void error_message(const char *msg);

/** An error handler that flushes stdout and stderr streams, and sends msg to
 * the stderr before aborting. When MPI is in use MPI_Abort is invoked. This
 * implements the behavior after TECA 4.1.0
 */
TECA_EXPORT
void error_message_abort(const char *msg);

/** Install a custom error haandler. The error handler must have the following
 * signature.
 *
 * void error_handler(const char *msg);
 *
 */
TECA_EXPORT
void set_error_handler(p_teca_error_handler handler);

/// Install the teca_error::error_message error handler
TECA_EXPORT
void set_error_message_handler();

/// Install the teca_error::error_message_abort error handler
TECA_EXPORT
void set_error_message_abort_handler();
};

/// @cond

// the operator<< overloads have to be namespace std in order for
// boost to find them. they are needed for mutitoken program options
namespace std
{
/// send a vector to a stream
template <typename T>
TECA_EXPORT
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
TECA_EXPORT
std::ostream &operator<<(std::ostream &os, const std::vector<std::string> &vec);

/// send an array to a stream
template <typename T, size_t N>
TECA_EXPORT
std::ostream &operator<<(std::ostream &os, const std::array<T,N> &vec)
{
    if (N)
    {
        os << vec[0];
        for (size_t i = 1; i < N; ++i)
            os << ", " << vec[i];
    }
    return os;
}
}

#ifndef SWIG
/// send a fixed length c-array to the stream
template <typename num_t, int len,
    typename = typename std::enable_if<!std::is_same<num_t,char>::value,bool>::type>
TECA_EXPORT
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
TECA_EXPORT int have_tty();


#define ANSI_RED "\033[1;31;40m"
#define ANSI_GREEN "\033[1;32;40m"
#define ANSI_YELLOW "\033[1;33;40m"
#define ANSI_WHITE "\033[1;37;40m"
#define ANSI_OFF "\033[0m"

#define BEGIN_HL(_color) (have_tty()?_color:"")
#define END_HL (have_tty()?ANSI_OFF:"")

/// @endcond


/** Send a message into the stream with an ANSI color coded message that
 * include MPI ranks and thread id.
 */
#define TECA_MESSAGE(_strm, _head, _head_color, _msg)                   \
_strm                                                                   \
    << BEGIN_HL(_head_color) << _head << END_HL                         \
    << " " << teca_parallel_id() << " [" << __FILE__ << ":" << __LINE__ \
    << " " << TECA_VERSION_DESCR << "]" << std::endl                    \
    << BEGIN_HL(_head_color) << _head << END_HL << " "                  \
    << BEGIN_HL(ANSI_WHITE) << "" _msg << END_HL << std::endl;

/// Send a message into the stream that include MPI ranks and thread id.
#define TECA_MESSAGE_RAW(_strm, _head, _msg)                            \
_strm                                                                   \
    << _head << " " << teca_parallel_id() << " [" << __FILE__           \
    << ":" << __LINE__ << " " << TECA_VERSION_DESCR << "]" << std::endl \
    << _head << " " << "" _msg << std::endl;

/** Constructs an the error message using TECA_MESSAGE and invokes the
 * error handler.
 */
#define TECA_FATAL_ERROR(_msg)                                          \
{                                                                       \
    std::ostringstream ess;                                             \
    TECA_MESSAGE(ess, "ERROR:", ANSI_RED, _msg)                         \
    teca_error::error_handler(ess.str().c_str());                       \
}

/// Constructs an error message and sends it to the stderr stream
#define TECA_ERROR(_msg) TECA_MESSAGE(std::cerr, "ERROR:", ANSI_RED, _msg)

/// Constructs a warning message and sends it to the stderr stream
#define TECA_WARNING(_msg) TECA_MESSAGE(std::cerr, "WARNING:", ANSI_YELLOW, _msg)

/// Constructs a status message and sends it to the stderr stream
#define TECA_STATUS(_msg) TECA_MESSAGE(std::cerr, "STATUS:", ANSI_GREEN, _msg)

#endif
