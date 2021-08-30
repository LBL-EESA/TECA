#include "teca_common.h"

#include "teca_mpi.h"
#include <cstdlib>

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


namespace teca_error
{
// **************************************************************************
void error_message(const char *msg)
{
    // flush any pending user output
    std::cout.flush();
    std::cerr.flush();

    // send the error message
    std::cerr << std::endl << msg << std::endl;
}

// **************************************************************************
void error_message_abort(const char *msg)
{
    // flush any pending user output
    std::cout.flush();
    std::cerr.flush();

    // send the error message
    std::cerr << std::endl << msg << std::endl
        << "aborting ... " << std::endl;

    // abort
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif
    abort();
}

// **************************************************************************
void set_error_handler(p_teca_error_handler handler)
{
    teca_error::error_handler = handler;
}

// **************************************************************************
void set_error_message_handler()
{
    teca_error::error_handler = teca_error::error_message;
}

// **************************************************************************
void set_error_message_abort_handler()
{
    teca_error::error_handler = teca_error::error_message_abort;
}

// global error handler instance
p_teca_error_handler error_handler = error_message_abort;
};
