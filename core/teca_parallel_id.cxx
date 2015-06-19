#include "teca_config.h"
#include "teca_parallel_id.h"

#include <ostream>
#include <sstream>
#include <thread>

using std::ostringstream;
using std::ostream;

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

ostream &operator<<(ostream &os, const teca_parallel_id &)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    ostringstream oss;
    oss << "[" << rank << ":" << std::this_thread::get_id() << "]";
    os << oss.str();
    return os;
}
