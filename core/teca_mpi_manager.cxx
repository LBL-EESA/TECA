#include "teca_mpi_manager.h"
#include "teca_config.h"
#include "teca_common.h"
#include "teca_profiler.h"
#include "teca_system_util.h"

#include <cstdlib>
#include <cstring>

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using seconds_t =
    std::chrono::duration<double, std::chrono::seconds::period>;

// --------------------------------------------------------------------------
teca_mpi_manager::teca_mpi_manager(int &argc, char **&argv)
    : m_rank(0),  m_size(1)
{
    teca_profiler::enable(0x01);
    teca_profiler::start_event("total_run_time");
    teca_profiler::start_event("app_initialize");

#if defined(TECA_HAS_MPI)
    // let the user disable MPI_Init. This is primarilly to work around Cray's
    // practice of calling abort from MPI_Init on login nodes.
    bool init_mpi = true;
    teca_system_util::get_environment_variable("TECA_INITIALIZE_MPI", init_mpi);
    if (init_mpi)
    {
        int mpi_thread_required = MPI_THREAD_SERIALIZED;
        int mpi_thread_provided = 0;
        MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);
        if (mpi_thread_provided < mpi_thread_required)
        {
            TECA_ERROR("This MPI does not support thread serialized");
            abort();
        }
    }
    else
    {
        TECA_WARNING("TECA_INITIALIZE_MPI=FALSE MPI_Init was not called.")
    }
#endif

    teca_profiler::disable();
    teca_profiler::set_communicator(MPI_COMM_WORLD);
    teca_profiler::initialize();

#if defined(TECA_HAS_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
#else
    (void)argc;
    (void)argv;
#endif

    teca_profiler::end_event("app_initialize");
}

// --------------------------------------------------------------------------
teca_mpi_manager::~teca_mpi_manager()
{
    teca_profiler::start_event("app_finalize");
    teca_profiler::finalize();

#if defined(TECA_HAS_MPI)
    int ok = 0;
    MPI_Initialized(&ok);
    if (ok)
        MPI_Finalize();
#endif

    teca_profiler::end_event("app_finalize");
    teca_profiler::end_event("total_run_time");

    if (m_rank == 0)
        teca_profiler::flush();
}
