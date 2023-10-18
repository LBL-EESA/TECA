#include <iostream>
#include <string>
#include <cstdlib>
#include <thread>

__attribute__((constructor)) void init(void)
{
#if defined(TECA_DEBUG)
    std::cerr << "teca_core initializing ... " << std::endl;
#endif
/* TODO -- problems with MPI
 * with multiple MPI ranks per node, MPI OMP_PROC_BIND should be false, otherwise true.
 * we have no way here to know if MPI is in use and how many ranks per node.
 * these variables need to be initialized here to have any affect.
    if (!getenv("OMP_NUM_THREADS"))
    {
        int n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        setenv("OMP_NUM_THREADS", std::to_string(n_threads).c_str(), 1);
    }

    if (!getenv("OMP_PROC_BIND"))
        setenv("OMP_PROC_BIND", "true", 1);

    if (!getenv("OMP_PLACES"))
       setenv("OMP_PLACES", "cores", 1);
*/
#if defined(TECA_DEBUG)
    setenv("OMP_DISPLAY_ENV", "true", 1);
    setenv("OMP_DISPLAY_AFFINITY", "true", 1);
#endif
}

#if defined(TECA_DEBUG)
__attribute__((destructor))  void fini(void)
{
    std::cerr << "teca_core finalizing ... " << std::endl;
}
#endif
