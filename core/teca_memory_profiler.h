#ifndef teca_memory_profiler_h
#define teca_memory_profiler_h

#include "teca_config.h"
#include "teca_mpi.h"
#include <string>

extern "C" void *profile(void *argp);

/// MemoryProfiler - A sampling memory use profiler.
/**
 * The class samples process memory usage at the specified interval
 * given in seconds. For each sample the time is acquired. Calling
 * Initialize starts profiling, and Finalize ends it. During
 * Finalization the buffers are written using MPI-I/O to the
 * file name provided.
 */
class TECA_EXPORT teca_memory_profiler
{
public:
    teca_memory_profiler();
    ~teca_memory_profiler();

    teca_memory_profiler(const teca_memory_profiler &) = delete;
    void operator=(const teca_memory_profiler &) = delete;

    // start and stop the profiler
    int initialize();
    int finalize();

    // Set the interval in seconds between querying
    // the processes memory use.
    void set_interval(double interval);
    double get_interval() const;

    // Set the communicator for parallel I/O
    void set_communicator(MPI_Comm comm);

    // Set the file name to write the data to
    void set_filename(const std::string &filename);
    const char *get_filename() const;

    friend void *profile(void *argp);

private:
    struct internals_type;
    internals_type *internals;
};

#endif
