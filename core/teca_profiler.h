#ifndef teca_profiler_h
#define teca_profiler_h

#include "teca_config.h"
#include "teca_mpi.h"

#include <string>
#include <thread>
#include <ostream>


/// A class containing methods managing memory and time profiling.
/**
 * Each timed event logs rank, event name, start and end time, and
 * duration.
 */
class TECA_EXPORT teca_profiler
{
public:
    // Initialize logging from environment variables, and/or the timer
    // API below. This is a collective call with respect to the timer's
    // communicator.
    //
    // If found in the environment the following variable override the
    // the current settings
    //
    //     PROFILER_ENABLE       : bit mask turns on or off logging,
    //                             0x01 -- event profiling enabled
    //                             0x02 -- memory profiling enabled
    //     PROFILER_LOG_FILE     : path to write timer log to
    //     MEMPROF_LOG_FILE      : path to write memory profiler log to
    //     MEMPROF_INTERVAL      : number of seconds between memory recordings
    //
    static int initialize();

    // Finalize the log. this is where logs are written and cleanup occurs.
    // All processes in the communicator must call, and it must be called
    // prior to MPI_Finalize.
    static int finalize();

    // this can occur after MPI_Finalize. It should only be called by rank 0.
    // Any remaining events will be appended to the log file. This is necessary
    // to time MPI_Initialize/Finalize and log associated I/O.
    static int flush();

    // Sets the communicator for MPI calls. This must be called prior to
    // initialization.
    // default value: MPI_COMM_NULL
    static void set_communicator(MPI_Comm comm);

    // Sets the path to write the timer log to
    // overridden by PROFILER_LOG_FILE environment variable
    // default value; Timer.csv
    static void set_timer_log_file(const std::string &file_name);

    // Sets the path to write the timer log to
    // overridden by MEMPROF_LOG_FILE environment variable
    // default value: MemProfLog.csv
    static void set_mem_prof_log_file(const std::string &file_name);

    // Sets the number of seconds in between memory use recordings
    // overridden by MEMPROF_INTERVAL environment variable.
    static void set_mem_prof_interval(int interval);

    // Enable/Disable logging. Overridden by PROFILER_ENABLE environment
    // variable. In the default format a CSV file is generated capturing each
    // ranks timer events. default value: disabled
    static void enable(int arg = 0x03);
    static void disable();

    // return true if logging is enabled.
    static bool enabled();

    // @brief Log start of an event.
    //
    // This marks the beginning of a event that must be logged.  The @arg
    // eventname must match when calling end_event() to mark the end of the
    // event.
    static int start_event(const char *eventname);

    // @brief Log end of a log-able event.
    //
    // This marks the end of a event that must be logged.  The @arg eventname
    // must match when calling end_event() to mark the end of the event.
    static int end_event(const char *eventname);

    // write contents of the string to the file.
    static int write_c_stdio(const char *file_name, const char *mode,
       const std::string &str);

    // write contents of the string to the file in rank order
    // the file is truncated first or created
    static int write_mpi_io(MPI_Comm comm, const char *file_name,
        const std::string &str);

    // checks to see if all active events have been ended.
    // will report errors if not
    static int validate();

    // setnd the current contents of the log to the stream
    static int to_stream(std::ostream &os);
};

/// A helper class that times it's life.
/**
 * A timer event is created that starts at the object's construction and ends
 * at its destruction. The pointer to the event name must be valid throughout
 * the objects life.
 */
template <int buffer_size>
class TECA_EXPORT teca_time_event
{
public:
    // logs an event named
    // <class_name>::<method> port=<p>
    teca_time_event(const char *class_name,
        const char *method, int port) : eventname(buffer)
    {
        snprintf(buffer, buffer_size, "%s::%s port=%d",
            class_name, method, port);
        teca_profiler::start_event(eventname);
    }

    // logs an event named
    // <class_name>::<method>
    teca_time_event(const char *class_name,
        int n_threads, int n_reqs) : eventname(buffer)
    {
        snprintf(buffer, buffer_size,
            "%s thread_pool process n_threads=%d n_reqs=%d",
            class_name, n_threads, n_reqs);
        teca_profiler::start_event(eventname);
    }


    // logs an event named:
    // <class_name>::<method>
    teca_time_event(const char *class_name,
        const char *method) : eventname(buffer)
    {
        buffer[0] = '\0';
        strcat(buffer, class_name);
        strcat(buffer, method);
        teca_profiler::start_event(eventname);
    }

    // logs an event named:
    // <name>
    teca_time_event(const char *name) : eventname(name)
    { teca_profiler::start_event(name); }

    ~teca_time_event()
    { teca_profiler::end_event(this->eventname); }

private:
    char buffer[buffer_size];
    const char *eventname;
};

#if defined(TECA_ENABLE_PROFILER)
#define TECA_PROFILE_PIPELINE(_n, _alg, _meth, _port, _code)    \
{                                                               \
    teca_time_event<_n> event(_alg->get_class_name(),           \
         _meth, _port);                                         \
    _code                                                       \
}

#define TECA_PROFILE_METHOD(_n, _alg, _meth, _code)             \
{                                                               \
    teca_time_event<_n>                                         \
        event(_alg->get_class_name(), "::" _meth);              \
    _code                                                       \
}

#define TECA_PROFILE_THREAD_POOL(_n, _alg, _nt, _nr, _code)     \
{                                                               \
    teca_time_event<_n>                                         \
        event(_alg->get_class_name(), _nt, _nr);                \
    _code                                                       \
}
#else
#define TECA_PROFILE_PIPELINE(_n, _alg, _meth, _port, _code)    \
{                                                               \
    _code                                                       \
}

#define TECA_PROFILE_METHOD(_n, _alg, _meth, _code)             \
{                                                               \
    _code                                                       \
}

#define TECA_PROFILE_THREAD_POOL(_n, _alg, _nt, _nr, _code)     \
{                                                               \
    _code                                                       \
}
#endif
#endif
