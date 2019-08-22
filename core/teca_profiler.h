#ifndef teca_profiler_h
#define teca_profiler_h

#include "teca_config.h"
#include <string>
#include <mpi.h>

// A class containing methods managing memory and time profiling
// Each timed event logs rank, event name, start and end time, and
// duration.
class teca_profiler
{
public:
    // Initialize logging from environment variables, and/or the timer
    // API below. This is a collective call with respect to the timer's
    // communicator.
    //
    // If found in the environment the following variable override the
    // the current settings
    //
    //     PROFILER_ENABLE          : integer turns on or off logging
    //     PROFILER_LOG_FILE        : path to write timer log to
    //     MEMPROF_LOG_FILE      : path to write memory profiler log to
    //     MEMPROF_INTERVAL      : number of seconds between memory recordings
    //
    static int initialize();

    // Finalize the log. this is where logs are written and cleanup occurs.
    // All processes in the communicator must call, and it must be called
    // prior to MPI_Finalize.
    static int finalize();

    // Sets the communicator for MPI calls. This must be called prior to
    // initialization.
    // default value: MPI_COMM_NULL
    static void set_communicator(MPI_Comm comm);

    // Sets the path to write the timer log to
    // overriden by PROFILER_LOG_FILE environment variable
    // default value; Timer.csv
    static void set_timer_log_file(const std::string &file_name);

    // Sets the path to write the timer log to
    // overriden by MEMPROF_LOG_FILE environment variable
    // default value: MemProfLog.csv
    static void set_mem_prof_log_file(const std::string &file_name);

    // Sets the number of seconds in between memory use recordings
    // overriden by MEMPROF_INTERVAL environment variable.
    static void set_mem_prof_interval(int interval);

    // Enable/Disable logging. Overriden by PROFILER_ENABLE environment
    // variable. In the default format a CSV file is generated capturing each
    // ranks timer events. default value: disabled
    static void enable();
    static void disable();

    // return true if loggin is enabled.
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
};

// teca_time_event -- A helper class that times it's life.
// A timer event is created that starts at the object's construction and ends
// at its destruction. The pointer to the event name must be valid throughout
// the objects life.
template <int buffer_size>
class teca_time_event
{
public:
    // logs an event named
    // <class_name>::<method> port=<port>
    teca_time_event(const char *class_name,
        const char *method, int port) : eventname(buffer)
    {
        snprintf(buffer, buffer_size, "%s::%s port=%d",
            class_name, method, port);
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
#else
#define TECA_PROFILE_PIPELINE(_n, _alg, _meth, _port, _code)    \
{                                                               \
    _code                                                       \
}
#endif

#endif
