#include "teca_profiler.h"
#include "teca_memory_profiler.h"
#include "teca_common.h"

#include <sys/time.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdint.h>
#include <strings.h>
#include <cstdlib>
#include <cstdlib>
#include <cstdio>

#include <map>
#include <list>
#include <vector>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <thread>
#include <mutex>

namespace impl
{
#if defined(TECA_ENABLE_PROFILER)
// return high res system time relative to system epoch
static double get_system_time()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec/1.0e6;
}
#endif

// container for data captured in a timing event
struct event
{
    event();

    // serializes the event in CSV format into the stream.
    void to_stream(std::ostream &str) const;

    enum { START=0, END=1, DELTA=2 }; // record fields

    // user provided identifier for the record
    std::string name;

    // event duration, initially start time, end time and duration
    // are recorded. when summarizing this contains min,max and sum
    // of the summariezed set of events
    double time[3];

    // the thread id that generated the event
    std::thread::id tid;
};

#if defined(TECA_ENABLE_PROFILER)
// timer controls and data
static MPI_Comm comm = MPI_COMM_NULL;

//  TODO -- enable/disbal;e by thread
static bool logging_enabled = false;

static std::string timer_log_file = "timer.csv";

using event_log_type = std::unordered_map<std::thread::id, std::list<event>>;
static event_log_type event_log;
static std::mutex event_log_mutex;

// memory profiler
static teca_memory_profiler mem_prof;
#endif

// --------------------------------------------------------------------------
event::event() : time{0,0,0}, tid(std::this_thread::get_id())
{
}

//-----------------------------------------------------------------------------
void event::to_stream(std::ostream &str) const
{
#if defined(TECA_ENABLE_PROFILER)
    int ok = 0;
    MPI_Initialized(&ok);

    int rank = 0;
    if (ok)
        MPI_Comm_rank(impl::comm, &rank);

    str << rank << ", " << this->tid << ", \"" << this->name << "\", "
        << this->time[START] << ", " << this->time[END] << ", "
        << this->time[DELTA] << std::endl;
#else
    (void)str;
#endif
}
}

// ----------------------------------------------------------------------------
void teca_profiler::set_communicator(MPI_Comm comm)
{
#if defined(TECA_ENABLE_PROFILER)
    int ok = 0;
    MPI_Initialized(&ok);
    if (ok)
    {
        if (impl::comm != MPI_COMM_NULL)
            MPI_Comm_free(&impl::comm);

        MPI_Comm_dup(comm, &impl::comm);
    }
#else
    (void)comm;
#endif
}

// ----------------------------------------------------------------------------
void teca_profiler::set_timer_log_file(const std::string &file)
{
#if defined(TECA_ENABLE_PROFILER)
    impl::timer_log_file = file;
#else
    (void)file;
#endif
}

// ----------------------------------------------------------------------------
void teca_profiler::set_mem_prof_log_file(const std::string &file)
{
#if defined(TECA_ENABLE_PROFILER)
    impl::mem_prof.set_filename(file);
#else
    (void)file;
#endif
}

// ----------------------------------------------------------------------------
void teca_profiler::set_mem_prof_interval(int interval)
{
#if defined(TECA_ENABLE_PROFILER)
    impl::mem_prof.set_interval(interval);
#else
    (void)interval;
#endif
}

// ----------------------------------------------------------------------------
int teca_profiler::initialize()
{
#if defined(TECA_ENABLE_PROFILER)
    int ok = 0;
    MPI_Initialized(&ok);
    if (ok)
    {
        // always use isolated comm space
        if (impl::comm == MPI_COMM_NULL)
            teca_profiler::set_communicator(MPI_COMM_WORLD);

        impl::mem_prof.set_communicator(impl::comm);
    }

    // look for overrides in the environment
    char *tmp = nullptr;
    if ((tmp = getenv("PROFILER_ENABLE")))
        impl::logging_enabled = atoi(tmp);

    if ((tmp = getenv("PROFILER_LOG_FILE")))
        impl::timer_log_file = tmp;

    if ((tmp = getenv("MEMPROF_LOG_FILE")))
        impl::mem_prof.set_filename(tmp);

    if ((tmp = getenv("MEMPROF_INTERVAL")))
        impl::mem_prof.set_interval(atof(tmp));

    if (impl::logging_enabled)
    {
        teca_profiler::start_event("profile_lifetime");
        impl::mem_prof.initialize();
    }

    // report what options are in use
    int rank = 0;
    if (ok)
       MPI_Comm_rank(impl::comm, &rank);

    if ((rank == 0) && impl::logging_enabled)
        std::cerr << "Profiler configured with logging "
            << (impl::logging_enabled ? "enabled" : "disabled")
            << " timer log file \"" << impl::timer_log_file
            << "\", memory profiler log file \"" << impl::mem_prof.get_filename()
            << "\", sampling interval " << impl::mem_prof.get_interval()
            << " seconds" << std::endl;
#endif
    return 0;
}

// ----------------------------------------------------------------------------
int teca_profiler::finalize()
{
#if defined(TECA_ENABLE_PROFILER)
    int ok = 0;
    MPI_Initialized(&ok);

    if (impl::logging_enabled)
    {
        teca_profiler::end_event("profile_lifetime");

        // output timer log
        int rank = 0;
        int n_ranks = 1;

        if (ok)
        {
            MPI_Comm_rank(impl::comm, &rank);
            MPI_Comm_size(impl::comm, &n_ranks);
        }

        // serialize the logged events in CSV format
        std::ostringstream oss;
        oss.precision(std::numeric_limits<double>::digits10 + 2);
        oss.setf(std::ios::scientific, std::ios::floatfield);

        if (rank == 0)
            oss << "# rank, thread, name, start time, end time, delta" << std::endl;

        // not locking this as it's intended to be accessed only from the main
        // thread, and all other threads are required to be finished by now
        impl::event_log_type::iterator tliter = impl::event_log.begin();
        impl::event_log_type::iterator tlend = impl::event_log.end();

        for (; tliter != tlend; ++tliter)
        {
            std::list<impl::event>::iterator iter = tliter->second.begin();
            std::list<impl::event>::iterator end = tliter->second.end();

            for (; iter != end; ++iter)
                iter->to_stream(oss);
        }

        // free up resources
        impl::event_log.clear();

        if (ok)
        {
            // compute the file offset
            long n_bytes = oss.str().size();
            std::vector<long> gsizes(n_ranks);
            gsizes[rank] = n_bytes;

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                gsizes.data(), 1, MPI_LONG, impl::comm);

            long offset = 0;
            for (int i = 0; i < rank; ++i)
                offset += gsizes[i];

            long file_size = 0;
            for (int i = 0; i < n_ranks; ++i)
                file_size += gsizes[i];

            // write the buffer
            MPI_File fh;
            MPI_File_open(impl::comm, impl::timer_log_file.c_str(),
                MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

            MPI_File_set_view(fh, offset, MPI_BYTE, MPI_BYTE,
                "native", MPI_INFO_NULL);

            MPI_File_write(fh, oss.str().c_str(), n_bytes,
                MPI_BYTE, MPI_STATUS_IGNORE);

            MPI_File_set_size(fh, file_size);

            MPI_File_close(&fh);
        }
        else
        {
            FILE *fh = fopen(impl::timer_log_file.c_str(), "w");
            if (!fh)
            {
                const char *estr = strerror(errno);
                TECA_ERROR("Failed to open \""
                    << impl::timer_log_file << "\" " << estr)
                return -1;
            }

            long n_bytes = oss.str().size();
            long nwritten = fwrite(oss.str().c_str(), 1, n_bytes, fh);
            if (nwritten != n_bytes)
            {
                const char *estr = strerror(errno);
                TECA_ERROR("Failed to write " << n_bytes << " bytes. " << estr)
                return -1;
            }

            fclose(fh);
        }

        // output the memory use profile and clean up resources
        impl::mem_prof.finalize();
    }

    // free up other resources
    if (ok)
        MPI_Comm_free(&impl::comm);
#endif
    return 0;
}

//-----------------------------------------------------------------------------
bool teca_profiler::enabled()
{
#if defined(TECA_ENABLE_PROFILER)
    std::lock_guard<std::mutex> lock(impl::event_log_mutex);
    return impl::logging_enabled;
#else
    return false;
#endif
}

//-----------------------------------------------------------------------------
void teca_profiler::enable()
{
#if defined(TECA_ENABLE_PROFILER)
    // TODO -- this should be per thread
    std::lock_guard<std::mutex> lock(impl::event_log_mutex);
    impl::logging_enabled = true;
#endif
}

//-----------------------------------------------------------------------------
void teca_profiler::disable()
{
#if defined(TECA_ENABLE_PROFILER)
    // TODO -- this should be per thread
    std::lock_guard<std::mutex> lock(impl::event_log_mutex);
    impl::logging_enabled = false;
#endif
}

//-----------------------------------------------------------------------------
int teca_profiler::start_event(const char* eventname)
{
#if defined(TECA_ENABLE_PROFILER)
    if (impl::logging_enabled)
    {
        impl::event evt;
        evt.name = eventname;
        evt.time[impl::event::START] = impl::get_system_time();

        {
        std::lock_guard<std::mutex> lock(impl::event_log_mutex);
        impl::event_log[evt.tid].push_back(evt);
        }
    }
#else
    (void)eventname;
#endif
    return 0;
}

//-----------------------------------------------------------------------------
int teca_profiler::end_event(const char* eventname)
{
#if defined(TECA_ENABLE_PROFILER)
    if (impl::logging_enabled)
    {
        // get this thread's event log
        std::thread::id tid = std::this_thread::get_id();

        impl::event_log_mutex.lock();
        impl::event_log_type::iterator iter = impl::event_log.find(tid);
        if (iter == impl::event_log.end())
        {
            impl::event_log_mutex.unlock();
            TECA_ERROR("failed to end event \"" << eventname
                << "\" thread  " << tid << " has no events")
            return -1;
        }

        impl::event &evt = iter->second.back();
        impl::event_log_mutex.unlock();

#ifdef NDEBUG
        (void)eventname;
#else
        if (strcmp(eventname, evt.name.c_str()) != 0)
        {
            TECA_ERROR("Mismatched start_event/end_event. Expecting: '"
                << evt.name.c_str() << "' Got: '" << eventname << "'")
            abort();
        }
#endif
        evt.time[impl::event::END] = impl::get_system_time();
        evt.time[impl::event::DELTA] = evt.time[impl::event::END] - evt.time[impl::event::START];
    }
#else
    (void)eventname;
#endif
    return 0;
}
