#include "teca_memory_profiler.h"

#if defined(TECA_HAS_VTK)
#include <vtksys/SystemInformation.hxx>
#endif

#include <vector>
#include <deque>
#include <sstream>
#include <sys/time.h>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>

namespace teca_timer
{
    struct teca_memory_profiler::internals_type
    {
        internals_type() : comm(MPI_COMM_WORLD), filename("MemProf.csv"),
            interval(60.0), data_mutex(PTHREAD_MUTEX_INITIALIZER) {}

        MPI_Comm comm;
        std::string filename;
        double interval;
        std::deque<long long> mem_use;
        std::deque<double> time_pt;
        pthread_t thread;
        pthread_mutex_t data_mutex;
        #if defined(TECA_HAS_VTK)
        vtksys::SystemInformation sys_info;
        #endif
    };

    extern "C" void *profile(void *argp)
    {
        teca_timer::teca_memory_profiler::internals_type *internals =
            reinterpret_cast<teca_timer::teca_memory_profiler::internals_type*>(argp);

        while (1)
        {
            // capture the current time and memory usage.
            struct timeval tv;
            gettimeofday(&tv, nullptr);

            double cur_time = tv.tv_sec + tv.tv_usec/1.0e6;

            #if defined(TECA_HAS_VTK)
            long long cur_mem = internals->sys_info.GetProcMemoryUsed();
            #endif

            pthread_mutex_lock(&internals->data_mutex);

            // log time and mem use
            internals->time_pt.push_back(cur_time);
            #if defined(TECA_HAS_VTK)
            internals->mem_use.push_back(cur_mem);
            #endif

            // get next interval
            double interval = internals->interval;

            pthread_mutex_unlock(&internals->data_mutex);

            // check for shut down code
            if (interval < 0)
                pthread_exit(nullptr);

            // suspend the thread for the requested interval
            long long secs = floor(interval);
            long nsecs = (interval - secs)*1e9;
            struct timespec sleep_time = {secs, nsecs};

            int ierr = 0;
            int tries = 0;
            while ((ierr = nanosleep(&sleep_time, &sleep_time)) && (errno == EINTR) && (++tries < 1000));
            if (ierr)
            {
                const char *estr = strerror(errno);
                std::cerr << "Error: nanosleep had an error \"" << estr << "\"" << std::endl;
                abort();
            }
        }

        return nullptr;
    }


    // --------------------------------------------------------------------------
    teca_memory_profiler::teca_memory_profiler()
    {
        this->internals = new internals_type;
    }

    // --------------------------------------------------------------------------
    teca_memory_profiler::~teca_memory_profiler()
    {
        delete this->internals;
    }

    // --------------------------------------------------------------------------
    int teca_memory_profiler::initialize()
    {
        if (pthread_create(&this->internals->thread,
            nullptr, profile, this->internals))
        {
            const char *estr = strerror(errno);
            std::cerr << "Error: Failed to create memory profiler. "
                << estr << std::endl;
            return -1;
        }

        return 0;
    }

    // --------------------------------------------------------------------------
    int teca_memory_profiler::finalize()
    {
        int rank = 0;
        int n_ranks = 1;

        MPI_Comm_rank(this->internals->comm, &rank);
        MPI_Comm_size(this->internals->comm, &n_ranks);

        pthread_mutex_lock(&this->internals->data_mutex);

        // tell the thread to quit
        this->internals->interval = -1;

        // create the ascii buffer
        // use ascii in the file as a convenince
        std::ostringstream oss;
        oss.precision(std::numeric_limits<double>::digits10 + 2);
        oss.setf(std::ios::scientific, std::ios::floatfield);

        if (rank == 0)
            oss << "# rank, time, memory kiB" << std::endl;

        long n_elem = this->internals->mem_use.size();
        for (long i = 0; i < n_elem; ++i)
        {
            oss << rank << ", " << this->internals->time_pt[i]
                << ", " << this->internals->mem_use[i] << std::endl;
        }

        // free resources
        this->internals->time_pt.clear();
        this->internals->mem_use.clear();

        pthread_mutex_unlock(&this->internals->data_mutex);

        // cancle the profiler thread
        pthread_cancel(this->internals->thread);

        // compute the file offset
        long n_bytes = oss.str().size();

        std::vector<long> gsizes(n_ranks);
        gsizes[rank] = n_bytes;

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            gsizes.data(), 1, MPI_LONG, this->internals->comm);

        long offset = 0;
        for (int i = 0; i < rank; ++i)
            offset += gsizes[i];

        long file_size = 0;
        for (int i = 0; i < n_ranks; ++i)
            file_size += gsizes[i];

        // write the buffer
        MPI_File fh;
        MPI_File_open(this->internals->comm, this->internals->filename.c_str(),
            MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

        MPI_File_set_view(fh, offset, MPI_BYTE, MPI_BYTE,
            "native", MPI_INFO_NULL);

        MPI_File_write(fh, oss.str().c_str(), n_bytes,
            MPI_BYTE, MPI_STATUS_IGNORE);

        MPI_File_set_size(fh, file_size);

        MPI_File_close(&fh);

        // wait for the proiler thread to finish
        pthread_join(this->internals->thread, nullptr);

        return 0;
    }

    // --------------------------------------------------------------------------
    double teca_memory_profiler::get_interval() const
    {
        return this->internals->interval;
    }

    // --------------------------------------------------------------------------
    void teca_memory_profiler::set_interval(double interval)
    {
        pthread_mutex_lock(&this->internals->data_mutex);
        this->internals->interval = interval;
        pthread_mutex_unlock(&this->internals->data_mutex);
    }

    // --------------------------------------------------------------------------
    void teca_memory_profiler::set_communicator(MPI_Comm comm)
    {
        this->internals->comm = comm;
    }

    // --------------------------------------------------------------------------
    void teca_memory_profiler::set_filename(const std::string &filename)
    {
        this->internals->filename = filename;
    }

    // --------------------------------------------------------------------------
    const char *teca_memory_profiler::get_filename() const
    {
        return this->internals->filename.c_str();
    }

}
