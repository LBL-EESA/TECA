#include "teca_memory_profiler.h"
#include "teca_common.h"

#if defined(_WIN32)
#include <windows.h>
#include <errno.h>
#include <psapi.h>
#else
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <fenv.h>
#include <mach/host_info.h>
#include <mach/mach.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#endif

#if defined(__linux)
#include <fenv.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

#include <ctype.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#include <cstdio>


// *****************************************************************************
#if defined(__linux) || defined(__APPLE__)
int load_lines(FILE* file, std::vector<std::string>& lines)
{
    // Load each line in the given file into a the vector.
    int n_read = 0;
    const int buf_size = 1024;
    char buf[buf_size] = { '\0' };
    while (!feof(file) && !ferror(file))
    {
        errno = 0;
        if (fgets(buf, buf_size, file) == 0)
        {
            if (ferror(file) && (errno == EINTR))
            {
                clearerr(file);
            }
            continue;
        }
        char* p_buf = buf;
        while (*p_buf)
        {
            if (*p_buf == '\n')
                *p_buf = '\0';
            p_buf += 1;
        }
        lines.push_back(buf);
        ++n_read;
    }
    if (ferror(file))
        return 0;

    return n_read;
}

#if defined(__linux)
// *****************************************************************************
int load_lines(const char* file_name, std::vector<std::string>& lines)
{
    FILE* file = fopen(file_name, "r");
    if (file == 0)
        return 0;

    int n_read = load_lines(file, lines);
    fclose(file);
    return n_read;
}
#endif

// ****************************************************************************
template <typename T>
int name_value(std::vector<std::string>& lines, std::string name, T& value)
{
    size_t n_lines = lines.size();
    for (size_t i = 0; i < n_lines; ++i)
    {
        size_t at = lines[i].find(name);
        if (at == std::string::npos)
            continue;

        std::istringstream is(lines[i].substr(at + name.size()));
        is >> value;
        return 0;
    }
    return -1;
}
#endif

#if defined(__linux)
// ****************************************************************************
template <typename T>
int get_fields_from_file(const char* file_name, const char** field_names, T* values)
{
    std::vector<std::string> fields;
    if (!load_lines(file_name, fields))
        return -1;

    int i = 0;
    while (field_names[i] != NULL)
    {
        int ierr = name_value(fields, field_names[i], values[i]);
        if (ierr)
            return -(i + 2);

        i += 1;
    }
    return 0;
}

// ****************************************************************************
template <typename T>
int get_field_from_file(const char* file_name, const char* field_name, T& value)
{
    const char* field_names[2] = { field_name, NULL };
    T values[1] = { T(0) };
    int ierr = get_fields_from_file(file_name, field_names, values);
    if (ierr)
        return ierr;

    value = values[0];
    return 0;
}
#endif

// ****************************************************************************
#if defined(__APPLE__)
template <typename T>
int get_fields_from_command(const char* command, const char** field_names,
                                                 T* values)
{
    FILE* file = popen(command, "r");
    if (file == 0)
        return -1;

    std::vector<std::string> fields;
    int nl = load_lines(file, fields);
    pclose(file);
    if (nl == 0)
        return -1;

    int i = 0;
    while (field_names[i] != NULL)
    {
        int ierr = name_value(fields, field_names[i], values[i]);
        if (ierr)
            return -(i + 2);

        i += 1;
    }
    return 0;
}
#endif

// intrernal data used by the memory profiler
struct teca_memory_profiler::internals_type
{
    internals_type() : comm(MPI_COMM_WORLD), filename("mem_prof.csv"),
        interval(60.0), data_mutex(PTHREAD_MUTEX_INITIALIZER),
        total_virtual_memory(0), available_virtual_memory(0),
        total_physical_memory(0), available_physical_memory(0)
        {}

    // initialize member vars with data about ram available on this system
    int initialize_memory();
    int initialize_apple_memory();
    int initialize_windows_memory();
    int initialize_linux_memory();

    // Retrieve memory information in megabyte.
    // virtual address space is not very useful in HPC context,
    // if you start swapping you are not high performance and
    // on Cray's swap is disabled.
    long long get_total_virtual_memory();
    long long get_available_virtual_memory();

    // Retrieve memory information in megabyte.
    // This is ram installed in memory banks on the node.
    // note that a process may not have access to all of it
    // subject to things like rlimit's
    long long get_total_physical_memory();
    long long get_available_physical_memory();

    // Get total system RAM in units of KiB.
    long long get_host_memory_total();

    // Get total system RAM in units of KiB. This may differ from the
    // host total if a host-wide resource limit is applied.
    long long get_host_memory_available(const char *env_var_name);

    // Get RAM used by all processes in the host, in units of KiB.
    long long get_host_memory_used();

    // get the amount of ram available for use by the given process
    // Get total system RAM in units of KiB. This may differ from the
    // host total if a per-process resource limit is applied.
    long long get_proc_memory_available(
        const char* host_limit_env_var_name,
        const char* proc_limit_env_var_name);

    // Get system RAM used by the process associated with the given
    // process id in units of KiB.
    long long get_proc_memory_used();

    MPI_Comm comm;
    std::string filename;
    double interval;
    std::deque<long long> mem_use;
    std::deque<double> time_pt;
    pthread_t thread;
    pthread_mutex_t data_mutex;
    long long total_virtual_memory;
    long long available_virtual_memory;
    long long total_physical_memory;
    long long available_physical_memory;
};

// **************************************************************************
extern "C" void *profile(void *argp)
{
    teca_memory_profiler::internals_type *internals =
        reinterpret_cast<teca_memory_profiler::internals_type*>(argp);

    while (1)
    {
        // capture the current time and memory usage.
        struct timeval tv;
        gettimeofday(&tv, nullptr);

        double cur_time = tv.tv_sec + tv.tv_usec/1.0e6;
        long long cur_mem = internals->get_proc_memory_used();

        pthread_mutex_lock(&internals->data_mutex);

        // log time and mem use
        internals->time_pt.push_back(cur_time);
        internals->mem_use.push_back(cur_mem);

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
            TECA_ERROR("Error: nanosleep had an error \"" << estr << "\"")
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
#if defined(TECA_HAS_MPI)
    int n_ranks = 1;

    int ok = 0;
    MPI_Initialized(&ok);

    if (ok)
    {
        MPI_Comm_rank(this->internals->comm, &rank);
        MPI_Comm_size(this->internals->comm, &n_ranks);
    }
#endif

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

#if defined(TECA_HAS_MPI)
    if (ok)
    {
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
    }
    else
#endif
    {
        FILE *fh = fopen(this->internals->filename.c_str(), "w");
        if (!fh)
        {
            const char *estr = strerror(errno);
            TECA_ERROR("Failed to open \""
                << this->internals->filename << "\" " << estr)
            return -1;
        }

        long nwritten = fwrite(oss.str().c_str(), 1, n_bytes, fh);
        if (nwritten != n_bytes)
        {
            const char *estr = strerror(errno);
            TECA_ERROR("Failed to write " << n_bytes << " bytes. " << estr)
            return -1;
        }

        fclose(fh);
    }

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


/*
std::string system_information::get_memory_description(
    const char* host_limit_env_var_name, const char* proc_limit_env_var_name)
{
    std::ostringstream oss;
    oss << "Host Total: " << iostreamlong long(this->get_host_memory_total())
            << " KiB, Host Available: "
            << iostreamlong long(this->get_host_memory_available(host_limit_env_var_name))
            << " KiB, Process Available: "
            << iostreamlong long(this->get_proc_memory_available(host_limit_env_var_name,
                                                                                                             proc_limit_env_var_name))
            << " KiB";
    return oss.str();
}
*/

// --------------------------------------------------------------------------
int teca_memory_profiler::internals_type::initialize_memory()
{
#if defined(__APPLE__)
    return this->initialize_apple_memory();
#elif defined(_WIN32)
    return this->initialize_windows_memory();
#elif defined(__linux)
    return this->initialize_linux_memory();
#else
    return -1;
#endif
}

// --------------------------------------------------------------------------
long long
teca_memory_profiler::internals_type::get_host_memory_total()
{
#if defined(_WIN32)
#if defined(_MSC_VER) && _MSC_VER < 1300
    MEMORYSTATUS stat;
    stat.dw_length = sizeof(stat);
    global_memory_status(&stat);
    return stat.dw_total_phys / 1024;
#else
    MEMORYSTATUSEX statex;
    statex.dw_length = sizeof(statex);
    global_memory_status_ex(&statex);
    return statex.ull_total_phys / 1024;
#endif
#elif defined(__linux)
    long long mem_total = 0;
    int ierr = get_field_from_file("/proc/meminfo", "mem_total:", mem_total);
    if (ierr)
        return -1;
    return mem_total;
#elif defined(__APPLE__)
    uint64_t mem;
    size_t len = sizeof(mem);
    int ierr = sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    if (ierr)
        return -1;
    return mem / 1024;
#else
    return 0;
#endif
}

/**
*/
// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_host_memory_available(
    const char* host_limit_env_var_name)
{
    long long mem_total = this->get_host_memory_total();

    // the following mechanism is provided for systems that
    // apply resource limits across groups of processes.
    // this is of use on certain SMP systems (eg. SGI UV)
    // where the host has a large amount of ram but a given user's
    // access to it is severly restricted. The system will
    // apply a limit across a set of processes. Units are in KiB.
    if (host_limit_env_var_name)
    {
        const char* host_limit_env_var_value = getenv(host_limit_env_var_name);
        if (host_limit_env_var_value)
        {
            long long host_limit = atoll(host_limit_env_var_value);
            if (host_limit > 0)
                mem_total = std::min(host_limit, mem_total);

        }
    }

    return mem_total;
}

// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_proc_memory_available(
    const char* host_limit_env_var_name, const char* proc_limit_env_var_name)
{
    long long mem_avail = this->get_host_memory_available(host_limit_env_var_name);

    // the following mechanism is provide for systems where rlimits
    // are not employed. Units are in KiB.
    if (proc_limit_env_var_name)
    {
        const char* proc_limit_env_var_value = getenv(proc_limit_env_var_name);
        if (proc_limit_env_var_value)
        {
            long long proc_limit = atoll(proc_limit_env_var_value);
            if (proc_limit > 0)
                mem_avail = std::min(proc_limit, mem_avail);
        }
    }

#if defined(__linux)
    int ierr;
    struct rlimit rlim;
    ierr = getrlimit(RLIMIT_DATA, &rlim);
    if ((ierr == 0) && (rlim.rlim_cur != RLIM_INFINITY))
        mem_avail = std::min((long long)rlim.rlim_cur / 1024, mem_avail);

    ierr = getrlimit(RLIMIT_AS, &rlim);
    if ((ierr == 0) && (rlim.rlim_cur != RLIM_INFINITY))
        mem_avail = std::min((long long)rlim.rlim_cur / 1024, mem_avail);

#elif defined(__APPLE__)
    struct rlimit rlim;
    int ierr;
    ierr = getrlimit(RLIMIT_DATA, &rlim);
    if ((ierr == 0) && (rlim.rlim_cur != RLIM_INFINITY))
        mem_avail = std::min((long long)rlim.rlim_cur / 1024, mem_avail);

    ierr = getrlimit(RLIMIT_RSS, &rlim);
    if ((ierr == 0) && (rlim.rlim_cur != RLIM_INFINITY))
        mem_avail = std::min((long long)rlim.rlim_cur / 1024, mem_avail);
#endif

    return mem_avail;
}

// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_host_memory_used()
{
#if defined(_WIN32)
#if defined(_MSC_VER) && _MSC_VER < 1300
    MEMORYSTATUS stat;
    stat.dw_length = sizeof(stat);
    global_memory_status(&stat);
    return (stat.dw_total_phys - stat.dw_avail_phys) / 1024;
#else
    MEMORYSTATUSEX statex;
    statex.dw_length = sizeof(statex);
    global_memory_status_ex(&statex);
    return (statex.ull_total_phys - statex.ull_avail_phys) / 1024;
#endif
#elif defined(__linux)
    // First try to use mem_available, but it only works on newer kernels
    const char* names2[3] = { "mem_total:", "mem_available:", NULL };
    long long values2[2] = { 0LL };
    int ierr = get_fields_from_file("/proc/meminfo", names2, values2);
    if (ierr)
    {
        const char* names4[5] = { "mem_total:", "mem_free:",
            "Buffers:", "Cached:", NULL };

        long long values4[4] = { 0LL };

        ierr = get_fields_from_file("/proc/meminfo", names4, values4);
        if (ierr)
            return ierr;

        long long& mem_total = values4[0];
        long long& mem_free = values4[1];
        long long& mem_buffers = values4[2];
        long long& mem_cached = values4[3];
        return mem_total - mem_free - mem_buffers - mem_cached;
    }
    long long& mem_total = values2[0];
    long long& mem_avail = values2[1];
    return mem_total - mem_avail;
#elif defined(__APPLE__)
    long long psz = getpagesize();
    if (psz < 1)
        return -1;

    const char* names[3] = { "Pages wired down:", "Pages active:", NULL };
    long long values[2] = { 0LL };
    int ierr = get_fields_from_command("vm_stat", names, values);
    if (ierr)
        return -1;

    long long& vm_wired = values[0];
    long long& vm_active = values[1];
    return ((vm_active + vm_wired) * psz) / 1024;
#else
    return 0;
#endif
}

// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_proc_memory_used()
{
#if defined(_WIN32) && defined(KWSYS_SYS_HAS_PSAPI)
    long pid = get_current_process_id();
    HANDLE h_proc;
    h_proc = open_process(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, false, pid);
    if (h_proc == 0)
    {
        return -1;
    }
    PROCESS_MEMORY_COUNTERS pmc;
    int ok = get_process_memory_info(h_proc, &pmc, sizeof(pmc));
    close_handle(h_proc);
    if (!ok)
    {
        return -2;
    }
    return pmc.working_set_size / 1024;
#elif defined(__linux)
    long long mem_used = 0;
    int ierr = get_field_from_file("/proc/self/status", "VmRSS:", mem_used);
    if (ierr)
        return -1;
    return mem_used;
#elif defined(__APPLE__)
    long long mem_used = 0;
    pid_t pid = getpid();
    std::ostringstream oss;
    oss << "ps -o rss= -p " << pid;
    FILE* file = popen(oss.str().c_str(), "r");
    if (file == 0)
    {
        return -1;
    }
    oss.str("");
    while (!feof(file) && !ferror(file))
    {
        char buf[256] = { '\0' };
        errno = 0;
        long long n_read = fread(buf, 1, 256, file);

        if (ferror(file) && (errno == EINTR))
            clearerr(file);

        if (n_read)
            oss << buf;
    }
    int ierr = ferror(file);
    pclose(file);
    if (ierr)
        return -2;

    std::istringstream iss(oss.str());
    iss >> mem_used;

    return mem_used;
#else
    return 0;
#endif
}

// --------------------------------------------------------------------------
int teca_memory_profiler::internals_type::initialize_windows_memory()
{
#if defined(_WIN32)
#if defined(_MSC_VER) && _MSC_VER < 1300
    MEMORYSTATUS ms;
    unsigned long tv, tp, av, ap;
    ms.DwLength = sizeof(ms);
    GlobalMemoryStatus(&ms);
#define MEM_VAL(value) dw##value
#else
    MEMORYSTATUSEX ms;
    DWORDLONG tv, tp, av, ap;
    ms.DwLength = sizeof(ms);
    if (0 == GlobalMemoryStatusEx(&ms))
        return 0;

#define MEM_VAL(value) ull##value
#endif
    tv = ms.MEM_VAL(TotalPageFile);
    tp = ms.MEM_VAL(TotalPhys);
    av = ms.MEM_VAL(AvailPageFile);
    ap = ms.MEM_VAL(AvailPhys);
    this->total_virtual_memory = tv >> 10 >> 10;
    this->total_physical_memory = tp >> 10 >> 10;
    this->available_virtual_memory = av >> 10 >> 10;
    this->available_physical_memory = ap >> 10 >> 10;
    return 0;
#else
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_memory_profiler::internals_type::initialize_linux_memory()
{
#if defined(__linux)
    unsigned long tv = 0;
    unsigned long tp = 0;
    unsigned long av = 0;
    unsigned long ap = 0;

    char buffer[1024]; // for reading lines

    int linux_major = 0;
    int linux_minor = 0;

    // Find the Linux kernel version first
    struct utsname uname_info;
    int error_flag = uname(&uname_info);
    if (error_flag != 0)
    {
        TECA_ERROR("Problem calling uname(): " << strerror(errno))
        return -1;
    }

    if (strlen(uname_info.release) >= 3)
    {
        // release looks like "2.6.3-15mdk-i686-up-4GB"
        char major_char = uname_info.release[0];
        char minor_char = uname_info.release[2];

        if (isdigit(major_char))
        {
            linux_major = major_char - '0';
        }

        if (isdigit(minor_char))
        {
            linux_minor = minor_char - '0';
        }
    }

    FILE* fd = fopen("/proc/meminfo", "r");
    if (!fd)
    {
        TECA_ERROR("Problem opening /proc/meminfo")
        return -1;
    }

    if (linux_major >= 3 || ((linux_major >= 2) && (linux_minor >= 6)))
    {
        // new /proc/meminfo format since kernel 2.6.x
        // Rigorously, this test should check from the developping version 2.5.x
        // that introduced the new format...
        enum
        {
            m_mem_total,
            m_mem_free,
            m_buffers,
            m_cached,
            m_swap_total,
            m_swap_free
        };
        const char* format[6] = {
            "MemTotal:%lu kB", "MemFree:%lu kB", "Buffers:%lu kB",
            "Cached:%lu kB", "SwapTotal:%lu kB", "SwapFree:%lu kB" };
        bool have[6] = { false, false, false, false, false, false };
        unsigned long value[6];
        int count = 0;
        while (fgets(buffer, static_cast<int>(sizeof(buffer)), fd))
        {
            for (int i = 0; i < 6; ++i)
            {
                if (!have[i] && sscanf(buffer, format[i], &value[i]) == 1)
                {
                    have[i] = true;
                    ++count;
                }
            }
        }
        if (count == 6)
        {
            this->total_physical_memory = value[m_mem_total] / 1024;
            this->available_physical_memory =
                (value[m_mem_free] + value[m_buffers] + value[m_cached]) / 1024;
            this->total_virtual_memory = value[m_swap_total] / 1024;
            this->available_virtual_memory = value[m_swap_free] / 1024;
        }
        else
        {
            TECA_ERROR("Problem parsing /proc/meminfo")
            fclose(fd);
            return -1;
        }
    }
    else
    {
        // /proc/meminfo format for kernel older than 2.6.x
        unsigned long temp;
        unsigned long cached_mem;
        unsigned long buffers_mem;
        // Skip "total: used:..."
        char* r = fgets(buffer, static_cast<int>(sizeof(buffer)), fd);
        int status = 0;
        if (r == buffer)
        {
            status += fscanf(fd, "Mem: %lu %lu %lu %lu %lu %lu\n", &tp, &temp, &ap,
                                             &temp, &buffers_mem, &cached_mem);
        }
        if (status == 6)
        {
            status += fscanf(fd, "Swap: %lu %lu %lu\n", &tv, &temp, &av);
        }
        if (status == 9)
        {
            this->total_virtual_memory = tv >> 10 >> 10;
            this->total_physical_memory = tp >> 10 >> 10;
            this->available_virtual_memory = av >> 10 >> 10;
            this->available_physical_memory =
                (ap + buffers_mem + cached_mem) >> 10 >> 10;
        }
        else
        {
            TECA_ERROR("Problem parsing /proc/meminfo")
            fclose(fd);
            return -1;
        }
    }
    fclose(fd);

    return 0;
#else
    return -1;
#endif
}

// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_total_virtual_memory()
{
    return this->total_virtual_memory;
}

// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_available_virtual_memory()
{
    return this->available_virtual_memory;
}

// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_total_physical_memory()
{
    return this->total_physical_memory;
}

// --------------------------------------------------------------------------
long long teca_memory_profiler::internals_type::get_available_physical_memory()
{
    return this->available_physical_memory;
}

// --------------------------------------------------------------------------
int teca_memory_profiler::internals_type::initialize_apple_memory()
{
#if defined(__APPLE__)
    int ierr = 0;
    uint64_t value = 0;
    size_t len = sizeof(value);
    sysctlbyname("hw.memsize", &value, &len, NULL, 0);
    this->total_physical_memory = static_cast<long long>(value / 1048576);

    // Parse values for Mac
    this->available_physical_memory = 0;
    vm_statistics_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat,
        &count) == KERN_SUCCESS)
    {
        len = sizeof(value);
        ierr = sysctlbyname("hw.pagesize", &value, &len, NULL, 0);
        int64_t available_memory = vmstat.free_count * value;
        this->available_physical_memory =
            static_cast<long long>(available_memory / 1048576);
    }

#ifdef VM_SWAPUSAGE
    // Virtual memory.
    int mib[2] = { CTL_VM, VM_SWAPUSAGE };
    long long miblen = sizeof(mib) / sizeof(mib[0]);
    struct xsw_usage swap;
    len = sizeof(swap);
    ierr = sysctl(mib, miblen, &swap, &len, NULL, 0);
    if (ierr == 0)
    {
        this->available_virtual_memory =
            static_cast<long long>(swap.xsu_avail / 1048576);
        this->total_virtual_memory = static_cast<long long>(swap.xsu_total / 1048576);
    }
#else
    this->available_virtual_memory = 0;
    this->total_virtual_memory = 0;
#endif

    return 0;
#else
    return -1;
#endif
}
