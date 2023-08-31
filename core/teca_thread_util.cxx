#include "teca_thread_util.h"
#include "teca_system_util.h"

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

#if defined(_GNU_SOURCE)
#include <utility>
#include <cstdint>
#include <functional>
#include <limits>
#endif
#include <iomanip>
#include <thread>
#include <cstring>
#include <vector>
#include <sstream>

namespace teca_thread_util
{
// **************************************************************************
int cpuid(uint64_t leaf, uint64_t level, uint64_t& ra, uint64_t& rb,
    uint64_t& rc, uint64_t& rd)
{
#if !defined(_WIN32)
    asm volatile("cpuid\n"
                 : "=a"(ra), "=b"(rb), "=c"(rc), "=d"(rd)
                 : "a"(leaf), "c"(level)
                 : "cc" );
    return 0;
#else
    return -1;
#endif
}

// **************************************************************************
void to_str(char *dest, uint64_t rx)
{
    char *crx = (char*)&rx;
    dest[0] = crx[0];
    dest[1] = crx[1];
    dest[2] = crx[2];
    dest[3] = crx[3];
}

// **************************************************************************
int detect_cpu_topology(int &n_threads, int &n_threads_per_core)
{
    // defaults should cpuid fail on this platform. hyperthreads are
    // treated as cores. this will lead to poor performance but without
    // cpuid we can't distinguish physical cores from hyperthreads.
    n_threads = std::thread::hardware_concurrency();
    n_threads_per_core = 1;

    // check if we have Intel or AMD
    uint64_t ra = 0, rb = 0, rc = 0, rd = 0;
    if (teca_thread_util::cpuid(0, 0, ra, rb, rc, rd))
        return -1;

    char vendor[13];
    to_str(vendor    , rb);
    to_str(vendor + 4, rd);
    to_str(vendor + 8, rc);
    vendor[12] = '\0';

    if (strcmp(vendor, "GenuineIntel") == 0)
    {
        // TODO: to do this correctly we need to detect number of chips per
        // board, and if hyperthreading has been enabled. for more info:
        // https://software.intel.com/en-us/articles/intel-64-architecture-processor-topology-enumeration/
        // see specifically table A3.

        // check if topology leaf is supported on this processor.
        if (teca_thread_util::cpuid(0, 0, ra, rb, rc, rd) || (ra < 0xb))
            return -1;

        // this is all Intel specific, AMD uses a different leaf in cpuid
        // rax=0xb, rcx=i  get's the topology leaf level i
        uint64_t level = 0;
        do
        {
            teca_thread_util::cpuid(0xb, level, ra, rb, rc, rd);
            n_threads = ((rc&0x0ff00) == 0x200) ? (0xffff&rb) : n_threads;
            n_threads_per_core = ((rc&0x0ff00) == 0x100) ? (0xffff&rb) : n_threads_per_core;
            level += 1;
        }
        while ((0xff00&rc) && (level < 16));

        // this should ever occur on intel cpu.
        if (level == 16)
            return -1;

        return 0;
    }
    else if (strcmp(vendor, "AuthenticAMD") == 0)
    {
        // hyperthreading in bit 28 of edx
        ra = 0, rb = 0, rc = 0, rd = 0;
        cpuid(0x01, 0x0, ra, rb, rc, rd);

        if (rd & 0x010000000)
            n_threads_per_core = 2;

        // core count in byte 0 of ecx
        ra = 0, rb = 0, rc = 0, rd = 0;
        cpuid(0x80000008, 0x0, ra, rb, rc, rd);

        n_threads = (rc & 0x0ff) + 1;

        return 0;
    }

    return -1;
}

#if defined(_GNU_SOURCE)
struct closest_core
{
    closest_core(int base_id, int n_cores)
        : m_base_id(base_id%n_cores) {}

    int operator()(int qq) const
    { return std::abs(m_base_id-qq); }

    int m_base_id;
};

struct closest_hyperthread
{
    closest_hyperthread(int base_id, int n_cores)
        : m_base_id(base_id%n_cores), m_n_cores(n_cores) {}

    int operator()(int qq) const
    { return std::abs(m_base_id-(qq%m_n_cores)); }

    int m_base_id;
    int m_n_cores;
};

struct least_used_hyperthread
{
    least_used_hyperthread(int *hyperthreads) :
        m_hyperthreads(hyperthreads) {}

    int operator()(int qq) const
    { return m_hyperthreads[qq]; }

    int *m_hyperthreads;
};

// **************************************************************************
int select(int n_slots, int *slots, bool any_slot,
    const std::function<int(int)> &dist_to)
{
    // scan for empy core, compute the distance, select the closest
    int q = std::numeric_limits<int>::max();
    int d = std::numeric_limits<int>::max();
    for (int qq = 0; qq < n_slots; ++qq)
    {
        if (any_slot || !slots[qq])
        {
            // this core is empty, prefer the closest
            int dd = dist_to(qq);
            if (dd <= d)
            {
                d = dd;
                q = qq;
            }
        }

    }
    return q;
}
#endif

// **************************************************************************
int generate_report(MPI_Comm comm, int local_proc, int base_id,
    const std::deque<int> &afin, const std::vector<int> &dev)
{
#if !defined(TECA_HAS_MPI)
    (void)comm;
#endif
    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &n_ranks);
    }
#endif

    // gather proc ids
    std::vector<int> local_procs;
    if (rank == 0)
    {
        local_procs.resize(n_ranks);
        local_procs[0] = local_proc;
#if defined(TECA_HAS_MPI)
    if (is_init)
        MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            local_procs.data(), 1, MPI_INT, 0, comm);
    }
    else
    {
        MPI_Gather(&local_proc, 1, MPI_INT, nullptr,
            0, MPI_DATATYPE_NULL, 0, comm);
#endif
    }

    // gather base core ids
    std::vector<int> base_ids;
    if (rank == 0)
    {
        base_ids.resize(n_ranks);
        base_ids[0] = base_id;
#if defined(TECA_HAS_MPI)
        if (is_init)
            MPI_Gather(MPI_IN_PLACE, 1, MPI_INT,
                base_ids.data(), 1, MPI_INT, 0, comm);
    }
    else
    {
        MPI_Gather(&base_id, 1, MPI_INT, nullptr,
            0, MPI_DATATYPE_NULL, 0, comm);
#endif
    }

    // gather host names
    std::vector<char> hosts;
    if (rank == 0)
    {
        hosts.resize(64*n_ranks);
        gethostname(hosts.data(), 64);
        hosts[63] = '\0';
#if defined(TECA_HAS_MPI)
        if (is_init)
            MPI_Gather(MPI_IN_PLACE, 64, MPI_BYTE, hosts.data(),
                64, MPI_BYTE, 0, comm);
    }
    else
    {
        char host[64];
        gethostname(host, 64);
        host[63] = '\0';
        MPI_Gather(host, 64, MPI_BYTE, nullptr,
            0, MPI_DATATYPE_NULL, 0, comm);
#endif
    }

    // gather thread affinity map
    std::vector<int> recv_cnt;
    if (rank == 0)
    {
        recv_cnt.resize(n_ranks);
        recv_cnt[0] = afin.size();
#if defined(TECA_HAS_MPI)
        if (is_init)
            MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, recv_cnt.data(),
                1, MPI_INT, 0, comm);
    }
    else
    {
        int cnt = afin.size();
        MPI_Gather(&cnt, 1, MPI_INT, nullptr,
            0, MPI_DATATYPE_NULL, 0, comm);
#endif
    }

    std::vector<int> devs;
    std::vector<int> afins;
    std::vector<int> displ;
    if (rank == 0)
    {
        int accum = 0;
        displ.resize(n_ranks);
        for (int i = 0; i < n_ranks; ++i)
        {
            displ[i] = accum;
            accum += recv_cnt[i];
        }

        afins.resize(accum);
        for (int i = 0; i < recv_cnt[0]; ++i)
            afins[i] = afin[i];

        devs.resize(accum);
        for (int i = 0; i < recv_cnt[0]; ++i)
            devs[i] = dev[i];

#if defined(TECA_HAS_MPI)
        if (is_init)
        {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, afins.data(),
                recv_cnt.data(), displ.data(), MPI_INT, 0, comm);

            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, devs.data(),
                recv_cnt.data(), displ.data(), MPI_INT, 0, comm);
        }
    }
    else
    {
        afins.assign(afin.begin(), afin.end());
        MPI_Gatherv(afins.data(), afins.size(), MPI_INT, nullptr,
            nullptr, nullptr, MPI_DATATYPE_NULL, 0, comm);

        MPI_Gatherv(dev.data(), dev.size(), MPI_INT, nullptr,
            nullptr, nullptr, MPI_DATATYPE_NULL, 0, comm);
#endif
    }

    if (rank == 0)
    {
        std::ostringstream oss;
        for (int i = 0; i < n_ranks; ++i)
        {
            oss << std::setw(4) << std::right << i << " : " << &hosts[i*64] << " : "
                << std::setw(3) << std::right << local_procs[i] << "."
                << std::setw(3) << std::left << base_ids[i] << " : ";

            for (int j = 0; j < recv_cnt[i]; ++j)
            {
                int q = displ[i]+j;
                oss << afins[q];
                if (devs[q] >= 0)
                    oss << "(" << devs[q] << ")";
                oss << " ";
            }
            oss << (i<n_ranks-1 ? "\n" : "");
        }
        TECA_STATUS("threadpool afinity:" << std::endl << oss.str())
    }

    return 0;
}

// **************************************************************************
int thread_parameters(MPI_Comm comm, int base_core_id, int n_requested,
    int threads_per_device, int ranks_per_device, bool bind, bool verbose,
    int &n_threads, std::deque<int> &affinity, std::vector<int> &device_ids)
{
#if !defined(TECA_HAS_CUDA)
    (void) threads_per_device;
    (void) ranks_per_device;
#endif

    // this rank is excluded from computations
    if (comm == MPI_COMM_NULL)
        return 0;

#if defined(TECA_HAS_MPI)
    int is_init = 0;
    int rank = 0;
    int n_ranks = 1;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &n_ranks);
    }
#endif
    // get the number of cores on this cpu
    int threads_per_chip = 1;
    int hw_threads_per_core = 1;
    if (teca_thread_util::detect_cpu_topology(threads_per_chip, hw_threads_per_core))
    {
        TECA_WARNING("failed to detect cpu topology. Assuming "
            << threads_per_chip/hw_threads_per_core << " physical cores.")
    }
    int threads_per_node = std::thread::hardware_concurrency();
    int cores_per_node = threads_per_node / hw_threads_per_core;

    // initialize to the user provided value. This will be used if
    // the functions needed to set affinity are not present. In that
    // case we set n_threads to 1 and report the failure.
    n_threads = n_requested;

#if defined(TECA_HAS_CUDA)
    // check for an override to the default number of MPI ranks per device
    int ranks_per_device_set = teca_system_util::get_environment_variable
        ("TECA_RANKS_PER_DEVICE", ranks_per_device);

    // determine the available CUDA GPUs
    std::vector<int> cuda_devices;
    if (teca_cuda_util::get_local_cuda_devices(comm,
        ranks_per_device, cuda_devices))
    {
        TECA_WARNING("Failed to determine the local CUDA devices."
            " Falling back to the default device.")
        cuda_devices.resize(1, 0);
    }

    if ((ranks_per_device_set == 0) && verbose && (rank == 0))
    {
        TECA_STATUS("TECA_RANKS_PER_DEVICE = " << ranks_per_device)
    }

    int n_cuda_devices = cuda_devices.size();

    // determine the number of threads to service each CUDA GPU
    int default_per_device = 8;

    if (!teca_system_util::get_environment_variable
        ("TECA_THREADS_PER_DEVICE", threads_per_device) &&
        verbose && (rank == 0))
    {
        TECA_STATUS("TECA_THREADS_PER_DEVICE = " << threads_per_device)
    }

    int n_device_threads = n_cuda_devices *
        (threads_per_device < 0 ? default_per_device : threads_per_device);

#endif

#if !defined(_GNU_SOURCE)
    // functions we need to set thread affinity are not available on this
    // platform. Set 1 thread per rank and if the caller asked us to return the
    // error.
    (void)bind;
    (void)verbose;
    (void)comm;
    (void)base_core_id;
    (void)affinity;

    int ret = 0;

    if (n_requested < 1)
    {
#if defined(TECA_HAS_MPI)
        if (is_init && (n_ranks > 1))
        {
            ret = -1;
            n_threads = 1;
            if (rank == 0)
            {
                TECA_WARNING("Can not automatically detect thread affinity"
                    " parameters on this platform. The default is 1 thread"
                    " per MPI rank.")
            }
        }
        else
        {
#endif
            // MPI is not in use, assume we have exclusive access to all CPU
            // cores
            n_threads = cores_per_node;
            affinity.resize(cores_per_node);
            for (int i = 0; i < n_threads; ++i)
            {
                affinity[i] = i;
            }
#if defined(TECA_HAS_MPI)
        }
#endif
    }

#if defined(TECA_HAS_CUDA)
    // assign CUDA device or a CPU core to each thread
    if ((verbose > 1) && (n_threads < n_cuda_devices))
    {
        TECA_WARNING(<< n_threads
            << " threads are insufficient to service " << n_cuda_devices
            << " CUDA devices. " << n_cuda_devices - n_threads
            << " CUDA devices will not be utilized.")
    }

    device_ids.resize(n_threads);
    for (int i = 0; i < n_threads; ++i)
    {
        // select the CUDA device [0, n_cuda_devices) that this thread will
        // utilize. Once all devices are assigned a thread the remaining
        // threads will make use of CPU cores, specfied by a device_id of -1
        int device_id = -1;

        if (i < n_device_threads)
            device_id = cuda_devices[i % n_cuda_devices];

        device_ids[i] = device_id;
    }
#else
    device_ids.resize(n_threads, -1);
#endif
    if (!ret && verbose)
        generate_report(comm, 0, 0, affinity, device_ids);

    return ret;
#else
    // get the core that this rank's main thread is running on. typically MPI
    // ranks land on unqiue cores but if needed one can use the batch system to
    // explicitly bind ranks to unique cores
    if (base_core_id < 0)
        base_core_id = sched_getcpu();

    // get the number of MPI ranks on this node, and their core id's
    int n_procs = 1;
    int proc_id = 0;

    std::vector<int> base_core_ids;

#if defined(TECA_HAS_MPI)
    if (is_init)
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED,
            0, MPI_INFO_NULL, &node_comm);

        MPI_Comm_size(node_comm, &n_procs);
        MPI_Comm_rank(node_comm, &proc_id);

        base_core_ids.resize(n_procs);
        base_core_ids[proc_id] = base_core_id;

        MPI_Allgather(MPI_IN_PLACE,0, MPI_DATATYPE_NULL,
            base_core_ids.data(), 1, MPI_UNSIGNED, node_comm);

        MPI_Comm_free(&node_comm);
    }
    else
    {
        base_core_ids.push_back(base_core_id);
    }
#else
    base_core_ids.push_back(base_core_id);
#endif

    // thread pool size is based on core and process count
    int nlg = 0;
    // map threads to physical cores
    nlg = cores_per_node % n_procs;
    n_threads = cores_per_node / n_procs + (proc_id < nlg ? 1 : 0);
    if (n_requested > 0)
    {
        // use exactly this many
        n_threads = n_requested;
    }
    else if (n_requested < -1)
    {
        // use at most this many
        n_threads = std::min(-n_requested, n_threads);
        n_requested = n_threads;
    }
    else
    {
    }

    // if the user runs more MPI ranks than cores some of the ranks
    // will have no cores to use.
    if (n_procs > cores_per_node)
    {
        TECA_WARNING(<< n_procs << " MPI ranks running on this node but only "
            << cores_per_node << " CPU cores are available. Performance will"
            " be degraded.")

        n_threads = 1;
        affinity.push_back(base_core_id);
#if defined(TECA_HAS_CUDA)
        device_ids.push_back(n_cuda_devices < 1 ? -1 : cuda_devices[0]);
#else
        device_ids.push_back(-1);
#endif
        return -1;
    }

    // assign each thread to a CUDA device or CPU core
#if defined(TECA_HAS_CUDA)
    if (verbose && (n_ranks < 2) && (n_threads < n_cuda_devices))
    {
        TECA_WARNING(<< n_threads
            << " threads is insufficient to service " << n_cuda_devices
            << " CUDA devices. " << n_cuda_devices - n_threads
            << " CUDA devices will not be utilized.")
    }

    device_ids.resize(n_threads);
    for (int i = 0; i < n_threads; ++i)
    {
        // select the CUDA device [0, n_cuda_devices) that thread i will
        // utilize. Once all devices are assigned a thread the remaining
        // threads will make use of CPU cores, specfied by setting cuda_device
        // to -1
        int device_id = -1;

        if (i < n_device_threads)
            device_id = cuda_devices[i % n_cuda_devices];

        device_ids[i] = device_id;
    }
#else
    device_ids.resize(n_threads, -1);
#endif

    // stop now if we are not binding threads to cores
    if (!bind)
    {
        if (verbose)
            TECA_STATUS("thread to core binding disabled. n_threads=" << n_threads)

        return 0;
    }

    // track which threads/cores are in use
    // using two maps, the first track use of cores, the second hyperthreads
    // the hyper thread map is 2d with core id on first dim and hyperthread id
    // on the second.
    long map_bytes = threads_per_node*sizeof(int);
    int *thread_use = static_cast<int*>(malloc(map_bytes));
    memset(thread_use, 0, map_bytes);

    map_bytes = cores_per_node*sizeof(int);
    int *core_use = static_cast<int*>(malloc(map_bytes));
    memset(core_use, 0, map_bytes);

    // there are enough cores that each thread can have it's own core
    // mark the cores which have the root thread as used so that we skip them.
    // if we always did this in the fully packed case we'd always be assigning
    // hyperthreads off core. it is better to keep them local.
    if (((n_threads+1)*n_procs) < cores_per_node)
    {
        for (int i = 0; i < n_procs; ++i)
        {
            int bcid = base_core_ids[i];
            core_use[bcid%cores_per_node] = 1;
            thread_use[bcid] = 1;
        }
    }

    // mark resources used by other processes, up to and including this process.
    // also record the core ids we will bind to.
    for (int i = 0; i <= proc_id; ++i)
    {
        int proc_base = base_core_ids[i];
        int proc_n_threads = n_requested > 0 ? n_requested : cores_per_node/n_procs + (i < nlg ? 1 : 0);
        for (int j = 0; j < proc_n_threads; ++j)
        {
            // scan for empty core
            int q = teca_thread_util::select(cores_per_node,
                core_use, false, closest_core(proc_base, cores_per_node));

            if (q < cores_per_node)
            {
                // found one, mark core as used
                core_use[q] = 1;

                // now find an empty hyperthread on this core and
                // mark it as taken. if this is for us record the id
                // for later use
                int p = 0;
                while (thread_use[q+p*cores_per_node] && (p < hw_threads_per_core)) ++p;

                int pp = q+p*cores_per_node;

                // mark hyperthread as used
                thread_use[pp] = 1;

                // store the core id we will bind one of our threads to it
                if (i == proc_id)
                    affinity.push_back(pp);
            }
            else
            {
                // if we are here it means all the cores have at least one
                // hyperthread assigned. find the first empty hyperthread on
                // any core. if that fails then find the least used hyperthread.
                // if this for us record the id for later use
                int q = teca_thread_util::select(threads_per_node,
                    thread_use, false, closest_hyperthread(proc_base,
                    cores_per_node));

                if (q >= threads_per_node)
                    q = select(threads_per_node, thread_use, true,
                        least_used_hyperthread(thread_use));

                if (q < threads_per_node)
                {
                    // found one. increment usage
                    thread_use[q] += 1;

                    // store the core id we will bind one of our threads to it
                    if (i == proc_id)
                        affinity.push_back(q);
                }
            }
        }
    }

    free(core_use);
    free(thread_use);

    if (verbose)
        generate_report(comm, proc_id, base_core_id, affinity, device_ids);
#endif
    return 0;
}

}
