#include "teca_thread_pool.h"

#if defined(_GNU_SOURCE)
#include <cstring>
#include <utility>
#include <cstdint>
#include <sstream>
#include <iomanip>
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif
#endif

namespace internal
{
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
int detect_cpu_topology(int &n_threads, int &n_threads_per_core)
{
    // TODO: to do this correctly we need to detect number of chips per
    // board, and if hyperthreading has been enabled. for more info:
    // https://software.intel.com/en-us/articles/intel-64-architecture-processor-topology-enumeration/
    // see specifically table A3.

    // defaults should cpuid fail on this platform. hyperthreads are
    // treated as cores. this will lead to poor performance but without
    // cpuid we can't distinguish physical cores from hyperthreads.
    n_threads = std::thread::hardware_concurrency();
    n_threads_per_core = 1;

    // check if topology leaf is supported on this processor.
    uint64_t ra = 0, rb = 0, rc = 0, rd = 0;
    if (internal::cpuid(0, 0, ra, rb, rc, rd) || (ra < 0xb))
        return -1;

    // this is all Intel specific, AMD uses a different leaf in cpuid
    // rax=0xb, rcx=i  get's the topology leaf level i
    uint64_t level = 0;
    do
    {
        internal::cpuid(0xb, level, ra, rb, rc, rd);
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

// **************************************************************************
int generate_report(bool local, int local_proc, int base_id,
    const std::deque<int> &afin)
{
#if !defined(TECA_HAS_MPI)
    (void)local;
#endif
    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
    if (!local)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    }
#endif

    // gather proc ids
    std::vector<int> local_procs;
    if (rank == 0)
    {
        local_procs.resize(n_ranks);
        local_procs[0] = local_proc;
#if defined(TECA_HAS_MPI)
        if (!local)
        {
            MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_procs.data(),
                1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    else if (!local)
    {
        MPI_Gather(&local_proc, 1, MPI_INT, nullptr,
            0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
#endif
    }

    // gather base core ids
    std::vector<int> base_ids;
    if (rank == 0)
    {
        base_ids.resize(n_ranks);
        base_ids[0] = base_id;
#if defined(TECA_HAS_MPI)
        if (!local)
        {
            MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, base_ids.data(),
                1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    else if (!local)
    {
        MPI_Gather(&base_id, 1, MPI_INT, nullptr,
            0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
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
        if (!local)
        {
            MPI_Gather(MPI_IN_PLACE, 64, MPI_BYTE, hosts.data(),
                64, MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    }
    else if (!local)
    {
        char host[64];
        gethostname(host, 64);
        host[63] = '\0';
        MPI_Gather(host, 64, MPI_BYTE, nullptr,
            0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
#endif
    }

    // gather thread affinity map
    std::vector<int> recv_cnt;
    if (rank == 0)
    {
        recv_cnt.resize(n_ranks);
        recv_cnt[0] = afin.size();
#if defined(TECA_HAS_MPI)
        if (!local)
        {
            MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, recv_cnt.data(),
                1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    else if (!local)
    {
        int cnt = afin.size();
        MPI_Gather(&cnt, 1, MPI_INT, nullptr,
            0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
#endif
    }

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
#if defined(TECA_HAS_MPI)
        if (!local)
        {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, afins.data(),
                recv_cnt.data(), displ.data(), MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    else if (!local)
    {
        afins.assign(afin.begin(), afin.end());
        MPI_Gatherv(afins.data(), afins.size(), MPI_INT, nullptr,
            nullptr, nullptr, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
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
                oss << afins[displ[i]+j] << " ";
            }
            oss << (i<n_ranks-1 ? "\n" : "");
        }
        TECA_STATUS("threadpool afinity:" << std::endl << oss.str())
    }

    return 0;
}

// **************************************************************************
int thread_parameters(int base_core_id, int n_req, bool local,
    bool bind, bool verbose, std::deque<int> &affinity)
{
    std::vector<int> base_core_ids;

    int n_threads = n_req;

    // get the number of MPI ranks on this node, and their core id's
    int n_procs = 1;
    int proc_id = 0;

    if (local)
    {
        base_core_ids.push_back(base_core_id);
    }
    else
    {
#if defined(TECA_HAS_MPI)
        MPI_Comm comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
            0, MPI_INFO_NULL, &comm);

        MPI_Comm_size(comm, &n_procs);
        MPI_Comm_rank(comm, &proc_id);

        base_core_ids.resize(n_procs);
        base_core_ids[proc_id] = base_core_id;

        MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,
            base_core_ids.data(), 1, MPI_UNSIGNED, comm);

        MPI_Comm_free(&comm);
#else
        base_core_ids.push_back(base_core_id);
#endif
    }

    // get the number of cores on this cpu
    int threads_per_chip = 1;
    int hw_threads_per_core = 1;
    if (internal::detect_cpu_topology(threads_per_chip, hw_threads_per_core))
    {
        TECA_WARNING("failed to detect cpu topology. Assuming "
            << threads_per_chip/hw_threads_per_core << " physical cores.")
    }
    int threads_per_node = std::thread::hardware_concurrency();
    int cores_per_node = threads_per_node/hw_threads_per_core;

    // thread pool size is based on core and process count
    int nlg = 0;
    if (n_req > 0)
    {
        // user specified override
        n_threads = n_req;
    }
    else
    {
        // map threads to physical cores
        nlg = cores_per_node % n_procs;
        n_threads = cores_per_node/n_procs + (proc_id < nlg ? 1 : 0);
    }

    // stop now if we are not binding threads to cores
    if (!bind)
    {
        if (verbose)
            TECA_STATUS("thread to core binding disabled")

        return n_threads;
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
    // if we always did this in the fully apcked case we'd always be assigning
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
        int proc_n_threads = n_req > 0 ? n_req : cores_per_node/n_procs + (i < nlg ? 1 : 0);
        for (int j = 0; j < proc_n_threads; ++j)
        {
            // scan for empty core
            int q = select(cores_per_node, core_use, false,
                closest_core(proc_base, cores_per_node));

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
                int q = select(threads_per_node, thread_use, false,
                    closest_hyperthread(proc_base, cores_per_node));

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
        generate_report(local, proc_id, base_core_id, affinity);

    return n_threads;
}
#endif
}
