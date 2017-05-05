#ifndef teca_thread_pool_h
#define teca_thread_pool_h

#include "teca_common.h"
#include "teca_algorithm_fwd.h"
#include "teca_threadsafe_queue.h"

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <algorithm>
#if defined(_GNU_SOURCE)
#include <pthread.h>
#include <sched.h>
#include <deque>
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif
#endif

namespace internal
{
int thread_parameters(int base_core_id, int n_req, bool local,
    bool bind, bool verbose, std::deque<int> &affinity);
}

template <typename task_t, typename data_t>
class teca_thread_pool;

template <typename task_t, typename data_t>
using p_teca_thread_pool = std::shared_ptr<teca_thread_pool<task_t, data_t>>;

// a class to manage a fixed size pool of threads that dispatch
// I/O work
template <typename task_t, typename data_t>
class teca_thread_pool
{
public:
    teca_thread_pool() = delete;

    // construct/destruct the thread pool.
    // arguments:
    //   n        number of threads to create for the pool. -1 will
    //            create 1 thread per physical CPU core. If local is false
    //            all MPI ranks running on the same node are taken into
    //            account, resulting in 1 thread per core node wide.
    //
    //   local    consider other MPI ranks on the node. This introduces
    //            MPI collective operations, so all ranks in comm world
    //            must call it.
    //
    //   bind     bind each thread to a specific core.
    //
    //   verbose  print a report of the thread to core bindings
    teca_thread_pool(int n, bool local, bool bind, bool verbose);
    ~teca_thread_pool() noexcept;

    // get rid of copy and asignment
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_thread_pool)

    // add a data request task to the queue, returns a future
    // from which the generated dataset can be accessed.
    void push_task(task_t &task);

    // wait for all of the requests to execute and transfer
    // datasets in the order that corresponding requests
    // were added to the queue.
    template <template <typename ... > class container_t, typename ... args>
    void wait_data(container_t<data_t, args ...> &data);

    // get the number of threads
    unsigned int size() const noexcept
    { return m_threads.size(); }

private:
    // create n threads for the pool
    void create_threads(int n_threads, bool local, bool bind, bool verbose);

private:
    std::atomic<bool> m_live;
    teca_threadsafe_queue<task_t> m_queue;

    std::vector<std::future<data_t>>
        m_futures;

    std::vector<std::thread> m_threads;
};

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
teca_thread_pool<task_t, data_t>::teca_thread_pool(int n, bool local,
    bool bind, bool verbose) : m_live(true)
{
    this->create_threads(n, local, bind, verbose);
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
void teca_thread_pool<task_t, data_t>::create_threads(int n, bool local,
    bool bind, bool verbose)
{
#if !defined(_GNU_SOURCE)
    (void)bind;
    (void)verbose;
    (void)local;
    if (n < 1)
    {
        TECA_WARNING("Cannot autmatically detect threading parameters "
            "on this platform. The default is 1 thread per process.")
        n = 1;
    }
    int n_threads = n;
#else
    int base_core_id = sched_getcpu();
    std::deque<int> core_ids;
    int n_threads = internal::thread_parameters
        (base_core_id, n, local, bind, verbose, core_ids);
#endif

    // allocate the threads
    for (int i = 0; i < n_threads; ++i)
    {
        m_threads.push_back(std::thread([this]()
        {
            // "main" for each thread in the pool
            while (m_live.load())
            {
                task_t task;
                if (m_queue.try_pop(task))
                    task();
                else
                    std::this_thread::yield();
            }
        }));
#if defined(_GNU_SOURCE)
        // bind each to a hyperthread
        if (bind)
        {
            int core_id = core_ids.front();
            core_ids.pop_front();

            cpu_set_t core_mask;
            CPU_ZERO(&core_mask);
            CPU_SET(core_id, &core_mask);

            if (pthread_setaffinity_np(m_threads[i].native_handle(),
                sizeof(cpu_set_t), &core_mask))
            {
                TECA_WARNING("Failed to set thread affinity.")
            }
        }
#endif
    }
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
teca_thread_pool<task_t, data_t>::~teca_thread_pool() noexcept
{
    m_live = false;
    std::for_each(m_threads.begin(), m_threads.end(),
        [](std::thread &t) { t.join(); });
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
void teca_thread_pool<task_t, data_t>::push_task(task_t &task)
{
    m_futures.push_back(task.get_future());
    m_queue.push(std::move(task));
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
template <template <typename ... > class container_t, typename ... args>
void teca_thread_pool<task_t, data_t>::wait_data(container_t<data_t, args ...> &data)
{
    // wait on all pending requests and gather the generated
    // datasets
    std::for_each(m_futures.begin(), m_futures.end(),
        [&data] (std::future<data_t> &f)
        {
            data.push_back(f.get());
        });
    m_futures.clear();
}

#endif
