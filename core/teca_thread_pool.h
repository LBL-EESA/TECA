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
#if defined(_GNU_SOURCE)
#include <pthread.h>
#include <sched.h>
#endif

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
    // construct/destruct the thread pool. If the size of the
    // pool is unspecified the number of cores - 1 threads
    // will be used.
    teca_thread_pool(bool bind);
    teca_thread_pool(int n, bool bind);
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
    void create_threads(int n_threads, bool bind = true);

private:
    std::atomic<bool> m_live;
    teca_threadsafe_queue<task_t> m_queue;

    std::vector<std::future<data_t>>
        m_futures;

    std::vector<std::thread> m_threads;
    unsigned int m_number_of_cores;
};

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
teca_thread_pool<task_t, data_t>::teca_thread_pool(bool bind) :
    m_live(true), m_number_of_cores(std::thread::hardware_concurrency())
{
    this->create_threads(-1, bind);
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
teca_thread_pool<task_t, data_t>::teca_thread_pool(int n, bool bind)
    : m_live(true), m_number_of_cores(std::thread::hardware_concurrency())
{
    this->create_threads(n, bind);
}

// --------------------------------------------------------------------------
template <typename task_t, typename data_t>
void teca_thread_pool<task_t, data_t>::create_threads(int n, bool bind)
{
#if defined(_GNU_SOURCE)
    int my_core_id = sched_getcpu();
#else
    (void)bind;
#endif
    unsigned int n_threads = n < 1 ? m_number_of_cores-1 : n;
    for (unsigned int i = 0; i < n_threads; ++i)
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
        // binds to cores round robin, starting from the active core.
        // this is done to ensure when MPI is in play all threads don't
        // get tied to the same cores, and to keep as many threads on the
        // same die as we can.
        if (bind)
        {
            unsigned int core_id = (my_core_id + i) % m_number_of_cores;

            cpu_set_t core_mask;
            CPU_ZERO(&core_mask);
            CPU_SET(core_id, &core_mask);

            if (pthread_setaffinity_np(m_threads[i].native_handle(),
                sizeof(cpu_set_t), &core_mask))
            {
                TECA_ERROR("Failed to set thread affinity. ")
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
