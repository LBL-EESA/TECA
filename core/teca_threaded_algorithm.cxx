#include "teca_threaded_algorithm.h"
#include "teca_metadata.h"
#if defined(TECA_HAS_CUDA)
#include "teca_cuda_thread_pool.h"
#else
#include "teca_cpu_thread_pool.h"
#endif
#include "teca_profiler.h"

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <cstdlib>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif


// **************************************************************************
p_teca_data_request_queue new_teca_data_request_queue(MPI_Comm comm,
    int n_threads, int threads_per_device, int ranks_per_device, bool bind,
    bool verbose)
{
#if defined(TECA_HAS_CUDA)
    return std::make_shared<teca_data_request_queue>(comm,
        n_threads, threads_per_device, ranks_per_device, bind,
        verbose);
#else
    (void) threads_per_device;
    (void) ranks_per_device;
    return std::make_shared<teca_data_request_queue>(
        comm, n_threads, bind, verbose);
#endif
}

// function that executes the data request and returns the
// requested dataset
class teca_data_request
{
public:
    teca_data_request(const p_teca_algorithm &alg,
        const teca_algorithm_output_port up_port,
        const teca_metadata &up_req, int pass_dev) : m_alg(alg),
        m_up_port(up_port), m_up_req(up_req), m_pass_dev(pass_dev)
    {}

    const_p_teca_dataset operator()(int device_id = -1)
    {
        // use the device assigned downstream
        if (m_pass_dev)
            m_up_req.get("device_id", device_id);

        // set the device on which to execute this pipeline
        m_up_req.set("device_id", device_id);

        // execute the pipeline
        return m_alg->request_data(m_up_port, m_up_req);
    }

public:
    p_teca_algorithm m_alg;
    teca_algorithm_output_port m_up_port;
    teca_metadata m_up_req;
    int m_pass_dev;
};


// internals for teca threaded algorithm
class teca_threaded_algorithm_internals
{
public:
    teca_threaded_algorithm_internals() {}

    void thread_pool_resize(MPI_Comm comm, int n_threads,
        int threads_per_device, int ranks_per_device, bool bind,
        bool verbose);

    unsigned int get_thread_pool_size() const noexcept
    { return this->thread_pool ? this->thread_pool->size() : 0; }

public:
    p_teca_data_request_queue thread_pool;
};

// --------------------------------------------------------------------------
void teca_threaded_algorithm_internals::thread_pool_resize(MPI_Comm comm,
    int n_threads, int threads_per_device, int ranks_per_device, bool bind,
    bool verbose)
{
    this->thread_pool = new_teca_data_request_queue(comm,
        n_threads, threads_per_device, ranks_per_device, bind, verbose);
}



// --------------------------------------------------------------------------
teca_threaded_algorithm::teca_threaded_algorithm() :
    bind_threads(1), stream_size(-1), poll_interval(1000000),
    threads_per_device(-1), ranks_per_device(1), propagate_device_assignment(0),
    internals(new teca_threaded_algorithm_internals)
{
}

// --------------------------------------------------------------------------
teca_threaded_algorithm::~teca_threaded_algorithm() noexcept
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_threaded_algorithm::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_threaded_algorithm":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, bind_threads,
            "bind software threads to hardware cores")
        TECA_POPTS_GET(int, prefix, thread_pool_size,
            "number of threads in pool. When n == -1, 1 thread per core is "
            "created")
        TECA_POPTS_GET(int, prefix, stream_size,
            "number of datasests to pass per execute call. -1 means wait "
            "for all.")
        TECA_POPTS_GET(long long, prefix, poll_interval,
            "number of nanoseconds to wait between scans of the thread pool "
            "for completed tasks")
        TECA_POPTS_GET(int, prefix, threads_per_device,
            "Sets the number of threads that service each CUDA GPU. If -1 the "
            "default of 8 threads per CUDA GPU is used. If 0 only the CPU is used.")
        TECA_POPTS_GET(int, prefix, ranks_per_device,
            "Sets the number of threads that service each CUDA GPU. If -1 the "
            "default of ranks allowed to access each GPU.")
        TECA_POPTS_GET(int, prefix, propagate_device_assignment,
            "When set device assignment is taken from the in coming request. "
            "Otherwise the thread executing the upstream pipeline provides the "
            "device assignment.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_threaded_algorithm::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, int, prefix, bind_threads)
    TECA_POPTS_SET(opts, int, prefix, stream_size)
    TECA_POPTS_SET(opts, long long, prefix, poll_interval)
    TECA_POPTS_SET(opts, int, prefix, threads_per_device)
    TECA_POPTS_SET(opts, int, prefix, ranks_per_device)
    TECA_POPTS_SET(opts, int, prefix, propagate_device_assignment)

    // force update the the thread pool settings
    std::string opt_name = (prefix.empty()?"":prefix+"::") + "thread_pool_size";
    if (opts.count(opt_name))
        this->set_thread_pool_size(opts[opt_name].as<int>());
}
#endif

// --------------------------------------------------------------------------
void teca_threaded_algorithm::set_thread_pool_size(int n)
{
    TECA_PROFILE_METHOD(128, this, "set_thread_pool_size",
        this->internals->thread_pool_resize(this->get_communicator(),
            n, this->threads_per_device, this->ranks_per_device,
            this->bind_threads, this->verbose);
        )
}

// --------------------------------------------------------------------------
unsigned int teca_threaded_algorithm::get_thread_pool_size() const noexcept
{
    return this->internals->get_thread_pool_size();
}

// --------------------------------------------------------------------------
void teca_threaded_algorithm::set_data_request_queue(
    const p_teca_data_request_queue &queue)
{
    this->internals->thread_pool = queue;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_threaded_algorithm::request_data(
    teca_algorithm_output_port &current,
    const teca_metadata &request)
{
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm comm = this->get_communicator();

        // this rank is excluded from execution
        if (comm == MPI_COMM_NULL)
            return nullptr;
    }
#endif

    // check that we have a thread pool. it is left to a derived class
    // to construct the thread pool with a call to set_thread_pool_size
    // so that use of MPI can be tailored to the specific scenario
    if (!this->get_thread_pool_size())
    {
        TECA_ERROR("no thread pool available, set_thread_pool_size")
        return nullptr;
    }

    // execute current algorithm to fulfill the request.
    // return the data
    p_teca_threaded_algorithm alg =
        std::dynamic_pointer_cast<teca_threaded_algorithm>(get_algorithm(current));

    unsigned int port = get_port(current);

    // check for cached data
    teca_metadata key = alg->get_cache_key(port, request);
    const_p_teca_dataset out_data = alg->get_output_data(port, key);
    if (!out_data)
    {
        // determine what data is available on our inputs
        unsigned int n_inputs = alg->get_number_of_input_connections();
        std::vector<teca_metadata> input_md(n_inputs);
        for (unsigned int i = 0; i < n_inputs; ++i)
        {
            input_md[i]
              = alg->get_output_metadata(alg->get_input_connection(i));
        }

        // get requests for upstream data
        std::vector<teca_metadata> up_reqs;
        TECA_PROFILE_PIPELINE(128, alg, "get_upstream_request", port,
            up_reqs = alg->get_upstream_request(port, input_md, request);
            )

        // push data requests on to the thread pool's work
        // queue. mapping the requests round-robbin on to
        // the inputs
        size_t n_up_reqs = up_reqs.size();

        TECA_PROFILE_THREAD_POOL(128, this,
            this->internals->thread_pool->size(), n_up_reqs,

            for (unsigned int i = 0; i < n_up_reqs; ++i)
            {
                if (!up_reqs[i].empty())
                {
                    teca_algorithm_output_port &up_port
                        = alg->get_input_connection( i % n_inputs );

                    teca_data_request_task task(
                        teca_data_request(get_algorithm(up_port), up_port,
                                          up_reqs[i], this->propagate_device_assignment)
                        );

                    this->internals->thread_pool->push_task(task);
                }
            }

            int n_tasks_remaining = 0;
            do
            {
                // get the requested data. will block until it's ready.
                std::vector<const_p_teca_dataset> input_data;

                n_tasks_remaining = this->internals->thread_pool->wait_some(
                    this->stream_size, this->poll_interval, input_data);

                // when streaming recycle last round's output, this is
                // neccessary for reductions
                if (out_data)
                    input_data.push_back(out_data);

                // execute override
                TECA_PROFILE_PIPELINE(128, alg, "execute", port,
                    out_data = alg->execute(port, input_data,
                       request, n_tasks_remaining);
                )
            }
            while (n_tasks_remaining);
        )

        // cache the output
        alg->cache_output_data(port, key, out_data);
    }

    return out_data;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_threaded_algorithm::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request, int streaming)
{
    (void)streaming;
    return this->teca_algorithm::execute(port, input_data, request);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_threaded_algorithm::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    return this->execute(port, input_data, request, 0);
}
