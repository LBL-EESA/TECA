#include "teca_index_reduce.h"
#include "teca_binary_stream.h"
#include "teca_mpi.h"
#include "teca_profiler.h"

#include <sstream>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif



namespace internal {

#if defined(TECA_HAS_MPI)

// TODO -- handle large messages, work around int in MPI api

// helper for sending binary data over MPI
int send(MPI_Comm comm, int dest, teca_binary_stream &s)
{
    unsigned long long n = s.size();
    if (MPI_Send(&n, 1, MPI_UNSIGNED_LONG_LONG, dest, 3210, comm))
    {
        TECA_ERROR("failed to send send message size")
        return -1;
    }

    if (n)
    {
        if (MPI_Send(s.get_data(), n, MPI_UNSIGNED_CHAR,
                dest, 3211, comm))
        {
            TECA_ERROR("failed to send message")
            return -2;
        }
    }

    return 0;
}

// helper for receiving data over MPI
int recv(MPI_Comm comm, int src, teca_binary_stream &s)
{
    unsigned long n = 0;
    MPI_Status stat;
    if (MPI_Recv(&n, 1, MPI_UNSIGNED_LONG_LONG, src, 3210, comm, &stat))
    {
        TECA_ERROR("failed to receive message size")
        return -2;
    }

    s.resize(n);

    if (n)
    {
        if (MPI_Recv(s.get_data(), n, MPI_UNSIGNED_CHAR, src, 3211, comm, &stat))
        {
            TECA_ERROR("failed to receive message")
            return -2;
        }
    }

    return 0;
}
#endif

// --------------------------------------------------------------------------
void block_decompose(MPI_Comm comm, unsigned long n_indices, unsigned long n_ranks,
    unsigned long rank, unsigned long &block_size, unsigned long &block_start,
    bool verbose)
{
#if !defined(TECA_HAS_MPI)
    (void)comm;
#endif
    unsigned long n_big_blocks = n_indices%n_ranks;
    if (rank < n_big_blocks)
    {
        block_size = n_indices/n_ranks + 1;
        block_start = block_size*rank;
    }
    else
    {
        block_size = n_indices/n_ranks;
        block_start = block_size*rank + n_big_blocks;
    }
    if (verbose)
    {
        std::vector<unsigned long> decomp = {block_start, block_size};
#if defined(TECA_HAS_MPI)
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (is_init)
        {
            if (rank == 0)
            {
                decomp.resize(2*n_ranks);
                    MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                        decomp.data(), 2, MPI_UNSIGNED_LONG, 0, comm);
            }
            else
            {
                MPI_Gather(decomp.data(), 2, MPI_UNSIGNED_LONG,
                    nullptr, 0, MPI_DATATYPE_NULL, 0, comm);
            }
        }
#endif
        if (rank == 0)
        {
            std::ostringstream oss;
            for (unsigned long i = 0; i < n_ranks; ++i)
            {
                unsigned long ii = 2*i;
                oss << i << " : ";
                if (decomp[ii+1])
                    oss << decomp[ii] << " - " << decomp[ii] + decomp[ii+1] - 1;
                else
                    oss << "- - -";
                if (i < n_ranks - 1)
                     oss << std::endl;
            }
            TECA_STATUS("map index decomposition:" << std::endl << oss.str())
        }
    }
}

};

// --------------------------------------------------------------------------
teca_index_reduce::teca_index_reduce() :
    extent{0,-1,0,0,0,0}, bounds{0.,-1.,0.,0.,0.,0.}, arrays{},
    start_index(0), end_index(-1)
{
    this->set_stream_size(2);
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_index_reduce::get_properties_description(const std::string &prefix,
    options_description &global_opts)
{
    this->teca_threaded_algorithm::get_properties_description(prefix, global_opts);

    options_description opts("Options for "
        + (prefix.empty()?"teca_index_reduce":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(extent_type, prefix, extent,
            "an index space spatial subset of the data to request"
            " [i0, i1, j0, j1, k0, k1] (optional)")
        TECA_POPTS_MULTI_GET(bounds_type, prefix, bounds,
            "a world space subset of the data to request"
            " [x0, x1, y0, y1, z0, z1] (optional)")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, arrays,
            "the set of arrays to request (optional)")
        TECA_POPTS_GET(long, prefix, start_index, "first index to process")
        TECA_POPTS_GET(long, prefix, end_index, "last index to process. "
            "If set to -1 all indices are processed.")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_index_reduce::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_threaded_algorithm::set_properties(prefix, opts);



    TECA_POPTS_SET(opts, extent_type, prefix, extent)
    TECA_POPTS_SET(opts, bounds_type, prefix, bounds)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, arrays)
    TECA_POPTS_SET(opts, long, prefix, start_index)
    TECA_POPTS_SET(opts, long, prefix, end_index)
}
#endif

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_index_reduce::initialize_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request_in)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_index_reduce::initialize_upstream_request" << std::endl;
#endif
    (void) port;

    teca_metadata request(request_in);

    if (!request.has("bounds") && !request.has("extent"))
    {
        // if there are bounds or extent already in the request, leave them
        // alone. otherwise use the runtime provided values, or implement the
        // default behavior

        if (this->bounds[1] < this->bounds[0])
        {
            // no bounds provided, fall back to extents
            if (this->extent[1] < this->extent[0])
            {
                // no extent provided
                const teca_metadata &md = input_md[0];
                extent_type whole_extent{0,0,0,0,0,0};
                if (!md.get("whole_extent", whole_extent))
                {
                    // fall back to whole extent
                    request.set("extent", whole_extent);
                }
            }
            else
            {
                // use extent specified
                request.set("extent", this->extent);
            }
        }
        else
        {
            // use bounds specifed
            request.set("bounds", this->bounds);
        }
    }

    if (!request.has("arrays"))
    {
        // if the request already has arrays do not modify. otherwise use the
        // runtime provided values

       if (!this->arrays.empty())
           request.set("arrays", this->arrays);
    }

    return {request};
}

// --------------------------------------------------------------------------
teca_metadata teca_index_reduce::initialize_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_index_reduce::intialize_output_metadata" << std::endl;
#endif
    (void) port;

    teca_metadata output_md(input_md[0]);
    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_index_reduce::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &)
{
    std::vector<teca_metadata> up_req;

    unsigned long rank = 0;
    unsigned long n_ranks = 1;
    MPI_Comm comm = this->get_communicator();
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        // this is excluded from processing
        if (comm == MPI_COMM_NULL)
            return up_req;

        int tmp = 0;
        MPI_Comm_size(comm, &tmp);
        n_ranks = tmp;
        MPI_Comm_rank(comm, &tmp);
        rank = tmp;
    }
#endif

    // locate the keys that enable us to know how many
    // requests we need to make and what key to use
    const teca_metadata &md = input_md[0];
    std::string initializer_key;
    if (md.get("index_initializer_key", initializer_key))
    {
        TECA_FATAL_ERROR("No index initializer key has been specified")
        return up_req;
    }

    std::string request_key;
    if (md.get("index_request_key", request_key))
    {
        TECA_FATAL_ERROR("No index request key has been specified")
        return up_req;
    }

    // locate available indices
    long n_indeces = 0;
    if (md.get(initializer_key, n_indeces))
    {
        TECA_FATAL_ERROR("metadata is missing index initializer key")
        return up_req;
    }

    // apply restriction
    long last = this->end_index >= 0 ? this->end_index : n_indeces - 1;

    long first = ((this->start_index >= 0) && (this->start_index <= last))
        ? this->start_index : 0;

    n_indeces = last - first + 1;

    // partition indices across MPI ranks. each rank will end up with a unique
    // block of indices to process.
    unsigned long block_size = 1;
    unsigned long block_start = 0;

    internal::block_decompose(comm, n_indeces, n_ranks, rank, block_size,
        block_start, this->get_verbose());

    // get the filters basic request. do not pass the incoming request
    // this isolates the pipeline below the reduction from the pipeline
    // above it which will have different data structures and control keys

    std::vector<teca_metadata> base_req
        = this->initialize_upstream_request(port, input_md, teca_metadata());

    // apply the base request to local indices.
    // requests are mapped onto inputs round robbin
    for (unsigned long i = 0; i < block_size; ++i)
    {
        unsigned long index = i + block_start + first;
        unsigned long n_reqs = base_req.size();
        for (unsigned long j = 0; j < n_reqs; ++j)
        {
            teca_metadata tmp(base_req[j]);
            tmp.set(request_key, {index, index});
            tmp.set("index_request_key", request_key);
            up_req.emplace_back(std::move(tmp));
        }
    }

    return up_req;
}

// --------------------------------------------------------------------------
teca_metadata teca_index_reduce::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    // get output metadata from the implementation. by default a
    // copy is made
    teca_metadata output_md
        = this->initialize_output_metadata(port, input_md);

    // by default the reduction takes a massive dataset, with an index for each
    // representative piece, and reduces it in a single pass to a single output
    // if other behavior is needed implementations will have to override this
    // method
    output_md.set("index_initializer_key", std::string("number_of_passes"));
    output_md.set("index_request_key", std::string("pass_number"));
    output_md.set("number_of_passes", 1l);

    return output_md;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_index_reduce::reduce_local(int device_id,
    std::vector<const_p_teca_dataset> input_data) // pass by value is necessary
{
    unsigned long n_in = input_data.size();

    if (n_in == 0)
        return p_teca_dataset();

    do
    {
        if (n_in % 2)
            TECA_PROFILE_METHOD(128, this, "reduce",
                input_data[0] = this->reduce(device_id, input_data[0],
                    (n_in > 1 ? input_data[n_in-1] : nullptr));
                )

        n_in /= 2;
        for (unsigned long i = 0; i < n_in; ++i)
        {
            unsigned long ii = 2*i;
            TECA_PROFILE_METHOD(128, this, "reduce",
                input_data[i] = this->reduce(device_id, input_data[ii],
                    input_data[ii+1]);
                )
        }
    }
    while (n_in > 1);

    return input_data[0];
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_index_reduce::reduce_remote(int device_id,
    const_p_teca_dataset local_data)
{
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm comm = this->get_communicator();

        unsigned long rank = 0;
        unsigned long n_ranks = 1;
        int tmp = 0;
        MPI_Comm_size(comm, &tmp);
        n_ranks = tmp;
        MPI_Comm_rank(comm, &tmp);
        rank = tmp;

        // special case 1 rank, nothing to do
        if (n_ranks < 2)
            return local_data;

        // reduce remote datasets in binary tree order
        unsigned long id = rank + 1;
        unsigned long up_id = id/2;
        unsigned long left_id = 2*id;
        unsigned long right_id = left_id + 1;

        teca_binary_stream bstr;

        // recv from left
        if (left_id <= n_ranks)
        {
            if (internal::recv(comm, left_id-1, bstr))
            {
                TECA_ERROR("failed to recv from left")
                return p_teca_dataset();
            }

            p_teca_dataset left_data;
            if (local_data && bstr)
            {
                left_data = local_data->new_instance();
                left_data->from_stream(bstr);
            }

            TECA_PROFILE_METHOD(128, this, "reduce",
                local_data = this->reduce(device_id, local_data, left_data);
                )

            bstr.resize(0);
        }

        // recv from right
        if (right_id <= n_ranks)
        {
            if (internal::recv(comm, right_id-1,  bstr))
            {
                TECA_ERROR("failed to recv from right")
                return p_teca_dataset();
            }

            p_teca_dataset right_data;
            if (local_data && bstr)
            {
                right_data = local_data->new_instance();
                right_data->from_stream(bstr);
            }

            TECA_PROFILE_METHOD(128, this, "reduce",
                local_data = this->reduce(device_id, local_data, right_data);
                )

            bstr.resize(0);
        }

        // send up
        if (rank)
        {
            if (local_data)
                local_data->to_stream(bstr);

            if (internal::send(comm, up_id-1, bstr))
                TECA_ERROR("failed to send up")

            // all but root returns an empty dataset
            return p_teca_dataset();
        }
    }
#endif
    // rank 0 has all the data
    return local_data;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_index_reduce::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request, int streaming)
{
    (void)port;
    (void)request;

#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm comm = this->get_communicator();

        // this rank is excluded from processing
        if (comm == MPI_COMM_NULL)
            return nullptr;
    }
#endif

    // get the device to execute on
    int device_id = -1;
    if (request.get("device_id", device_id))
    {
        TECA_ERROR("The request is missing the device_id key."
            " Executing the reduction on the CPU.")
    }

    // note: it is not an error to have no input data.  this can occur if there
    // are fewer indices to process than there are MPI ranks.

    // reduce data from threads on this MPI rank
    const_p_teca_dataset tmp = this->reduce_local(device_id, input_data);

    // when streaming execute will be called multiple times with 1 or more
    // input datasets. When all the data has been passed streaming is 0. Only
    // then do we reduce remote data and finalize the reduction.
    if (streaming)
        return tmp;

    // reduce data across MPI ranks
    tmp = this->reduce_remote(device_id, tmp);

    // finalize the reduction
    tmp = this->finalize(device_id, tmp);

    if (!tmp)
        return nullptr;

    // make a shallow copy, metadata is always deep copied
    p_teca_dataset output_ds = tmp->new_instance();
    output_ds->shallow_copy(std::const_pointer_cast<teca_dataset>(tmp));
    output_ds->set_request_index("pass_number", 0ul);

    return output_ds;
}
