#include "teca_index_reduce.h"
#include "teca_binary_stream.h"

#include <sstream>

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

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
                dest, 3211, MPI_COMM_WORLD))
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
void block_decompose(unsigned long n_indices, unsigned long n_ranks,
    unsigned long rank, unsigned long &block_size, unsigned long &block_start,
    bool verbose)
{
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
        if (rank == 0)
        {
            decomp.resize(2*n_ranks);
#if defined(TECA_HAS_MPI)
            MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, decomp.data(),
                2, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Gather(decomp.data(), 2, MPI_UNSIGNED_LONG, nullptr,
                0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
#endif
        }
        if (rank == 0)
        {
            std::ostringstream oss;
            for (unsigned long i = 0; i < n_ranks; ++i)
            {
                unsigned long ii = 2*i;
                oss << i << " : " << decomp[ii] << " - " << decomp[ii] + decomp[ii+1] -1
                    << (i < n_ranks-1 ? "\n" : "");
            }
            TECA_STATUS("map index decomposition:"
                << std::endl << oss.str())
        }
    }
}

};

// --------------------------------------------------------------------------
teca_index_reduce::teca_index_reduce()
    : start_index(0), end_index(-1), enable_mpi(1)
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_index_reduce::get_properties_description(const std::string &prefix,
    options_description &global_opts)
{
    this->teca_threaded_algorithm::get_properties_description(prefix, global_opts);

    options_description opts("Options for "
        + (prefix.empty()?"teca_index_reduce":prefix));

    opts.add_options()
        TECA_POPTS_GET(long, prefix, start_index, "first index to process (0)")
        TECA_POPTS_GET(long, prefix, end_index, "last index to process. "
            "If set to -1 all indices are processed. (-1)")
        TECA_POPTS_GET(int, prefix, enable_mpi, "enable use of MPI (1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_index_reduce::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_threaded_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, long, prefix, start_index)
    TECA_POPTS_SET(opts, long, prefix, end_index)
    TECA_POPTS_SET(opts, int, prefix, enable_mpi)
}
#endif

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_index_reduce::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    std::vector<teca_metadata> up_req;

    // locate the keys that enable us to know how many
    // requests we need to make and what key to use
    const teca_metadata &md = input_md[0];
    std::string initializer_key;
    if (md.get("index_initializer_key", initializer_key))
    {
        TECA_ERROR("No index initializer key has been specified")
        return up_req;
    }

    std::string request_key;
    if (md.get("index_request_key", request_key))
    {
        TECA_ERROR("No index request key has been specified")
        return up_req;
    }

    // locate available indices
    long n_indeces = 0;
    if (md.get(initializer_key, n_indeces))
    {
        TECA_ERROR("metadata is missing index initializer key")
        return up_req;
    }

    // apply restriction
    long last = this->end_index >= 0 ? this->end_index : n_indeces - 1;

    long first = ((this->start_index >= 0) && (this->start_index <= last))
        ? this->start_index : 0;

    n_indeces = last - first + 1;

    // partition indices across MPI ranks. each rank
    // will end up with a unique block of indices
    // to process.
    unsigned long rank = 0;
    unsigned long n_ranks = 1;
    if (this->enable_mpi)
    {
#if defined(TECA_HAS_MPI)
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (is_init)
        {
            int tmp = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &tmp);
            n_ranks = tmp;
            MPI_Comm_rank(MPI_COMM_WORLD, &tmp);
            rank = tmp;
        }
    }
#endif
    unsigned long block_size = 1;
    unsigned long block_start = 0;

    internal::block_decompose(n_indeces, n_ranks, rank, block_size,
        block_start, this->get_verbose());

    // get the filters basic request
    std::vector<teca_metadata> base_req
        = this->initialize_upstream_request(port, input_md, request);

    // apply the base request to local indices.
    // requests are mapped onto inputs round robbin
    for (unsigned long i = 0; i < block_size; ++i)
    {
        unsigned long index = i + block_start + first;
        unsigned long n_reqs = base_req.size();
        for (unsigned long j = 0; j < n_reqs; ++j)
        {
            up_req.push_back(base_req[j]);
            up_req.back().insert(request_key, index);
        }
    }

    return up_req;
}

// --------------------------------------------------------------------------
teca_metadata teca_index_reduce::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    // figure out the request key
    std::string request_key;
    if (input_md[0].get("index_request_key", request_key))
    {
        TECA_ERROR("No index request key has been specified")
        return teca_metadata();
    }

    // make a copy and remove the request key
    teca_metadata output_md
        = this->initialize_output_metadata(port, input_md);

    output_md.remove(request_key);

    return output_md;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_index_reduce::reduce_local(
    std::vector<const_p_teca_dataset> input_data) // pass by value is intentional
{
    unsigned long n_in = input_data.size();

    if (n_in == 0)
        return p_teca_dataset();

    if (n_in == 1)
        return input_data[0];

    while (n_in > 1)
    {
        if (n_in % 2)
            input_data[0] = this->reduce(input_data[0], input_data[n_in-1]);

        unsigned long n = n_in/2;
        for (unsigned long i = 0; i < n; ++i)
        {
            unsigned long ii = 2*i;
            input_data[i] = this->reduce(input_data[ii], input_data[ii+1]);
        }

        n_in = n;
    }
    return input_data[0];
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_index_reduce::reduce_remote(
    const_p_teca_dataset local_data) // pass by value is intentional
{
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        unsigned long rank = 0;
        unsigned long n_ranks = 1;
        int tmp = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &tmp);
        n_ranks = tmp;
        MPI_Comm_rank(MPI_COMM_WORLD, &tmp);
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
            if (internal::recv(MPI_COMM_WORLD, left_id-1, bstr))
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

            local_data = this->reduce(local_data, left_data);

            bstr.resize(0);
        }

        // recv from right
        if (right_id <= n_ranks)
        {
            if (internal::recv(MPI_COMM_WORLD, right_id-1,  bstr))
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

            local_data = this->reduce(local_data, right_data);

            bstr.resize(0);
        }

        // send up
        if (rank)
        {
            if (local_data)
                local_data->to_stream(bstr);

            if (internal::send(MPI_COMM_WORLD, up_id-1, bstr))
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
const_p_teca_dataset teca_index_reduce::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;
    (void)request;

    // note: it is not an error to have no input data.
    // this can occur if there are fewer indices
    // to process than there are MPI ranks.

    if (this->enable_mpi)
        return this->reduce_remote(this->reduce_local(input_data));
    return this->reduce_local(input_data);
}
