#include "teca_index_executive.h"
#include "teca_config.h"
#include "teca_common.h"

#include <string>
#include <iostream>
#include <utility>

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_index_executive::teca_index_executive()
    : start_index(0), end_index(-1), stride(1)
{
}

// --------------------------------------------------------------------------
void teca_index_executive::set_index(long s)
{
    this->start_index = std::max(0l, s);
    this->end_index = s;
}

// --------------------------------------------------------------------------
void teca_index_executive::set_start_index(long s)
{
    this->start_index = std::max(0l, s);
}

// --------------------------------------------------------------------------
void teca_index_executive::set_end_index(long s)
{
    this->end_index = s;
}

// --------------------------------------------------------------------------
void teca_index_executive::set_stride(long s)
{
    this->stride = std::max(0l, s);
}

// --------------------------------------------------------------------------
void teca_index_executive::set_extent(unsigned long *ext)
{
    this->set_extent({ext[0], ext[1], ext[2], ext[3], ext[4], ext[4]});
}

// --------------------------------------------------------------------------
void teca_index_executive::set_extent(const std::vector<unsigned long> &ext)
{
    this->extent = ext;
}

// --------------------------------------------------------------------------
void teca_index_executive::set_arrays(const std::vector<std::string> &v)
{
    this->arrays = v;
}

// --------------------------------------------------------------------------
int teca_index_executive::initialize(MPI_Comm comm, const teca_metadata &md)
{
#if !defined(TECA_HAS_MPI)
    (void)comm;
#endif
    this->requests.clear();

    // locate the keys that enable us to know how many
    // requests we need to make and what key to use
    if (md.get("index_initializer_key", this->index_initializer_key))
    {
        TECA_ERROR("No index initializer key has been specified")
        return -1;
    }

    if (md.get("index_request_key", this->index_request_key))
    {
        TECA_ERROR("No index request key has been specified")
        return -1;
    }

    // locate available indices
    long n_indices = 0;
    if (md.get(this->index_initializer_key, n_indices))
    {
        TECA_ERROR("metadata is missing the initializer key \""
            << this->index_initializer_key << "\"")
        return -1;
    }

    // apply restriction
    long last
        = this->end_index >= 0 ? this->end_index : n_indices - 1;

    long first
        = ((this->start_index >= 0) && (this->start_index <= last))
            ? this->start_index : 0;

    n_indices = last - first + 1;

    // partition indices across MPI ranks. each rank
    // will end up with a unique block of indices
    // to process.
    size_t rank = 0;
    size_t n_ranks = 1;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        int tmp = 0;
        MPI_Comm_size(comm, &tmp);
        n_ranks = tmp;
        MPI_Comm_rank(comm, &tmp);
        rank = tmp;
    }
#endif
    size_t n_big_blocks = n_indices%n_ranks;
    size_t block_size = 1;
    size_t block_start = 0;
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

    // consrtuct base request
    teca_metadata base_req;
    if (this->extent.empty())
    {
        std::vector<unsigned long> whole_extent(6, 0l);
        md.get("whole_extent", whole_extent);
        base_req.set("extent", whole_extent);
    }
    else
        base_req.set("extent", this->extent);
    base_req.set("arrays", this->arrays);

    // apply the base request to local indices.
    for (size_t i = 0; i < block_size; ++i)
    {
        size_t index = i + block_start + first;
        if ((index % this->stride) == 0)
        {
            this->requests.push_back(base_req);
            this->requests.back().set(this->index_request_key, index);
        }
    }

    if (this->get_verbose())
        cerr << teca_parallel_id()
            << " teca_index_executive::initialize index_initializer_key="
            << this->index_initializer_key << " index_request_key="
            << this->index_request_key << " first=" << this->start_index
            << " last=" << this->end_index << " stride=" << this->stride
            << endl;

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_index_executive::get_next_request()
{
    teca_metadata req;
    if (!this->requests.empty())
    {
        req = this->requests.back();
        this->requests.pop_back();

        if (this->get_verbose())
        {
            std::vector<unsigned long> ext;
            req.get("extent", ext);

            unsigned long index;
            req.get(this->index_request_key, index);

            cerr << teca_parallel_id()
                << " teca_index_executive::get_next_request "
                << this->index_request_key << "=" << index
                << " extent=" << ext[0] << ", " << ext[1] << ", " << ext[2]
                << ", " << ext[3] << ", " << ext[4] << ", " << ext[5] << endl;
        }
    }

    return req;
}
