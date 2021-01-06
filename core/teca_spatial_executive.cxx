#include "teca_spatial_executive.h"
#include "teca_config.h"
#include "teca_common.h"

#include <string>
#include <sstream>
#include <iostream>
#include <utility>

using extent_t = std::array(unsigned long, 6>;


namespace
{

// split block 1 into 2 blocks in the d direction. block1 is modified in place
// and the new block is returned in block 2. return 1 if the split succeeded
// and 0 if it failed.
int split(extent_t &block_1, extent_t &block_2, int d)
{
    // compute length in this direction
    int i0 = 2*d;
    int i1 = i0 + 1;

    unsigned long ni = block_1[i1] - block_1[i0] + 1;

    // can't split in this direction
    if (ni < 2)
        return 0;

    // compute the new length
    unsigned long no = ni/2;

    // copy input
    block_2 = block_1;

    // split
    block_1[i1] = block_1[i0] + no;
    block_2[i0] = std::min(block_2[i1], block_1[i1] + 1);

    return 1;
}

// given an input extent etx partition in into n_blocks disjoint blocks.
// return the list of new bloacks in blocks. return 0 if successful.
int partition(const extent_t &ext, int n_blocks,
    std::deque<extent_t> &blocks)
{
    // get the length in each direction
    unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nz = ext[5] - ext[4] + 1;
    unsigned long nxyz = nx*ny*nz;

    // check that it is possible to generate the requested number of blocks
    if (nxyz < n_blocks)
    {
        TECA_ERROR("Can't split " << nxyz << " cells into " << n_blocks)
        return -1;
    }

    // which directions can we split in?
    std::vector<int> dirs;
    if (nx > 1)
        dirs.push_back(0);
    if (ny > 1)
        dirs.push_back(1);
    if (nz > 1)
        dirs.push_back(2);

    int n_dirs = dirs.size();

    // start with the full extent
    blocks.push_back(ext);

    // split each block until the desired number is reached.
    while (blocks.size() < n_blocks)
    {
        // alternate splitable directions
        for (int d = 0; d < n_dirs; ++d)
        {
            // make a pass overt each block split it into 2 until the
            // desired number is realized
            unsigned long n = blocks.size();
            for (unsigned long i = 0; i < n; ++i)
            {
                // take the next block from the front
                extent_t b2;
                extent_t b1 = blocks.front();
                blocks.pop_front();

                // add the new blocks to the back
                if (split(b1, b2, dirs[d]))
                    blocks.push_back(b2);
                blocks.push_back(b1);

                // are we there yet?
                if (blocks.size() == n_blocks)
                    return 0;
            }
        }
    }

    return 0;
}
}

// --------------------------------------------------------------------------
teca_spatial_executive::teca_spatial_executive()
    : start_index(0), end_index(-1), stride(1)
{
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_index(long s)
{
    this->start_index = std::max(0l, s);
    this->end_index = s;
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_start_index(long s)
{
    this->start_index = std::max(0l, s);
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_end_index(long s)
{
    this->end_index = s;
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_stride(long s)
{
    this->stride = std::max(0l, s);
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_extent(unsigned long *ext)
{
    this->set_extent({ext[0], ext[1], ext[2], ext[3], ext[4], ext[4]});
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_extent(const std::vector<unsigned long> &ext)
{
    this->extent = ext;
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_bounds(double *bds)
{
    this->set_bounds({bds[0], bds[1], bds[2], bds[3], bds[4], bds[5]});
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_bounds(const std::vector<double> &bds)
{
    this->bounds = bds;
}

// --------------------------------------------------------------------------
void teca_spatial_executive::set_arrays(const std::vector<std::string> &v)
{
    this->arrays = v;
}

// --------------------------------------------------------------------------
int teca_spatial_executive::initialize(MPI_Comm comm, const teca_metadata &md)
{
    this->requests.clear();

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
#else
    (void)comm;
#endif

    // partition the domain such that each MPI rank has a unique subset
    // of the data to process.
    extent_t whole_extent;
    if (md.get("whole_extent", whole_extent))
    {
        TECA_ERROR("Failed to get whole_extent")
        return -1;
    }

    std::deque<extent_t> rank_extents;
    if (::partition(whole_extent, n_ranks, rank_extents))
    {
        TECA_ERROR("Failed to partition the domain")
        return -1;
    }

    extent_t local_extent = rank_extents[rank];

    rank_extents.clear();

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

    /*if (this->bounds.empty())
    {
        if (this->extent.empty())
        {
            std::vector<unsigned long> whole_extent(6, 0l);
            md.get("whole_extent", whole_extent);
            base_req.set("extent", whole_extent);
        }
        else
            base_req.set("extent", this->extent);
    }
    else
        base_req.set("bounds", this->bounds);*/
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

    // print some info about the set of requests
    if (this->get_verbose())
    {
        std::ostringstream oss;
        oss << teca_parallel_id()
            << " teca_spatial_executive::initialize index_initializer_key="
            << this->index_initializer_key << " " << this->index_initializer_key
            << "=" << n_indices << " index_request_key=" << this->index_request_key
            << " first=" << this->start_index << " last=" << this->end_index
            << " stride=" << this->stride << " block_start=" << block_start + first
            << " block_size=" << block_size;
        std::cerr << oss.str() << std::endl;
    }

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_spatial_executive::get_next_request()
{
    teca_metadata req;
    if (!this->requests.empty())
    {
        req = this->requests.back();
        this->requests.pop_back();

        // print the details of the current request. this is a execution
        // progress indicator of where the run is at
        if (this->get_verbose())
        {
            std::vector<unsigned long> ext;
            req.get("extent", ext);

            std::vector<double> bds;
            req.get("bounds", bds);

            unsigned long index;
            req.get(this->index_request_key, index);

            std::ostringstream oss;
            oss << teca_parallel_id()
               << " teca_spatial_executive::get_next_request "
               << this->index_request_key << "=" << index;

            if (bds.empty())
            {
                if (!ext.empty())
                    oss << " extent=" << ext[0] << ", " << ext[1] << ", "
                        << ext[2] << ", " << ext[3] << ", " << ext[4] << ", " << ext[5];
            }
            else
            {
                oss << " bounds=" << bds[0] << ", " << bds[1] << ", "
                    << bds[2] << ", " << bds[3] << ", " << bds[4] << ", " << bds[5];
            }

            std::cerr << oss.str() << std::endl;
        }
    }

    return req;
}
