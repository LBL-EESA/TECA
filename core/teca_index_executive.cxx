#include "teca_index_executive.h"
#include "teca_config.h"
#include "teca_common.h"
#include "teca_system_util.h"
#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

#include <string>
#include <sstream>
#include <iostream>
#include <utility>
#include <deque>


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
void teca_index_executive::set_bounds(double *bds)
{
    this->set_bounds({bds[0], bds[1], bds[2], bds[3], bds[4], bds[5]});
}

// --------------------------------------------------------------------------
void teca_index_executive::set_bounds(const std::vector<double> &bds)
{
    this->bounds = bds;
}

// --------------------------------------------------------------------------
void teca_index_executive::set_arrays(const std::vector<std::string> &v)
{
    this->arrays = v;
}
// --------------------------------------------------------------------------
void teca_index_executive::set_device_ids(const std::vector<int> &ids)
{
    this->device_ids = ids;
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
        TECA_FATAL_ERROR("No index initializer key has been specified")
        return -1;
    }

    if (md.get("index_request_key", this->index_request_key))
    {
        TECA_FATAL_ERROR("No index request key has been specified")
        return -1;
    }

    // locate available indices
    long n_indices = 0;
    if (md.get(this->index_initializer_key, n_indices))
    {
        TECA_FATAL_ERROR("metadata is missing the initializer key \""
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

    // determine the available CUDA GPUs
#if defined(TECA_HAS_CUDA)
    // check for an override to the default number of MPI ranks per device
    int ranks_per_device = 1;
    int ranks_per_device_set = teca_system_util::get_environment_variable
        ("TECA_RANKS_PER_DEVICE", ranks_per_device);

    std::vector<int> devices(this->device_ids);
    if (devices.empty())
    {
        if (teca_cuda_util::get_local_cuda_devices(comm,
            ranks_per_device, devices))
        {
            TECA_WARNING("Failed to determine the local CUDA device_ids."
                " Falling back to the default device.")
            devices.resize(1, 0);
        }
    }
    int n_devices = devices.size();
    size_t q = 0;
#endif

    // consrtuct base request
    teca_metadata base_req;
    if (this->bounds.empty())
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
        base_req.set("bounds", this->bounds);
    base_req.set("arrays", this->arrays);

    // apply the base request to local indices.
    for (size_t i = 0; i < block_size; ++i)
    {
        size_t index = i + block_start + first;
        if ((index % this->stride) == 0)
        {
            int device_id = -1;

#if defined(TECA_HAS_CUDA)
            // assign eaach request a device to execute on
            if (n_devices > 0)
                device_id = devices[q % n_devices];

            ++q;
#endif
            this->requests.push_back(base_req);
            this->requests.back().set("index_request_key", this->index_request_key);
            this->requests.back().set(this->index_request_key, {index, index});
            this->requests.back().set("device_id", device_id);
        }
    }

    // print some info about the set of requests
    if (this->get_verbose())
    {
        std::ostringstream oss;
        oss << teca_parallel_id()
            << " teca_index_executive::initialize index_initializer_key="
            << this->index_initializer_key << " " << this->index_initializer_key
            << "=" << n_indices << " index_request_key=" << this->index_request_key
            << " first=" << this->start_index << " last=" << this->end_index
            << " stride=" << this->stride << " block_start=" << block_start + first
            << " block_size=" << block_size;
#if defined(TECA_HAS_CUDA)
        oss << " n_cuda_devices=" << devices.size()
            << " device_ids=" << devices;
#endif
        std::cerr << oss.str() << std::endl;
    }

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

        // print the details of the current request. this is a execution
        // progress indicator of where the run is at
        if (this->get_verbose())
        {
            std::vector<unsigned long> ext;
            req.get("extent", ext);

            std::vector<double> bds;
            req.get("bounds", bds);

            unsigned long index = 0;
            req.get(this->index_request_key, index);

            std::ostringstream oss;
            oss << teca_parallel_id()
               << " teca_index_executive::get_next_request "
               << this->index_request_key << "=" << index;

            if (bds.empty())
            {
                if (!ext.empty())
                    oss << " extent=[" << ext << "]";
            }
            else
            {
                oss << " bounds=[" << bds << "]";
            }

#if defined(TECA_HAS_CUDA)
            int device_id = -1;
            req.get("device_id", device_id);
            oss << " device_id=" << device_id;
#endif

            std::cerr << oss.str() << std::endl;
        }
    }

    return req;
}
