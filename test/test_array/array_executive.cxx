#include "array_executive.h"

#include "teca_common.h"
#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

#include <string>
#include <iostream>
#include <deque>

using std::vector;
using std::string;
using std::cerr;
using std::endl;


// --------------------------------------------------------------------------
int array_executive::initialize(MPI_Comm comm, const teca_metadata &md)
{
    (void)comm;

    this->requests.clear();

    // figure out the keys
    std::string initializer_key;
    if (md.get("index_initializer_key", initializer_key))
    {
        TECA_ERROR("No index initializer key has been specified")
        return -1;
    }

    std::string request_key;
    if (md.get("index_request_key", request_key))
    {
        TECA_ERROR("No index request key has been specified")
        return -1;
    }

    // locate available indices
    long n_indices = 1;
    if (md.get(initializer_key, n_indices))
    {
        TECA_ERROR("metadata is missing the initializer key")
        return -1;
    }

    vector<string> array_names;
    if (md.get("array_names", array_names))
    {
        TECA_ERROR("array_names meta data not found")
        return -1;
    }

    vector<double> time;
    if (md.get("time", time))
    {
        //TECA_ERROR("time meta data not found")
        //return -2;
        time.push_back(0.0);
    }

    vector<size_t> extent;
    if (md.get("extent", extent))
    {
        TECA_ERROR("extent meta data not found")
        return -3;
    }

    // determine the available CUDA GPUs
    std::vector<int> device_ids;
#if defined(TECA_HAS_CUDA)
    int ranks_per_device = -1;
    if (teca_cuda_util::get_local_cuda_devices(comm,
        ranks_per_device, device_ids))
    {
        TECA_WARNING("Failed to determine the local CUDA device_ids."
            " Falling back to the default device.")
        device_ids.resize(1, 0);
    }
#endif

    int n_devices = device_ids.size();

    // add the CPU
    if (n_devices < 1)
    {
        device_ids.push_back(-1);
        n_devices = 1;
    }

    // for each time request each array
    size_t n_times = time.size();
    size_t n_arrays = array_names.size();
    for (size_t i = 0, q = 0; i < n_times; ++i)
    {
        for (size_t j = 0; j < n_arrays; ++j)
        {
            int device_id = device_ids[q % n_devices];
            ++q;

            teca_metadata req;
            req.set("index_request_key", std::string("time_step"));
            req.set("time_step", {i, i});
            req.set("time", time[i]);
            req.set("array_name", array_names[j]);
            req.set("extent", extent);
            req.set("device_id", device_id);

            this->requests.push_back(req);
        }
    }

#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_executive::initialize n_times="
        << n_times << " n_arrays=" << n_arrays
        << " n_devices=" << device_ids.size()
        << " device_ids=" << device_ids << endl;
#endif

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata array_executive::get_next_request()
{
    teca_metadata req;
    if (!this->requests.empty())
    {
        req = this->requests.back();
        this->requests.pop_back();

        double time;
        string active_array;

        req.get("array_name", active_array);
        req.get("time", time);

#ifndef TECA_NDEBUG
        cerr << teca_parallel_id()
            << "array_executive::get_next_request array="
            << active_array << " time=" << time << endl;
#endif
    }
    return req;
}
