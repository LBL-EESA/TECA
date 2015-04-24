#include "array_executive.h"

#include "teca_common.h"

#include <string>
#include <iostream>

using std::vector;
using std::string;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
int array_executive::initialize(const teca_metadata &md)
{
    this->requests.clear();

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

    // for each time request each array
    size_t n_times = time.size();
    size_t n_arrays = array_names.size();
    for (size_t i = 0; i < n_times; ++i)
    {
        for (size_t j = 0; j < n_arrays; ++j)
        {
            teca_metadata req;
            req.insert("time_step", i);
            req.insert("time", time[i]);
            req.insert("array_name", array_names[j]);
            req.insert("extent", extent);
            this->requests.push_back(req);
        }
    }

#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_executive::initialize n_times="
        << n_times << " n_arrays=" << n_arrays << endl;
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
