#include "array_executive.h"

#include "teca_common.h"

#include <string>
#include <iostream>

using std::vector;
using std::string;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
int array_executive::initialize(const teca_meta_data &md)
{
    this->requests.clear();

    vector<string> array_names;
    if (md.get_prop("array_names", array_names))
    {
        TECA_ERROR("array_names meta data not found")
        return -1;
    }

    vector<double> time;
    if (md.get_prop("time", time))
    {
        //TECA_ERROR("time meta data not found")
        //return -2;
        time.push_back(0.0);
    }

    vector<size_t> extent;
    if (md.get_prop("extent", extent))
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
            teca_meta_data req;
            req.set_prop("time", time[i]);
            req.set_prop("array_name", array_names[j]);
            req.set_prop("extent", extent);
            this->requests.push_back(req);
        }
    }

    cerr << teca_parallel_id()
        << "array_executive::initialize n_times="
        << n_times << " n_arrays=" << n_arrays << endl;

    return 0;
}

// --------------------------------------------------------------------------
teca_meta_data array_executive::get_next_request()
{
    teca_meta_data req;
    if (!this->requests.empty())
    {
        req = this->requests.back();
        this->requests.pop_back();

        double time;
        string active_array;

        req.get_prop("array_name", active_array);
        req.get_prop("time", time);

        cerr << teca_parallel_id()
            << "array_executive::get_next_request array="
            << active_array << " time=" << time << endl;
    }
    return req;
}
