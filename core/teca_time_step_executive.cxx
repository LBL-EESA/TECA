#include "teca_time_step_executive.h"

#include "teca_common.h"

#include <string>
#include <iostream>
#include <utility>

using std::vector;
using std::string;
using std::cerr;
using std::endl;

#define TECA_TIME_STEP_EXECUTIVE_DEBUG

// --------------------------------------------------------------------------
teca_time_step_executive::teca_time_step_executive()
    : process_all(true), first_step(0), last_step(-1), stride(1)
{
}

// --------------------------------------------------------------------------
void teca_time_step_executive::set_step(long s)
{
    this->first_step = std::max(0l, s);
    this->last_step = s;
}

// --------------------------------------------------------------------------
void teca_time_step_executive::set_first_step(long s)
{
    this->first_step = std::max(0l, s);
}

// --------------------------------------------------------------------------
void teca_time_step_executive::set_last_step(long s)
{
    this->last_step = s;
}

// --------------------------------------------------------------------------
void teca_time_step_executive::set_stride(long s)
{
    this->stride = std::max(0l, s);
}

// --------------------------------------------------------------------------
void teca_time_step_executive::set_extent(unsigned long *ext)
{
    this->set_extent({ext[0], ext[1], ext[2], ext[3], ext[4], ext[4]});
}

// --------------------------------------------------------------------------
void teca_time_step_executive::set_extent(const std::vector<unsigned long> &ext)
{
    this->extent = ext;
}

// --------------------------------------------------------------------------
void teca_time_step_executive::set_arrays(const std::vector<std::string> &v)
{
    this->arrays = v;
}

// --------------------------------------------------------------------------
int teca_time_step_executive::initialize(const teca_metadata &md)
{
    this->requests.clear();

    unsigned long number_of_steps;
    if (md.get("number_of_time_steps", number_of_steps))
    {
        TECA_ERROR("metadata missing \"number_of_time_steps\"")
        return -1;
    }

    unsigned long last_step
        = (this->last_step >= 0 ? this->last_step : number_of_steps - 1);

    if ((this->first_step >= static_cast<long>(number_of_steps))
        || (this->first_step > static_cast<long>(last_step))
        || (last_step >= number_of_steps))
    {
        TECA_ERROR(
            << "Inavlid time step range " << this->first_step
            << ", " << last_step << ". " << number_of_steps
            << " time steps are available.")
        return -1;
    }

    vector<unsigned long> whole_extent;
    if (md.get("whole_extent", whole_extent))
    {
        TECA_ERROR("metadata missing \"whole_extent\"")
        return -1;
    }

    unsigned long step = this->first_step;
    do
    {
        teca_metadata req;
        req.insert("time_step", step);
        if (this->extent.empty())
        {
            req.insert("extent", whole_extent);
        }
        else
        {
            req.insert("extent", this->extent);
        }
        req.insert("arrays", this->arrays);
        this->requests.push_back(req);
        step += this->stride;
    }
    while ((step <= last_step) && this->process_all);

#if defined(TECA_TIME_STEP_EXECUTIVE_DEBUG)
    cerr << teca_parallel_id()
        << "teca_time_step_executive::initialize first="
        << this->first_step << " last=" << last_step << " stride="
        << this->stride << endl;
#endif

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_time_step_executive::get_next_request()
{
    teca_metadata req;
    if (!this->requests.empty())
    {
        req = this->requests.back();
        this->requests.pop_back();

#if defined(TECA_TIME_STEP_EXECUTIVE_DEBUG)
        vector<unsigned long> ext;
        req.get("extent", ext);

        unsigned long time_step;
        req.get("time_step", time_step);

        cerr << teca_parallel_id()
            << "teca_time_step_executive::get_next_request time_step="
            << time_step << " extent=" << ext[0] << ", " << ext[1] << ", "
            << ext[2] << ", " << ext[3] << ", " << ext[4] << ", " << ext[5]
            << endl;
#endif
    }

    return req;
}
