#include "teca_tc_trajectory.h"

#include "teca_database.h"
#include "teca_table.h"

#include "teca_variant_array.h"
#include "teca_metadata.h"

#include "gfdl_trajectory.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

/*
#if defined(TECA_HAS_UDUNITS)
#include "calcalcs.h"
#endif

    // compute the current date from the offset
    int curr_year = 0;
    int curr_month = 0;
    int curr_day = 0;
    int curr_hour = 0;
    int curr_minute = 0;
    double curr_second = 0;

#if defined(TECA_HAS_UDUNITS)
    if (calcalcs::date(time_offset, time_units.c_str(), &curr_year, &curr_month,
        &curr_day, &curr_hour, &curr_minute, &curr_second, calendar.c_str()))
    {
        TECA_ERROR("Failed to get the current date.")
    }
#else
    TECA_ERROR("Calendaring features are not present")
#endif
*/



using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_tc_trajectory::teca_tc_trajectory() :
    max_daily_distance(900.0),
    min_wind_speed(17.0),
    min_peak_wind_speed(17.0),
    min_vorticity(3.5e-5),
    core_temperature_delta(0.5),
    min_thickness(50.0),
    min_duration(2.0),
    low_search_latitude(-40.0),
    high_search_latitude(40.0),
    use_splines(0),
    use_thickness(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_tc_trajectory::~teca_tc_trajectory()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_tc_trajectory::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for " + prefix + "(teca_tc_trajectory)");

    opts.add_options()
        TECA_POPTS_GET(double, prefix, max_daily_distance, "TODO")
        TECA_POPTS_GET(double, prefix, min_wind_speed, "TODO")
        TECA_POPTS_GET(double, prefix, min_peak_wind_speed, "TODO")
        TECA_POPTS_GET(double, prefix, min_vorticity, "TODO")
        TECA_POPTS_GET(double, prefix, core_temperature_delta, "TODO")
        TECA_POPTS_GET(double, prefix, min_thickness, "TODO")
        TECA_POPTS_GET(double, prefix, min_duration, "TODO")
        TECA_POPTS_GET(double, prefix, low_search_latitude, "TODO")
        TECA_POPTS_GET(double, prefix, high_search_latitude, "TODO")
        TECA_POPTS_GET(int,    prefix, use_splines, "TODO")
        TECA_POPTS_GET(int,    prefix, use_thickness, "TODO")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_tc_trajectory::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, double, prefix, max_daily_distance)
    TECA_POPTS_SET(opts, double, prefix, min_wind_speed)
    TECA_POPTS_SET(opts, double, prefix, min_peak_wind_speed)
    TECA_POPTS_SET(opts, double, prefix, min_vorticity)
    TECA_POPTS_SET(opts, double, prefix, core_temperature_delta)
    TECA_POPTS_SET(opts, double, prefix, min_thickness)
    TECA_POPTS_SET(opts, double, prefix, min_duration)
    TECA_POPTS_SET(opts, double, prefix, low_search_latitude)
    TECA_POPTS_SET(opts, double, prefix, high_search_latitude)
    TECA_POPTS_SET(opts, int, prefix, use_splines)
    TECA_POPTS_SET(opts, int, prefix, use_thickness)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_tc_trajectory::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_tc_trajectory::get_output_metadata" << endl;
#endif
    (void)port;
    teca_metadata out_md(input_md[0]);
    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_tc_trajectory::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_tc_trajectory::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    teca_metadata req(request);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_tc_trajectory::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_tc_trajectory::execute" << endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_database db_in
        = std::dynamic_pointer_cast<const teca_database>(input_data[0]);

    if (!db_in)
    {
        TECA_ERROR("Input dataset is not a teca_database")
        return nullptr;
    }

    const_p_teca_table tc_summary = db_in->get_table("summary");


    const_p_teca_table tc_details = db_in->get_table("detections");


    p_teca_table traj_summary = teca_table::New();
    p_teca_table traj_details = teca_table::New();

    if (::gfdl_tc_trajectory(tc_summary, tc_details,
        this->max_daily_distance, this->min_wind_speed, this->min_peak_wind_speed,
        this->min_duration, this->min_vorticity, this->core_temperature_delta,
        this->min_thickness, this->high_search_latitude, this->low_search_latitude,
        this->use_splines, this->use_thickness, traj_summary, traj_details))
    {
        TECA_ERROR("Trajectory computation fialed.")
        return nullptr;
    }

    p_teca_database db_out = teca_database::New();
    db_out->append_table("summary", traj_summary);
    db_out->append_table("trajectories", traj_details);

    return db_out;
}
