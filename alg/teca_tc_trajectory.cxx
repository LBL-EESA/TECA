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
    min_wind_duration(2.0),
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
        TECA_POPTS_GET(double, prefix, max_daily_distance, "max distance a storm can move on the same track in single day")
        TECA_POPTS_GET(double, prefix, min_wind_speed, "minimum wind speed to be worthy of tracking")
        TECA_POPTS_GET(double, prefix, min_wind_duration, "minimum number of days wind speed must exceed the min")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_tc_trajectory::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, double, prefix, max_daily_distance)
    TECA_POPTS_SET(opts, double, prefix, min_wind_speed)
    TECA_POPTS_SET(opts, double, prefix, min_wind_duration)
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
    const_p_teca_table candidates
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // in parallel only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (!candidates)
    {
        if (rank == 0)
        {
            TECA_ERROR("empty input or not a table")
        }
        return nullptr;
    }

    // get the candidate storm properties
    const long *p_step =
        dynamic_cast<const teca_variant_array_impl<long>*>(
        candidates->get_column("step").get())->get();

    const double *p_time =
        dynamic_cast<const teca_variant_array_impl<double>*>(
        candidates->get_column("time").get())->get();

    const int *p_storm_id =
        dynamic_cast<const teca_variant_impl<int>*>(
        candidates->get_column("storm_id").get())->get();

    const_p_teca_variant_array lon = candidates->get_column("lon");
    const_p_teca_variant_array lat = candidates->get_column("lat");

    const_p_teca_variant_array wind_max =
        candidates->get_column("surface_wind_max");

    const_p_teca_variant_array vort_max =
        candidates->get_column("850mb_vorticty_max");

    const_p_teca_variant_array psl_min =
        candidates->get_column("seal_level_pressure_min");

    const int *p_have_twc =
        dynamic_cast<const teca_variant_impl<int>*>(
        candidates->get_column("have_core_temp").get())->get();

    const int *p_have_thick =
        dynamic_cast<const teca_variant_impl<int>*>(
        candidates->get_column("have_thickness").get())->get();

    const_p_teca_variant_array twc_max =
        candidates->get_column("core_temp_max");

    const_p_teca_variant_array thick_max =
        candidates->get_column("thickness_max");

    // create the table to hold storm tracks
    p_teca_table storm_tracks = teca_table::New();

    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        lon.get(), _COORD,

        const NT_COORD *p_lon = static_cast<const TT_COORD*>(x.get())->get();
        const NT_COORD *p_lat = static_cast<const TT_COORD*>(y.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            wind_max.get(), _VAR,

            // configure the output
            strom_tracks->declare_columns("track_id", int(), "step", long(),
                "storm_id", int(), "lon", NT_COORD(), "lat", NT_COORD(),
                "surface_wind_max", NT_VAR(), "850mb_vorticity_max", NT_VAR(),
                "sea_level_pressure_min", NT_VAR(), "have_core_temp", int(),
                "have_thickness", int(), "core_temp_max", NT_VAR(),
                "thickness_max", NT_VAR());

            const NT_VAR *p_wind_max =
                dynamic_cast<const TT_VAR*>(wind_max.get())->get();

            const NT_VAR *p_vort_max =
                dynamic_cast<const TT_VAR*>(vort_max.get())->get();

            const NT_VAR *p_psl_min =
                dynamic_cast<const TT_VAR*>(psl_min.get())->get();

            const NT_VAR *p_twc_max =
                dynamic_cast<const TT_VAR*>(twc_max.get())->get();

            const NT_VAR *p_thick_max =
                dynamic_cast<const TT_VAR*>(thick_max.get())->get();

            // invoke the track finder
            if (teca_gfdl::tc_trajectory(this->max_daily_distance,
                this->min_wind_speed, this->min_wind_duration, p_step, p_time,
                p_storm_id, p_lon, p_lat, p_wind_max, p_vort_max, p_psl_min,
                p_have_twc, p_have_thick, p_twc_max, p_thick_max,
                candidates->get_number_of_rows(), strom_tracks))
            {
                TECA_ERROR("GFDL TC trajectory analysis encountered an error")
                return nullptr;
            }
            )
        )

    return storm_tracks;
}
