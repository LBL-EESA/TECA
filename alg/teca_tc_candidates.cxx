#include "teca_tc_candidates.h"

#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_table.h"
#include "teca_database.h"
#include "teca_calendar.h"
#include "teca_coordinate_util.h"
#include "gfdl_tc_candidates.h"

#include <iostream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <chrono>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

//#define TECA_DEBUG 2

using std::cerr;
using std::endl;
using seconds_t = std::chrono::duration<double, std::chrono::seconds::period>;

// --------------------------------------------------------------------------
teca_tc_candidates::teca_tc_candidates() :
    max_core_radius(2.0),
    min_vorticity_850mb(1.6e-4),
    vorticity_850mb_window(7.75),
    max_pressure_delta(400.0),
    max_pressure_radius(5.0),
    max_core_temperature_delta(0.8),
    max_core_temperature_radius(5.0),
    max_thickness_delta(50.0),
    max_thickness_radius(4.0),
    search_lat_low(1.0),
    search_lat_high(0.0),
    search_lon_low(1.0),
    search_lon_high(0.0),
    minimizer_iterations(50)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_tc_candidates::~teca_tc_candidates()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_tc_candidates::get_properties_description(
    const std::string &prefix, options_description &opts)
{
    options_description ard_opts("Options for "
        + (prefix.empty()?"teca_tc_candidates":prefix));

    ard_opts.add_options()
        TECA_POPTS_GET(std::string, prefix, surface_wind_speed_variable,
            "name of wind speed variable")
        TECA_POPTS_GET(std::string, prefix, vorticity_850mb_variable,
            "name of 850 mb vorticity variable")
        TECA_POPTS_GET(std::string, prefix, sea_level_pressure_variable,
            "name of sea level pressure variable")
        TECA_POPTS_GET(std::string, prefix, core_temperature_variable,
            "name of core temperature variable")
        TECA_POPTS_GET(double, prefix, max_core_radius,
            "maximum number of degrees latitude separation between "
            "vorticity max and pressure min defining a storm (2.0)")
        TECA_POPTS_GET(double, prefix, min_vorticity_850mb,
            "minimum vorticty to be considered a tropical storm (1.6e-4)")
        TECA_POPTS_GET(double, prefix, vorticity_850mb_window,
            "size of the search window in degrees. storms core must have a "
            "local vorticity max centered on this window (7.74446)")
        TECA_POPTS_GET(double, prefix, max_pressure_delta,
            "maximum pressure change within specified radius (400.0)")
        TECA_POPTS_GET(double, prefix, max_pressure_radius,
            "radius in degrees over which max pressure change is computed (5.0)")
        TECA_POPTS_GET(double, prefix, max_core_temperature_delta,
            "maximum core temperature change over the specified radius (0.8)")
        TECA_POPTS_GET(double, prefix, max_core_temperature_radius,
            "radius in degrees over which max core temperature change is computed (5.0)")
        TECA_POPTS_GET(double, prefix, max_thickness_delta,
            "maximum thickness change over the specified radius (50.0)")
        TECA_POPTS_GET(double, prefix, max_thickness_radius,
            "radius in degrees over with max thickness change is comuted (4.0)")
        TECA_POPTS_GET(double, prefix, search_lat_low,
            "lowest latitude in degrees to search for storms (-80.0)")
        TECA_POPTS_GET(double, prefix, search_lat_high,
            "highest latitude in degrees to search for storms (80.0)")
        TECA_POPTS_GET(double, prefix, search_lon_low,
            "lowest longitude in degrees to search for stroms (1)")
        TECA_POPTS_GET(double, prefix, search_lon_high,
            "highest longitude in degrees to search for storms (0)")
        ;

    opts.add(ard_opts);
}

// --------------------------------------------------------------------------
void teca_tc_candidates::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, surface_wind_speed_variable)
    TECA_POPTS_SET(opts, std::string, prefix, vorticity_850mb_variable)
    TECA_POPTS_SET(opts, std::string, prefix, sea_level_pressure_variable)
    TECA_POPTS_SET(opts, std::string, prefix, core_temperature_variable)
    TECA_POPTS_SET(opts, double, prefix, max_core_radius)
    TECA_POPTS_SET(opts, double, prefix, min_vorticity_850mb)
    TECA_POPTS_SET(opts, double, prefix, vorticity_850mb_window)
    TECA_POPTS_SET(opts, double, prefix, max_pressure_delta)
    TECA_POPTS_SET(opts, double, prefix, max_pressure_radius)
    TECA_POPTS_SET(opts, double, prefix, max_core_temperature_delta)
    TECA_POPTS_SET(opts, double, prefix, max_core_temperature_radius)
    TECA_POPTS_SET(opts, double, prefix, max_thickness_delta)
    TECA_POPTS_SET(opts, double, prefix, max_thickness_radius)
    TECA_POPTS_SET(opts, double, prefix, search_lat_high)
    TECA_POPTS_SET(opts, double, prefix, search_lat_low)
    TECA_POPTS_SET(opts, double, prefix, search_lon_high)
    TECA_POPTS_SET(opts, double, prefix, search_lon_low)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_tc_candidates::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id()
        << "teca_tc_candidates::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata output_md(input_md[0]);
    return output_md;
}

// --------------------------------------------------------------------------
int teca_tc_candidates::get_active_extent(
    p_teca_variant_array lat,
    p_teca_variant_array lon,
    std::vector<unsigned long> &extent) const
{
    extent = {1, 0, 1, 0, 0, 0};

    unsigned long high_i = lon->size() - 1;
    if (this->search_lon_low > this->search_lon_high)
    {
        extent[0] = 0l;
        extent[1] = high_i;
    }
    else
    {
        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            lon.get(),
            NT *p_lon = std::dynamic_pointer_cast<TT>(lon)->get();

            if (teca_coordinate_util::index_of(p_lon, 0, high_i, static_cast<NT>(this->search_lon_low), false, extent[0])
                || teca_coordinate_util::index_of(p_lon, 0, high_i, static_cast<NT>(this->search_lon_high), true, extent[1]))
            {
                TECA_ERROR(
                    << "requested longitude ["
                    << this->search_lon_low << ", " << this->search_lon_high << ", "
                    << "] is not contained in the current dataset bounds ["
                    << p_lon[0] << ", " << p_lon[high_i] << "]")
                return -1;
            }
            )

    }
    if (extent[0] > extent[1])
    {
        TECA_ERROR("invalid longitude coordinate array type")
        return -1;
    }

    unsigned long high_j = lat->size() - 1;
    if (this->search_lat_low > this->search_lat_high)
    {
        extent[2] = 0l;
        extent[3] = high_j;
    }
    else
    {
        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            lat.get(),
            NT *p_lat = std::dynamic_pointer_cast<TT>(lat)->get();

            if (teca_coordinate_util::index_of(p_lat, 0, high_j, static_cast<NT>(this->search_lat_low), false, extent[2])
                || teca_coordinate_util::index_of(p_lat, 0, high_j, static_cast<NT>(this->search_lat_high), true, extent[3]))
            {
                TECA_ERROR(
                    << "requested latitude ["
                    << this->search_lat_low << ", " << this->search_lat_high
                    << "] is not contained in the current dataset bounds ["
                    << p_lat[0] << ", " << p_lat[high_j] << "]")
                return -1;
            }
            )

    }
    if (extent[2] > extent[3])
    {
        TECA_ERROR("invalid latitude coordinate array type")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_tc_candidates::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id()
        << "teca_tc_candidates::get_upstream_request" << endl;
#endif
    (void)port;

    std::vector<teca_metadata> up_reqs;
    teca_metadata md = input_md[0];

    // locate the extents of the user supplied region of
    // interest
    teca_metadata coords;
    if (md.get("coordinates", coords))
    {
        TECA_ERROR("metadata is missing \"coordinates\"")
        return up_reqs;
    }

    p_teca_variant_array lat;
    p_teca_variant_array lon;
    if (!(lat = coords.get("y")) || !(lon = coords.get("x")))
    {
        TECA_ERROR("metadata missing lat lon coordinates")
        return up_reqs;
    }

    std::vector<unsigned long> extent(6, 0l);
    if (this->get_active_extent(lat, lon, extent))
    {
        TECA_ERROR("failed to determine the active extent")
        return up_reqs;
    }

#if TECA_DEBUG > 1
    cerr << teca_parallel_id() << "active_bound = "
        << this->search_lon_low<< ", " << this->search_lon_high
        << ", " << this->search_lat_low << ", " << this->search_lat_high
        << endl;
    cerr << teca_parallel_id() << "active_extent = "
        << extent[0] << ", " << extent[1] << ", " << extent[2] << ", "
        << extent[3] << ", " << extent[4] << ", " << extent[5] << endl;
#endif

    // build the request
    std::set<std::string> arrays;
    request.get("arrays", arrays);
    arrays.insert(this->surface_wind_speed_variable);
    arrays.insert(this->vorticity_850mb_variable);
    arrays.insert(this->sea_level_pressure_variable);
    arrays.insert(this->core_temperature_variable);
    arrays.insert(this->thickness_variable);

    teca_metadata up_req(request);
    up_req.insert("arrays", arrays);
    up_req.insert("extent", extent);

    up_reqs.push_back(up_req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_tc_candidates::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id() << "teca_tc_candidates::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    if (!input_data.size())
    {
        TECA_ERROR("empty input")
        return nullptr;
    }

    // get the input dataset
    const_p_teca_cartesian_mesh mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);
    if (!mesh)
    {
        TECA_ERROR("teca_cartesian_mesh is required")
        return nullptr;
    }

    // get coordinate arrays
    const_p_teca_variant_array y = mesh->get_y_coordinates();
    const_p_teca_variant_array x = mesh->get_x_coordinates();

    if (!x || !y)
    {
        TECA_ERROR("mesh coordinates are missing.")
        return nullptr;
    }

    // get time step
    unsigned long time_step;
    mesh->get_time_step(time_step);

    // get temporal offset of the current timestep
    double time_offset = 0.0;
    mesh->get_time(time_offset);

    // get offset units
    std::string time_units;
    mesh->get_time_units(time_units);

    // get offset calendar
    std::string calendar;
    mesh->get_calendar(calendar);

    // get extent of data passed in
    std::vector<unsigned long> extent;
    mesh->get_extent(extent);

    long nlat = extent[3] - extent[2] + 1;
    long nlon = extent[1] - extent[0] + 1;

    // get wind speed array
    const_p_teca_variant_array surface_wind_speed
        = mesh->get_point_arrays()->get(this->surface_wind_speed_variable);

    if (!surface_wind_speed)
    {
        TECA_ERROR("Dataset missing wind speed variable \""
            << this->surface_wind_speed_variable << "\"")
        return nullptr;
    }

    // get vorticity_850mb array
    const_p_teca_variant_array vorticity_850mb
        = mesh->get_point_arrays()->get(this->vorticity_850mb_variable);

    if (!vorticity_850mb)
    {
        TECA_ERROR("Dataset missing vorticity_850mb variable \""
            << this->vorticity_850mb_variable << "\"")
        return nullptr;
    }

    // get core_temperature array
    const_p_teca_variant_array core_temperature
        = mesh->get_point_arrays()->get(this->core_temperature_variable);

    if (!core_temperature)
    {
        TECA_ERROR("Dataset missing core_temperature variable \""
            << this->core_temperature_variable << "\"")
        return nullptr;
    }

    // get sea_level_pressure array
    const_p_teca_variant_array sea_level_pressure
        = mesh->get_point_arrays()->get(this->sea_level_pressure_variable);

    if (!sea_level_pressure)
    {
        TECA_ERROR("Dataset missing sea_level_pressure variable \""
            << this->sea_level_pressure_variable << "\"")
        return nullptr;
    }

    // get thickness array
    const_p_teca_variant_array thickness;
    if (!(thickness = mesh->get_point_arrays()->get(this->thickness_variable)))
    {
        TECA_ERROR("Dataset missing thickness variable \""
            << this->thickness_variable << "\"")
        return nullptr;
    }

    // identify candidates
    p_teca_table candidates = teca_table::New();

    std::chrono::high_resolution_clock::time_point t0, t1;

    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        x.get(), _COORD,

        const NT_COORD *lon = static_cast<const TT_COORD*>(x.get())->get();
        const NT_COORD *lat = static_cast<const TT_COORD*>(y.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            surface_wind_speed.get(), _VAR,

            // configure the candidate table
            candidates->declare_columns("storm_id", int(),
                "lon", NT_COORD(), "lat", NT_COORD(), "surface_wind", NT_VAR(),
                "850mb_vorticity", NT_VAR(), "sea_level_pressure", NT_VAR(),
                "have_core_temp", int(), "have_thickness", int(),
                "core_temp", NT_VAR(), "thickness", NT_VAR());

            const NT_VAR *v = dynamic_cast<const TT_VAR*>(surface_wind_speed.get())->get();
            const NT_VAR *w = dynamic_cast<const TT_VAR*>(vorticity_850mb.get())->get();
            const NT_VAR *P = dynamic_cast<const TT_VAR*>(sea_level_pressure.get())->get();
            const NT_VAR *T = dynamic_cast<const TT_VAR*>(core_temperature.get())->get();
            const NT_VAR *th = dynamic_cast<const TT_VAR*>(thickness.get())->get();

            t0 = std::chrono::high_resolution_clock::now();
            // invoke the detector
            if (teca_gfdl::tc_candidates(this->max_core_radius,
                this->min_vorticity_850mb, this->vorticity_850mb_window,
                this->max_pressure_delta, this->max_pressure_radius,
                this->max_core_temperature_delta, this->max_core_temperature_radius,
                this->max_thickness_delta, this->max_thickness_radius, v, w,
                T, P, th, lat, lon, nlat, nlon, this->minimizer_iterations,
                time_step, candidates.get()))
            {
                TECA_ERROR("GFDL TC detector encountered an error")
                return nullptr;
            }
            t1 = std::chrono::high_resolution_clock::now();
            )
        )

    // build the output
    p_teca_table out_table = teca_table::New();
    out_table->set_calendar(calendar);
    out_table->set_time_units(time_units);

    // add time stamp
    out_table->declare_columns("step", long(), "time", double());
    unsigned long n_candidates = candidates->get_number_of_rows();
    for (unsigned long i = 0; i < n_candidates; ++i)
        out_table << time_step << time_offset;

    // add the candidates
    out_table->concatenate_cols(candidates);

#if TECA_DEBUG > 1
    out_table->to_stream(cerr);
    cerr << std::endl;
#endif
    seconds_t dt(t1 - t0);
    TECA_STATUS("teca_tc_candidates step=" << time_step
        << " t=" << time_offset << ", dt=" << dt.count() << " sec")

    return out_table;
}

// --------------------------------------------------------------------------
void teca_tc_candidates::to_stream(std::ostream &os) const
{
    (void)os;
}
