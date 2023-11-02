#include "teca_tc_classify.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_distance_function.h"
#include "teca_saffir_simpson.h"
#include "teca_geometry.h"
#include "teca_geography.h"

#include <iostream>
#include <string>
#include <sstream>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif
#if defined(TECA_HAS_UDUNITS)
#include "teca_calcalcs.h"
#endif
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::cerr;
using std::endl;

using namespace teca_variant_array_util;

// --------------------------------------------------------------------------
teca_tc_classify::teca_tc_classify() :
    track_id_column("track_id"), time_column("time"), x_coordinate_column("lon"),
    y_coordinate_column("lat"), surface_wind_column("surface_wind"),
    sea_level_pressure_column("sea_level_pressure")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    teca_geography::get_cyclone_basins(this->region_sizes,
        this->region_starts, this->region_x_coordinates,
        this->region_y_coordinates, this->region_ids,
        this->region_names, this->region_long_names);
}

// --------------------------------------------------------------------------
teca_tc_classify::~teca_tc_classify()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_tc_classify::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_tc_classify":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, track_id_column,
            "name of the column containing track ids")
        TECA_POPTS_GET(std::string, prefix, time_column,
            "name of the column containing time stamps")
        TECA_POPTS_GET(std::string, prefix, surface_wind_column,
            "name of column containing wind speeds")
        TECA_POPTS_GET(std::string, prefix, x_coordinate_column,
            "name of the column containing x cooridnates")
        TECA_POPTS_GET(std::string, prefix, y_coordinate_column,
            "name of the column containing y cooridnates")
        TECA_POPTS_MULTI_GET(std::vector<unsigned long>,
            prefix, region_sizes, "the number of points in each region")
        TECA_POPTS_MULTI_GET(std::vector<double>,
            prefix, region_x_coordinates, "list of x coordinates describing the regions")
        TECA_POPTS_MULTI_GET(std::vector<double>,
            prefix, region_y_coordinates, "list of y coordinates describing the regions")
        TECA_POPTS_MULTI_GET(std::vector<int>,
            prefix, region_ids, "list of numeric ids identifying each region. "
            " if not provided sequential ids are generated")
        TECA_POPTS_MULTI_GET(std::vector<std::string>,
            prefix, region_names, "list of names identifying each region. "
            "if not provided names are generated from ids")
        TECA_POPTS_MULTI_GET(std::vector<std::string>,
            prefix, region_long_names, "list of long/readable names identifying "
            "each region. if not provided names are generated from ids")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_tc_classify::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, track_id_column)
    TECA_POPTS_SET(opts, std::string, prefix, time_column)
    TECA_POPTS_SET(opts, std::string, prefix, surface_wind_column)
    TECA_POPTS_SET(opts, std::string, prefix, x_coordinate_column)
    TECA_POPTS_SET(opts, std::string, prefix, y_coordinate_column)
    TECA_POPTS_SET(opts, std::vector<unsigned long>, prefix, region_sizes)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, region_x_coordinates)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, region_y_coordinates)
    TECA_POPTS_SET(opts, std::vector<int>, prefix, region_ids)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, region_names)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, region_long_names)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_tc_classify::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_tc_classify::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input table
    const_p_teca_table in_table
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif
    if (!in_table)
    {
        if (rank == 0)
        {
            TECA_FATAL_ERROR("Input is empty or not a table")
        }
        return nullptr;
    }

    // get calendar and unit system
    std::string calendar;
    if ((in_table->get_calendar(calendar)) && calendar.empty())
    {
        TECA_FATAL_ERROR("Calendar is missing")
        return nullptr;
    }

    std::string time_units;
    if ((in_table->get_time_units(time_units)) && time_units.empty())
    {
        TECA_FATAL_ERROR("time units are missing")
        return nullptr;
    }

    if (time_units.find("days since") == std::string::npos)
    {
        TECA_FATAL_ERROR("Conversion for \"" << time_units << "\" not implemented")
        return nullptr;
    }

    // get the track ids
    const_p_teca_int_array track_ids =
        std::dynamic_pointer_cast<const teca_int_array>(
            in_table->get_column(this->track_id_column));

    if (!track_ids)
    {
        TECA_FATAL_ERROR("column \"" << this->track_id_column
            << "\" is not in the table")
        return nullptr;
    }

    // the spatial coorinates
    const_p_teca_variant_array x =
        in_table->get_column(this->x_coordinate_column);

    if (!x)
    {
        TECA_FATAL_ERROR("column \"" << this->x_coordinate_column
            << "\" is not in the table")
        return nullptr;
    }

    const_p_teca_variant_array y =
        in_table->get_column(this->y_coordinate_column);

    if (!y)
    {
        TECA_FATAL_ERROR("column \"" << this->y_coordinate_column
            << "\" is not in the table")
        return nullptr;
    }

    // time axis
    const_p_teca_variant_array time =
        in_table->get_column(this->time_column);

    if (!time)
    {
        TECA_FATAL_ERROR("column \"" << this->time_column
            << "\" is not in the table")
        return nullptr;
    }

    // get the surface wind speeds
    const_p_teca_variant_array surface_wind =
        in_table->get_column(this->surface_wind_column);

    if (!surface_wind)
    {
        TECA_FATAL_ERROR("column \"" << this->surface_wind_column
            << "\" is not in the table")
        return nullptr;
    }

    // get the surface wind speeds
    const_p_teca_variant_array sea_level_pressure =
        in_table->get_column(this->sea_level_pressure_column);

    if (!sea_level_pressure)
    {
        TECA_FATAL_ERROR("column \"" << this->sea_level_pressure_column
            << "\" is not in the table")
        return nullptr;
    }

    // scan the track ids and build the random access
    // data structure
    std::vector<unsigned long> track_starts(1, 0);
    size_t n_rows = track_ids->size();

    auto [spids, pids] = get_host_accessible<teca_int_array>(track_ids);

    sync_host_access_any(track_ids);

    for (size_t i = 1; i < n_rows; ++i)
        if (pids[i] != pids[i-1])
            track_starts.push_back(i);

    track_starts.push_back(n_rows);
    size_t n_tracks = track_starts.size() - 1;

    // record track id
    auto [out_ids, pout_ids] = ::New<teca_long_array>(n_tracks);

    for (size_t i =0; i < n_tracks; ++i)
        pout_ids[i] = pids[track_starts[i]];

    // record track start time
    p_teca_variant_array start_time = time->new_instance(n_tracks);
    VARIANT_ARRAY_DISPATCH(time.get(),

        auto [pstart_time] = data<TT>(start_time);
        auto [sptime, ptime] = get_host_accessible<CTT>(time);

        sync_host_access_any(time);

        for (size_t i = 0; i < n_tracks; ++i)
            pstart_time[i] = ptime[track_starts[i]];
        )

    // record track start position
    p_teca_variant_array start_x = x->new_instance(n_tracks);
    p_teca_variant_array start_y = x->new_instance(n_tracks);

    VARIANT_ARRAY_DISPATCH_FP(x.get(),

        assert_type<CTT>(y);
        auto [spx, px, spy, py] = get_host_accessible<CTT>(x, y);
        auto [pstart_x, pstart_y] = data<TT>(start_x, start_y);

        sync_host_access_any(x, y);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = track_starts[i];

            pstart_x[i] = px[q];
            pstart_y[i] = py[q];
        }
        )

    // compute the storm duration
    p_teca_variant_array duration = time->new_instance(n_tracks);
    VARIANT_ARRAY_DISPATCH(time.get(),

        auto [sptime, ptime] = get_host_accessible<CTT>(time);
        auto [pduration] = data<TT>(duration);

        sync_host_access_any(time);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long first = track_starts[i];
            unsigned long last = track_starts[i+1] - 1;
            pduration[i] = ptime[last] - ptime[first];
        }
        )

    // compute the distance traveled
    p_teca_variant_array length = x->new_instance(n_tracks);

    VARIANT_ARRAY_DISPATCH_FP(x.get(),

        assert_type<CTT>(y);
        auto [spx, px, spy, py] = get_host_accessible<CTT>(x, y);
        auto [plength] = data<TT>(length);

        sync_host_access_any(x, y);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long track_start = track_starts[i];
            unsigned long npts = track_starts[i+1] - track_start - 1;

            plength[i] = NT();

            for (unsigned long j = 0; j < npts; ++j)
            {
                unsigned long p = track_start + j;
                unsigned long q = p + 1;

                plength[i] += teca_distance(px[p], py[p], px[q], py[q]);
            }
        }
        )

    // rank the track on Saphir-Simpson scale
    // record the max wind speed, and position of it
    auto [category, pcategory] = ::New<teca_int_array>(n_tracks);

    auto [max_surface_wind_id,
          pmax_surface_wind_id] = ::New<teca_unsigned_long_array>(n_tracks);

    p_teca_variant_array max_surface_wind;

    VARIANT_ARRAY_DISPATCH_FP(surface_wind.get(),

        NT *pmax_surface_wind = nullptr;
        std::tie(max_surface_wind, pmax_surface_wind) = ::New<TT>(n_tracks);

        auto [spsw, psurface_wind] = get_host_accessible<CTT>(surface_wind);

        sync_host_access_any(surface_wind);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long track_start = track_starts[i];
            unsigned long npts = track_starts[i+1] - track_start;

            NT max_val = std::numeric_limits<NT>::lowest();
            unsigned long max_id = 0;

            for (size_t j = 0; j < npts; ++j)
            {
                unsigned long id = track_start + j;
                NT val = psurface_wind[id];
                bool max_changed = val > max_val;
                max_val = max_changed ? val : max_val;
                max_id = max_changed ? id : max_id;
            }

            pcategory[i] = teca_saffir_simpson::classify_mps(max_val);
            pmax_surface_wind[i] = max_val;
            pmax_surface_wind_id[i] = max_id;
        }
        )

    // location of the max surface wind
    p_teca_variant_array max_surface_wind_x = x->new_instance(n_tracks);
    p_teca_variant_array max_surface_wind_y = x->new_instance(n_tracks);
    VARIANT_ARRAY_DISPATCH_FP(x.get(),

        assert_type<CTT>(y);
        auto [spx, px, spy, py] = get_host_accessible<CTT>(x, y);

        auto [pmax_surface_wind_x,
              pmax_surface_wind_y] = data<TT>(max_surface_wind_x,
                                              max_surface_wind_y);
        sync_host_access_any(x, y);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmax_surface_wind_id[i];
            pmax_surface_wind_x[i] = px[q];
            pmax_surface_wind_y[i] = py[q];
        }
        )

    // time of max surface wind
    p_teca_variant_array max_surface_wind_t = time->new_instance(n_tracks);
    VARIANT_ARRAY_DISPATCH(time.get(),

        auto [sptime, ptime] = get_host_accessible<CTT>(time);
        auto [pmax_surface_wind_t] = data<TT>(max_surface_wind_t);

        sync_host_access_any(time);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmax_surface_wind_id[i];
            pmax_surface_wind_t[i] = ptime[q];
        }
        )

    // record the min sea level pressure
    auto min_sea_level_pressure = sea_level_pressure->new_instance(n_tracks);

    auto [min_sea_level_pressure_id,
          pmin_sea_level_pressure_id] = ::New<teca_unsigned_long_array>(n_tracks);

    VARIANT_ARRAY_DISPATCH_FP(sea_level_pressure.get(),

        auto [spsea_level_pressure,
              psea_level_pressure] = get_host_accessible<CTT>(sea_level_pressure);

        auto [pmin_sea_level_pressure] = data<TT>(min_sea_level_pressure);

        sync_host_access_any(sea_level_pressure);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long track_start = track_starts[i];
            unsigned long npts = track_starts[i+1] - track_start;

            NT min_val = std::numeric_limits<NT>::max();
            unsigned long min_id = 0;

            for (size_t j = 0; j < npts; ++j)
            {
                unsigned long q = track_start + j;
                NT val = psea_level_pressure[q];
                bool min_changed = val < min_val;
                min_val = min_changed ? val : min_val;
                min_id = min_changed ? q : min_id;
            }

            pmin_sea_level_pressure[i] = min_val;
            pmin_sea_level_pressure_id[i] = min_id;
        }
        )

    // location of the min sea level pressure
    p_teca_variant_array min_sea_level_pressure_x = x->new_instance(n_tracks);
    p_teca_variant_array min_sea_level_pressure_y = x->new_instance(n_tracks);
    VARIANT_ARRAY_DISPATCH_FP(x.get(),

        assert_type<CTT>(y);
        auto [spx, px, spy, py] = get_host_accessible<CTT>(x, y);

        auto [pmin_sea_level_pressure_x,
              pmin_sea_level_pressure_y] = data<TT>(min_sea_level_pressure_x,
                                                    min_sea_level_pressure_y);

        sync_host_access_any(x, y);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmin_sea_level_pressure_id[i];
            pmin_sea_level_pressure_x[i] = px[q];
            pmin_sea_level_pressure_y[i] = py[q];
        }
        )

    // time of min sea level pressure
    p_teca_variant_array min_sea_level_pressure_t = time->new_instance(n_tracks);
    VARIANT_ARRAY_DISPATCH(time.get(),

        auto [sptime, ptime] = get_host_accessible<CTT>(time);
        auto [pmin_sea_level_pressure_t] = data<TT>(min_sea_level_pressure_t);

        sync_host_access_any(time);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmin_sea_level_pressure_id[i];
            pmin_sea_level_pressure_t[i] = ptime[q];
        }
        )

    // ACE (accumulated cyclonigc energy)
    // The ACE of a season is calculated by summing the squares of the
    // estimated maximum sustained velocity of every active tropical storm
    // (wind speed 35 knots (65 km/h) or higher), at six-hour intervals. Since
    // the calculation is sensitive to the starting point of the six-hour
    // intervals, the convention is to use 0000, 0600, 1200, and 1800 UTC. If
    // any storms of a season happen to cross years, the storm's ACE counts for
    // the previous year.[2] The numbers are usually divided by 10,000 to make
    // them more manageable. The unit of ACE is 10^4 kn^2, and for use as an
    // index the unit is assumed. Thus:
    // {\displaystyle {\text{ACE}}=10^{-4}\sum v_{\max }^{2}}
    // {\text{ACE}}=10^{{-4}}\sum v_{\max }^{2} where vmax is estimated
    // sustained wind speed in knots.
    p_teca_variant_array ACE = surface_wind->new_instance(n_tracks);

    NESTED_VARIANT_ARRAY_DISPATCH_FP(
        time.get(), _T,

        auto [sptime, ptime] = get_host_accessible<CTT_T>(time);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            ACE.get(), _W,

            auto [spsw, psurface_wind] = get_host_accessible<CTT_W>(surface_wind);
            auto [pACE] = data<TT_W>(ACE);

            sync_host_access_any(time, surface_wind);

            for (size_t i = 0; i < n_tracks; ++i)
            {
                unsigned long track_start = track_starts[i];
                unsigned long npts = track_starts[i+1] - track_start - 1;

                pACE[i] = NT_W();

                // for now skip the first and last track point
                // could handle these as a special case if needed
                for (size_t j = 1; j < npts; ++j)
                {
                    unsigned long id = track_start + j;
                    NT_W dt = ptime[id+1] - ptime[id-1];
                    NT_W w = psurface_wind[id];
                    pACE[i] += w < teca_saffir_simpson::get_lower_bound_mps<NT_W>(0)
                        ? NT_W() : w*w*dt;
                }

                // correct the units
                // wind speed conversion : 1 m/s = 1.943844 kn
                // time unit conversions: 24 hours per day, 6 hours per time unit,
                // and we sample time in days at t +/- 1/2 => dt*24/2/6 => dt*2
                // by convention scale by 10^-4
                pACE[i] *= NT_W(2.0)*NT_W(3.778529496)*NT_W(1.0e-4);
            }
            )
        )

    // PDI (power dissipation index)
    // PDI = \sum v_{max}^{3} \delta t
    // see: Environmental Factors Affecting Tropical Cyclone Power Dissipation
    // KERRY EMANUEL, 15 NOVEMBER 2007, JOURNAL OF CLIMATE
    p_teca_variant_array PDI = surface_wind->new_instance(n_tracks);

    NESTED_VARIANT_ARRAY_DISPATCH_FP(
        time.get(), _T,

        auto [sptime, ptime] = get_host_accessible<CTT_T>(time);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            PDI.get(), _W,

            auto [spsw, psurface_wind] = get_host_accessible<CTT_W>(surface_wind);
            auto [pPDI] = data<TT_W>(PDI);

            sync_host_access_any(time, surface_wind);

            for (size_t i = 0; i < n_tracks; ++i)
            {
                unsigned long track_start = track_starts[i];
                unsigned long npts = track_starts[i+1] - track_start - 1;

                pPDI[i] = NT_W();

                // for now skip the first and last track point
                // could handle these as a special case if needed
                for (size_t j = 1; j < npts; ++j)
                {
                    unsigned long id = track_start + j;
                    NT_W dt = ptime[id+1] - ptime[id-1];
                    NT_W w = psurface_wind[id];
                    pPDI[i] += w < teca_saffir_simpson::get_lower_bound_mps<NT_W>(0)
                        ? NT_W() : w*w*w*dt;
                }

                // correct the units
                // time unit conversions: 24*3600 seconds per day
                // and we sample time in days at t +/- 1 => dt*24*3600/2
                pPDI[i] *= NT_W(43200);
            }
            )
        )

    // cyclogenisis, determine region of origin
    size_t n_regions = this->region_sizes.size();

    std::vector<unsigned long> rstarts(this->region_starts);
    if (rstarts.empty())
    {
        // generate starts
        rstarts.reserve(n_regions);
        rstarts.push_back(0);
        for (size_t  i = 0; i < n_regions; ++i)
            rstarts.push_back(rstarts[i] + this->region_sizes[i]);
    }

    std::vector<int> rids(this->region_ids);
    if (rids.empty())
    {
        // generate ids
        rids.reserve(n_regions);
        for (size_t  i = 0; i < n_regions; ++i)
            rids.push_back(i);
    }

    std::vector<std::string> rnames(this->region_names);
    if (rnames.empty())
    {
        // generate names
        std::ostringstream oss;
        rnames.reserve(n_regions);
        for (size_t  i = 0; i < n_regions; ++i)
        {
            oss.str("");
            oss << "r" << i;
            rnames.push_back(oss.str());
        }
    }

    std::vector<std::string> rlnames(this->region_long_names);
    if (rnames.empty())
    {
        // generate names
        std::ostringstream oss;
        rnames.reserve(n_regions);
        for (size_t  i = 0; i < n_regions; ++i)
        {
            oss.str("");
            oss << "region_" << i;
            rlnames.push_back(oss.str());
        }
    }

    auto [region_id, pregion_id] = ::New<teca_int_array>(n_tracks, -1);
    auto [region_name, pregion_name] = ::New<teca_string_array>(n_tracks);
    auto [region_long_name, pregion_long_name] = ::New<teca_string_array>(n_tracks);

    VARIANT_ARRAY_DISPATCH_FP(x.get(),

        assert_type<CTT>(y);
        auto [spx, px, spy, py] = get_host_accessible<CTT>(x, y);

        sync_host_access_any(x, y);

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = track_starts[i];

            double ptx = px[q];
            double pty = py[q];

            for (size_t j = 0; j < n_regions; ++j)
            {
                double *polyx = this->region_x_coordinates.data() + rstarts[j];
                double *polyy = this->region_y_coordinates.data() + rstarts[j];
                if (teca_geometry::point_in_poly(ptx, pty, polyx, polyy, this->region_sizes[j]))
                {
                    unsigned long rid = rids[j];
                    pregion_id[i] = rid;
                    pregion_name[i] = rnames[rid];
                    pregion_long_name[i] = rlnames[rid];
                    // early termination precludes a storm from being counted in
                    // multiple regions. if we want to allow overlapping regions
                    // this would need to change.
                    break;
                }
            }

            if (pregion_id[i] < 0)
            {
                TECA_WARNING("track " << i << " is not any of the regions!")
            }
        }
        )

    // construct the output
    p_teca_table out_table = teca_table::New();
    out_table->copy_metadata(in_table);

    out_table->append_column("track_id", out_ids);
    out_table->append_column("start_time", start_time);
    out_table->append_column("start_x", start_x);
    out_table->append_column("start_y", start_y);
    out_table->append_column("duration", duration);
    out_table->append_column("length", length);
    out_table->append_column("category", category);
    out_table->append_column("ACE", ACE);
    out_table->append_column("PDI", PDI);
    out_table->append_column("max_surface_wind", max_surface_wind);
    out_table->append_column("max_surface_wind_x", max_surface_wind_x);
    out_table->append_column("max_surface_wind_y", max_surface_wind_y);
    out_table->append_column("max_surface_wind_t", max_surface_wind_t);
    out_table->append_column("min_sea_level_pressure", min_sea_level_pressure);
    out_table->append_column("min_sea_level_pressure_x", min_sea_level_pressure_x);
    out_table->append_column("min_sea_level_pressure_y", min_sea_level_pressure_y);
    out_table->append_column("min_sea_level_pressure_t", min_sea_level_pressure_t);
    out_table->append_column("region_id", region_id);
    out_table->append_column("region_name", region_name);
    out_table->append_column("region_long_name", region_long_name);

    return out_table;
}
