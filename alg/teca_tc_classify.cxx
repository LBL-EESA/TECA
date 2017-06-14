#include "teca_tc_classify.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
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
#include "calcalcs.h"
#endif
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::cerr;
using std::endl;

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

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_tc_classify::set_properties(
    const std::string &prefix, variables_map &opts)
{
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
            TECA_ERROR("Input is empty or not a table")
        }
        return nullptr;
    }

    // get calendar and unit system
    std::string calendar;
    if ((in_table->get_calendar(calendar)) && calendar.empty())
    {
        TECA_ERROR("Calendar is missing")
        return nullptr;
    }

    std::string time_units;
    if ((in_table->get_time_units(time_units)) && time_units.empty())
    {
        TECA_ERROR("time units are missing")
        return nullptr;
    }

    if (time_units.find("days since") == std::string::npos)
    {
        TECA_ERROR("Conversion for \"" << time_units << "\" not implemented")
        return nullptr;
    }

    // get the track ids
    const_p_teca_int_array track_ids =
        std::dynamic_pointer_cast<const teca_int_array>(
            in_table->get_column(this->track_id_column));

    if (!track_ids)
    {
        TECA_ERROR("column \"" << this->track_id_column
            << "\" is not in the table")
        return nullptr;
    }

    // the spatial coorinates
    const_p_teca_variant_array x =
        in_table->get_column(this->x_coordinate_column);

    if (!x)
    {
        TECA_ERROR("column \"" << this->x_coordinate_column
            << "\" is not in the table")
        return nullptr;
    }

    const_p_teca_variant_array y =
        in_table->get_column(this->y_coordinate_column);

    if (!y)
    {
        TECA_ERROR("column \"" << this->y_coordinate_column
            << "\" is not in the table")
        return nullptr;
    }

    // time axis
    const_p_teca_variant_array time =
        in_table->get_column(this->time_column);

    if (!time)
    {
        TECA_ERROR("column \"" << this->time_column
            << "\" is not in the table")
        return nullptr;
    }

    // get the surface wind speeds
    const_p_teca_variant_array surface_wind =
        in_table->get_column(this->surface_wind_column);

    if (!surface_wind)
    {
        TECA_ERROR("column \"" << this->surface_wind_column
            << "\" is not in the table")
        return nullptr;
    }

    // get the surface wind speeds
    const_p_teca_variant_array sea_level_pressure =
        in_table->get_column(this->sea_level_pressure_column);

    if (!sea_level_pressure)
    {
        TECA_ERROR("column \"" << this->sea_level_pressure_column
            << "\" is not in the table")
        return nullptr;
    }

    // scan the track ids and build the random access
    // data structure
    std::vector<unsigned long> track_starts(1, 0);
    size_t n_rows = track_ids->size();
    const int *pids = track_ids->get();
    for (size_t i = 1; i < n_rows; ++i)
        if (pids[i] != pids[i-1])
            track_starts.push_back(i);
    track_starts.push_back(n_rows);
    size_t n_tracks = track_starts.size() - 1;

    // record track id
    p_teca_long_array out_ids = teca_long_array::New(n_tracks);
    long *pout_ids = out_ids->get();
    for (size_t i =0; i < n_tracks; ++i)
        pout_ids[i] = pids[track_starts[i]];

    // record track start time
    p_teca_variant_array start_time = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        start_time.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();
        NT *pstart_time = static_cast<TT*>(start_time.get())->get();
        for (size_t i = 0; i < n_tracks; ++i)
            pstart_time[i] = ptime[track_starts[i]];
        )

    // record track start position
    p_teca_variant_array start_x = x->new_instance(n_tracks);
    p_teca_variant_array start_y = x->new_instance(n_tracks);

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        start_x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *pstart_x = static_cast<TT*>(start_x.get())->get();
        NT *pstart_y = static_cast<TT*>(start_y.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = track_starts[i];

            pstart_x[i] = px[q];
            pstart_y[i] = py[q];
        }
        )

    // compute the storm duration
    p_teca_variant_array duration = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        duration.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();
        NT *pduration = static_cast<TT*>(duration.get())->get();
        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long first = track_starts[i];
            unsigned long last = track_starts[i+1] - 1;
            pduration[i] = ptime[last] - ptime[first];
        }
        )

    // compute the distance traveled
    p_teca_variant_array length = x->new_instance(n_tracks);

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        length.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *plength = static_cast<TT*>(length.get())->get();

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
    p_teca_int_array category = teca_int_array::New(n_tracks);
    int *pcategory = category->get();

    p_teca_variant_array max_surface_wind
        = surface_wind->new_instance(n_tracks);

    p_teca_unsigned_long_array max_surface_wind_id
        = teca_unsigned_long_array::New(n_tracks);

    unsigned long *pmax_surface_wind_id
        = max_surface_wind_id->get();

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        max_surface_wind.get(),

        const NT *psurface_wind =
            static_cast<const TT*>(surface_wind.get())->get();

        NT *pmax_surface_wind =
            static_cast<TT*>(max_surface_wind.get())->get();

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
    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        start_x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *pmax_surface_wind_x
            = static_cast<TT*>(max_surface_wind_x.get())->get();

        NT *pmax_surface_wind_y
            = static_cast<TT*>(max_surface_wind_y.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmax_surface_wind_id[i];
            pmax_surface_wind_x[i] = px[q];
            pmax_surface_wind_y[i] = py[q];
        }
        )

    // time of max surface wind
    p_teca_variant_array max_surface_wind_t = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        max_surface_wind_t.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();

        NT *pmax_surface_wind_t
            = static_cast<TT*>(max_surface_wind_t.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmax_surface_wind_id[i];
            pmax_surface_wind_t[i] = ptime[q];
        }
        )

    // record the min sea level pressure
    p_teca_variant_array min_sea_level_pressure =
        sea_level_pressure->new_instance(n_tracks);

    p_teca_unsigned_long_array min_sea_level_pressure_id
        = teca_unsigned_long_array::New(n_tracks);

    unsigned long *pmin_sea_level_pressure_id
        = min_sea_level_pressure_id->get();

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        min_sea_level_pressure.get(),
        const NT *psea_level_pressure =
            static_cast<const TT*>(sea_level_pressure.get())->get();

        NT *pmin_sea_level_pressure =
            static_cast<TT*>(min_sea_level_pressure.get())->get();

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
    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        start_x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *pmin_sea_level_pressure_x
            = static_cast<TT*>(min_sea_level_pressure_x.get())->get();

        NT *pmin_sea_level_pressure_y
            = static_cast<TT*>(min_sea_level_pressure_y.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmin_sea_level_pressure_id[i];
            pmin_sea_level_pressure_x[i] = px[q];
            pmin_sea_level_pressure_y[i] = py[q];
        }
        )

    // time of min sea level pressure
    p_teca_variant_array min_sea_level_pressure_t = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        min_sea_level_pressure_t.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();

        NT *pmin_sea_level_pressure_t
            = static_cast<TT*>(min_sea_level_pressure_t.get())->get();

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

    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        time.get(), _T,

        const NT_T *ptime = static_cast<TT_T*>(time.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
            ACE.get(), _W,

            NT_W *pACE = static_cast<TT_W*>(ACE.get())->get();

            const NT_W *psurface_wind =
                static_cast<const TT_W*>(surface_wind.get())->get();

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

    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        time.get(), _T,

        const NT_T *ptime = static_cast<TT_T*>(time.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
            PDI.get(), _W,

            NT_W *pPDI = static_cast<TT_W*>(PDI.get())->get();

            const NT_W *psurface_wind =
                static_cast<const TT_W*>(surface_wind.get())->get();

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

    p_teca_int_array region_id = teca_int_array::New(n_tracks, -1);
    int *pregion_id = region_id->get();

    p_teca_string_array region_name = teca_string_array::New(n_tracks);
    std::string *pregion_name = region_name->get();

    p_teca_string_array region_long_name = teca_string_array::New(n_tracks);
    std::string *pregion_long_name = region_long_name->get();

    TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

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
