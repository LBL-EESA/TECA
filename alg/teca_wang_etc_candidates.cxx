#include "teca_wang_etc_candidates.h"

#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_table.h"
#include "teca_database.h"
#include "teca_calendar.h"
#include "teca_coordinate_util.h"

#include <iostream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <chrono>
#include <utility>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

//#define TECA_DEBUG 2

using std::cerr;
using std::endl;
using seconds_t = std::chrono::duration<double, std::chrono::seconds::period>;

namespace wang_etc_internal
{

template<typename num_t>
void mem_copy(num_t *dest, const num_t *src, unsigned long di0,
    unsigned long di1, unsigned long dj0, unsigned long dj1,
    unsigned long ndi, unsigned long si0, unsigned long sj0,
    unsigned long nsi)
{
    for (unsigned long j = dj0, jj = sj0; j < dj1; ++j,++jj)
    {
        unsigned long d = j*ndi;
        unsigned long s = jj*nsi;
        for (unsigned long i = di0, ii = si0; i < di1; ++i,++ii)
        {
            dest[d+i] = src[s+ii];
        }
    }
}

template<typename num_t>
num_t *malloc_copy(const num_t *src, unsigned long n_lon,
    unsigned long n_lat, unsigned long n_ghosts,
    bool left_bc, bool right_bc)
{
    unsigned long n_lon_dest = n_lon +
        (left_bc ? n_ghosts : 0) + (right_bc ? n_ghosts : 0);

    num_t *dest = static_cast<num_t*>(malloc(n_lat*n_lon_dest*sizeof(num_t)));

    // main patch
    mem_copy(dest, src, (left_bc ? n_ghosts : 0),
        n_lon + (left_bc ? n_ghosts : 0), 0, n_lat, n_lon_dest,
        0, 0, n_lon);

    // left side
    if (left_bc)
        mem_copy(dest, src, 0, n_ghosts, 0, n_lat,
            n_lon_dest, n_lon-n_ghosts-1, 0, n_lon);

    // right side
    if (right_bc)
        mem_copy(dest, src, n_lon + (left_bc ? n_ghosts : 0),
            n_lon_dest, 0, n_lat, n_lon_dest, 0, 0, n_lon);

    return dest;
}

// return the index space coordinates at the minimum value of f
// on the patch i0, i1, j0, j1
template<typename num_t>
void minloc(const num_t *f, unsigned long nx_f, unsigned long i0, unsigned long i1,
    unsigned long j0, unsigned long j1, unsigned long &ii, unsigned long &jj)
{
    num_t val = f[j0*nx_f+i0];
    ii = i0;
    jj = j0;

    for (unsigned long j = j0; j <= j1; ++j)
    {
        const num_t *ff = f + j*nx_f;
        for (unsigned long i = i0; i <= i1; ++i)
        {
            if (val > ff[i])
            {
                ii = i;
                jj = j;
                val = ff[i];
            }
        }
    }
}

using ij_coord_t = std::pair<unsigned long, unsigned long>;

// computes a connected component of the field (f) on a patch (i0, i1, j0, j1).
// the center of the patch is used as the seed. values less than the threshold
// (f_crit) are in the component. The i,j pairs of points on the component boundary
// are returned.
template<typename num_t>
void component_boundary(num_t f_crit, num_t *f, unsigned long nx_f,
    unsigned long i0, unsigned long i1, unsigned long j0, unsigned long j1,
    std::vector<ij_coord_t> &border)
{
    unsigned long ni = i1 - i0 + 1;
    unsigned long nj = j1 - j0 + 1;

    unsigned long n_bytes = ni*nj;
    char *mask = static_cast<char*>(malloc(n_bytes));
    memset(mask, 0, n_bytes);

    std::vector<ij_coord_t> work;

    // add seed point at the center of the patch
    unsigned long i = (i0 + i1)/2ul;
    unsigned long j = (j0 + j1)/2ul;

    work.push_back(ij_coord_t(i, j));

    while (!work.empty())
    {
        ij_coord_t q = work.back();
        work.pop_back();

        i = q.first;
        j = q.second;
        unsigned long ii = j*nx_f + i;

        unsigned long r = i - i0;
        unsigned long s = j - j0;
        unsigned long rr = s*ni + r;

        if (f[ii] < f_crit)
        {
            // this value is below the threshold. add neighbors inside
            // the patch to the work queue.
            if ((j < j1) && !mask[rr+ni])
                work.push_back(ij_coord_t(i, j+1));
            if ((j > j0) && !mask[rr-ni])
                work.push_back(ij_coord_t(i, j-1));
            if ((i < i1) && !mask[rr+1])
                work.push_back(ij_coord_t(i+1, j));
            if ((i > i0) && !mask[rr-1])
                work.push_back(ij_coord_t(i-1, j));

            // mark as visited
            mask[rr] = 1;
        }
        else
        {
            // this value is on the border.
            border.push_back(ij_coord_t(i, j));

            // mark as visited
            mask[rr] = -1;
        }
    }

#if defined(TECA_DEBUG)
    // print the mask
    for (unsigned long j = 0; j < nj; ++j)
    {
        for (unsigned long i = 0; i < ni; ++ i)
        {
            unsigned long ii = j*ni + i;
            cerr << mask[ii] << " ";
        }
        cerr << endl;
    }
    free(mask);
#endif
}

// returns true if all the index space points (ij_coords) are circumscribed
// by radius of r_crit centered on the patch i0, i1, j0, j1
template<typename coord_t>
bool circumscribed(coord_t r_crit, const coord_t *lon, const coord_t *lat,
    unsigned long i_0, unsigned long i_1, unsigned long j_0, unsigned long j_1,
    std::vector<ij_coord_t> &coords)
{
    if (coords.empty())
        return false;

    unsigned long i = (i_0 + i_1)/2ul;
    unsigned long j = (j_0 + j_1)/2ul;

    coord_t x_0 = lon[i];
    coord_t y_0 = lat[j];

    coord_t r_crit_sq = r_crit*r_crit;

    while (!coords.empty())
    {
        ij_coord_t ij = coords.back();
        coords.pop_back();

        coord_t x = x_0 - lon[ij.first];
        coord_t y = y_0 - lat[ij.second];

        coord_t r_sq = x*x + y*y;

        if (r_sq > r_crit_sq)
            return false;
    }

    return true;
}

template<typename coord_t, typename num_t_1, typename num_t_2>
int find_candidates(unsigned long time_step, const coord_t *lon,
    const coord_t *lat, const num_t_1 *pressure, const num_t_1 *vorticity,
    const num_t_2 *elevation, unsigned long n_lon, unsigned long n_lat,
    coord_t win_size, num_t_1 min_pressure_delta, coord_t max_pressure_radius,
    num_t_1 min_vorticity, num_t_2 max_elevation, p_teca_table candidates)

{
    // find the number of grid points in the requested search window size
    coord_t delta_lon = lon[1] - lon[0];
    coord_t delta_lat = lat[1] - lat[0];

    bool left_bc = lon[0] - delta_lon <= 0.0;
    bool right_bc = lon[n_lon-1] + delta_lon >= 360.0;

    unsigned long n_win_half =
        static_cast<long>(((win_size/delta_lat) - 1.0)/2.0);

    if (n_win_half < 1)
    {
        TECA_ERROR("Invalid search window requested " << win_size)
        return -1;
    }

    // unique id for each candidate.
    unsigned long candidate_id = time_step*10000;

    // add ghost zones for peridic longitudinal boundary
    num_t_1 *pressure_gh = malloc_copy(pressure, n_lon, n_lat,
        n_win_half, left_bc, right_bc);

    num_t_1 *vorticity_gh = malloc_copy(vorticity, n_lon, n_lat,
        n_win_half, left_bc, right_bc);

    num_t_2 *elevation_gh = malloc_copy(elevation, n_lon, n_lat,
        n_win_half, left_bc, right_bc);

    coord_t *lon_gh = malloc_copy(lon, n_lon, 1, n_win_half,
        left_bc, right_bc);

    // fix sign of vorticity in southern hemisphere
    unsigned long eq = 0;
    while ((eq < n_lat) && (lat[eq] < coord_t()))
        eq += 1;

    for (unsigned long j = 0; j < eq; ++j)
        vorticity_gh[j] *= -1;

    // loop over input, sliding the search window
    unsigned long hi_j = n_lat - n_win_half;
    unsigned long hi_i = n_lon + (left_bc ? n_win_half : 0);

    unsigned long n_lon_gh = hi_i + (right_bc ? n_win_half : 0);

    for (unsigned long j = n_win_half; j < hi_j; ++j)
    {
        unsigned long j_0 = j - n_win_half;
        unsigned long j_1 = j + n_win_half;

        unsigned long jj = j*n_lon_gh;

        for (unsigned long i = n_win_half; i < hi_i; ++i)
        {
            unsigned long i_0 = i - n_win_half;
            unsigned long i_1 = i + n_win_half;

            unsigned long ii = jj + i;

            // step 1: exclude high elevation(>= 1500m)
            if (elevation_gh[ii] >= max_elevation)
                continue;

            // step 2: a local pressure minima must be at the
            // center of the search window
            unsigned long p,q;
            minloc(pressure_gh, n_lon_gh, i_0, i_1, j_0, j_1, p, q);
            if ((p != i) || (q != j))
                continue;

            // step 3: check for identical values on 3x3 grid, and
            // use laplacian as a tie breaker. TODO: not implementing this until
            // we define identical and if 3x3 grid is relevant to modern datasets

            // step 4: check that pressure decereases enough over the neighborhood
            std::vector<ij_coord_t> boundary;

            component_boundary(min_pressure_delta, pressure_gh,
                n_lon_gh, i_0, i_1, j_0, j_1, boundary);

            if (!circumscribed(max_pressure_radius,
                lon_gh, lat, i_0, i_1, j_0, j_1, boundary))
                continue;

            // step 5: check vorticity exceeds the threshold
            if (vorticity_gh[ii] < min_vorticity)
                continue;

            // we have an ETC track candidate, recored values
            ++candidate_id;
            candidates << candidate_id << lon_gh[i] << lat[j]
                << pressure_gh[ii] << vorticity_gh[ii];
        }
    }
    return 0;
}

}



// --------------------------------------------------------------------------
teca_wang_etc_candidates::teca_wang_etc_candidates() :
    min_vorticity(0.0),
    min_pressure_delta(10.0),
    max_pressure_radius(2.0),
    max_elevation(1500.0),
    search_window(7.75),
    search_lat_low(1.0),
    search_lat_high(0.0),
    search_lon_low(1.0),
    search_lon_high(0.0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_wang_etc_candidates::~teca_wang_etc_candidates()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_wang_etc_candidates::get_properties_description(
    const std::string &prefix, options_description &opts)
{
    options_description ard_opts("Options for "
        + (prefix.empty()?"teca_wang_etc_candidates":prefix));

    ard_opts.add_options()
        TECA_POPTS_GET(std::string, prefix, vorticity_variable,
            "name of 850 mb vorticity variable")
        TECA_POPTS_GET(std::string, prefix, pressure_variable,
            "name of sea level pressure variable")
        TECA_POPTS_GET(std::string, prefix, elevation_variable,
            "name of elevation variable")
        TECA_POPTS_GET(double, prefix, min_vorticity,
            "minimum vorticty to be considered a candidate (1.6e-4)")
        TECA_POPTS_GET(double, prefix, search_window,
            "size of the search window in degrees. storms core must have a "
            "local pressure minimum centered on this window (7.74446)")
        TECA_POPTS_GET(double, prefix, min_pressure_delta,
            "minimum drop in pressure specifdied in Pa over max_pressure_radius (10.0)")
        TECA_POPTS_GET(double, prefix, max_pressure_radius,
            "radius in degrees lat over which pressure must drop by min_pressure_delta (5.0)")
        TECA_POPTS_GET(double, prefix, search_lat_low,
            "lowest latitude in degrees to search for storms (1)")
        TECA_POPTS_GET(double, prefix, search_lat_high,
            "highest latitude in degrees to search for storms (0)")
        TECA_POPTS_GET(double, prefix, search_lon_low,
            "lowest longitude in degrees to search for stroms (1)")
        TECA_POPTS_GET(double, prefix, search_lon_high,
            "highest longitude in degrees to search for storms (0)")
        ;

    opts.add(ard_opts);
}

// --------------------------------------------------------------------------
void teca_wang_etc_candidates::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, vorticity_variable)
    TECA_POPTS_SET(opts, std::string, prefix, pressure_variable)
    TECA_POPTS_SET(opts, std::string, prefix, elevation_variable)
    TECA_POPTS_SET(opts, double, prefix, min_pressure_delta)
    TECA_POPTS_SET(opts, double, prefix, max_pressure_radius)
    TECA_POPTS_SET(opts, double, prefix, min_vorticity)
    TECA_POPTS_SET(opts, double, prefix, search_window)
    TECA_POPTS_SET(opts, double, prefix, search_lat_high)
    TECA_POPTS_SET(opts, double, prefix, search_lat_low)
    TECA_POPTS_SET(opts, double, prefix, search_lon_high)
    TECA_POPTS_SET(opts, double, prefix, search_lon_low)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_wang_etc_candidates::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id()
        << "teca_wang_etc_candidates::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata output_md(input_md[0]);
    return output_md;
}

// --------------------------------------------------------------------------
int teca_wang_etc_candidates::get_active_extent(p_teca_variant_array lat,
    p_teca_variant_array lon, std::vector<unsigned long> &extent) const
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

            if (teca_coordinate_util::index_of(p_lon, 0, high_i,
                static_cast<NT>(this->search_lon_low), false, extent[0]) ||
                teca_coordinate_util::index_of(p_lon, 0, high_i,
                static_cast<NT>(this->search_lon_high), true, extent[1]))
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

            if (teca_coordinate_util::index_of(p_lat, 0, high_j,
                static_cast<NT>(this->search_lat_low), false, extent[2]) ||
                teca_coordinate_util::index_of(p_lat, 0, high_j,
                static_cast<NT>(this->search_lat_high), true, extent[3]))
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
std::vector<teca_metadata> teca_wang_etc_candidates::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id()
        << "teca_wang_etc_candidates::get_upstream_request" << endl;
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
    arrays.insert(this->vorticity_variable);
    arrays.insert(this->pressure_variable);
    arrays.insert(this->elevation_variable);

    teca_metadata up_req(request);
    up_req.set("arrays", arrays);
    up_req.set("extent", extent);

    up_reqs.push_back(up_req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_wang_etc_candidates::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id() << "teca_wang_etc_candidates::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    std::chrono::high_resolution_clock::time_point t0, t1;
    t0 = std::chrono::high_resolution_clock::now();

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

    // get calendar
    std::string calendar;
    mesh->get_calendar(calendar);

    // get extent of data passed in
    std::vector<unsigned long> extent;
    mesh->get_extent(extent);

    unsigned long n_lat = extent[3] - extent[2] + 1;
    unsigned long n_lon = extent[1] - extent[0] + 1;

    // get vorticity array
    const_p_teca_variant_array vorticity
        = mesh->get_point_arrays()->get(this->vorticity_variable);

    if (!vorticity)
    {
        TECA_ERROR("Dataset missing vorticity variable \""
            << this->vorticity_variable << "\"")
        return nullptr;
    }

    // get pressure array
    const_p_teca_variant_array pressure
        = mesh->get_point_arrays()->get(this->pressure_variable);

    if (!pressure)
    {
        TECA_ERROR("Dataset missing pressure variable \""
            << this->pressure_variable << "\"")
        return nullptr;
    }

    // get elevation array
    const_p_teca_variant_array elevation
        = mesh->get_point_arrays()->get(this->elevation_variable);

    if (!elevation)
    {
        TECA_ERROR("Dataset missing elevation variable \""
            << this->elevation_variable << "\"")
        return nullptr;
    }

    // identify candidates
    p_teca_table candidates = teca_table::New();

    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        x.get(), _COORD,

        const NT_COORD *lon = static_cast<const TT_COORD*>(x.get())->get();
        const NT_COORD *lat = static_cast<const TT_COORD*>(y.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            elevation.get(), _VAR_1,

            const NT_VAR_1 *z = dynamic_cast<const TT_VAR_1*>(elevation.get())->get();

            NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
                pressure.get(), _VAR_2,

                // configure the candidate table
                candidates->declare_columns("storm_id", int(),
                    "lon", NT_COORD(), "lat", NT_COORD(),
                    "pressure", NT_VAR_2(), "vorticity", NT_VAR_2());

                const NT_VAR_2 *w = dynamic_cast<const TT_VAR_2*>(vorticity.get())->get();
                const NT_VAR_2 *P = dynamic_cast<const TT_VAR_2*>(pressure.get())->get();

                // invoke the detector
                if (wang_etc_internal::find_candidates<NT_COORD, NT_VAR_2, NT_VAR_1>(
                    time_step, lon, lat, P, w, z, n_lon, n_lat,
                    this->search_window, this->min_pressure_delta,
                    this->max_pressure_radius, this->min_vorticity,
                    this->max_elevation, candidates))
                {
                    TECA_ERROR("The Wang ETC detector encountered an error")
                    return nullptr;
                }
                )
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

    t1 = std::chrono::high_resolution_clock::now();
    seconds_t dt(t1 - t0);
    TECA_STATUS("teca_wang_etc_candidates step=" << time_step
        << " t=" << time_offset << ", dt=" << dt.count() << " sec")

    return out_table;
}
