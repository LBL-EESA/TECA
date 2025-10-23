#include "teca_latitude_range_filter.h"

#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_string_util.h"
#include "teca_mpi.h"

#include <iostream>
#include <set>
#include <chrono>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;
using seconds_t = std::chrono::duration<double, std::chrono::seconds::period>;

#if defined(TECA_HAS_CUDA)
namespace cuda_impl
{
template <typename coord_t, typename data_t>
__global__
void filter_by_lat_range(
    data_t * __restrict__ output,
    const data_t* __restrict__ input,
    const coord_t* __restrict__ lat,
    unsigned int nlon, unsigned int nlat,
    coord_t min_lat, coord_t max_lat)
{
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= nlat) return;

    coord_t lat_j = abs(lat[j]);

    size_t row = j * nlon;
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x;
         i < nlon; i += blockDim.x * gridDim.x)
    {
        output[row + i] = (lat_j >= min_lat && lat_j <= max_lat) ? input[row + i] : data_t(0);
    }
}
}
#endif
// --------------------------------------------------------------------------
//  Host helpers
namespace host_impl
{
template <typename num_t, typename coord_t>
void filter_by_lat_range(num_t *output, const num_t *input, const coord_t *lat,
    size_t n_lat_vals, size_t n_lon_vals, coord_t min_lat, coord_t max_lat)
{
    for (size_t j = 0; j < n_lat_vals; ++j)
    {
        size_t jj = j * n_lon_vals;
        for (size_t i = 0; i < n_lon_vals; ++i)
            output[jj + i] = (abs(lat[j]) >= min_lat && abs(lat[j]) <= max_lat) ?
                             input[jj + i] : num_t(0);
    }
}
}

// --------------------------------------------------------------------------
teca_latitude_range_filter::teca_latitude_range_filter()
    : min_lat(std::numeric_limits<double>::quiet_NaN())
    , max_lat(std::numeric_limits<double>::quiet_NaN())
    , variable_postfix("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_latitude_range_filter::~teca_latitude_range_filter()
{}

// --------------------------------------------------------------------------
#if defined(TECA_HAS_BOOST)
void teca_latitude_range_filter::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for " +
        (prefix.empty() ? "teca_latitude_range_filter" : prefix));

    opts.add_options()
        TECA_POPTS_GET(double, prefix, min_lat,
            "minimum latitude_range (inclusive) for the range filter")
        TECA_POPTS_GET(double, prefix, max_lat,
            "maximum latitude_range (inclusive) for the range filter")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix,
            filtered_variables,
            "variables to which the range filter will be applied")
        TECA_POPTS_GET(std::string, prefix, variable_postfix,
            "post‑fix for filtered variable names (if empty, replace input)")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);
    global_opts.add(opts);
}
// --------------------------------------------------------------------------
void teca_latitude_range_filter::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, double, prefix, min_lat)
    TECA_POPTS_SET(opts, double, prefix, max_lat)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, filtered_variables)
    TECA_POPTS_SET(opts, std::string, prefix, variable_postfix)
}
#endif
// --------------------------------------------------------------------------
int teca_latitude_range_filter::get_min_lat(const teca_metadata &request,
    double &min_lat)
{
    min_lat = 0;
    if (std::isnan(this->min_lat))
    {
        if (request.has("min_lat"))
            request.get("min_lat", min_lat);
        else
            return -1;
    }
    else
        min_lat = this->min_lat;
    return 0;
}
// --------------------------------------------------------------------------
int teca_latitude_range_filter::get_max_lat(const teca_metadata &request,
    double &max_lat)
{
    if (std::isnan(this->max_lat))
    {
        if (request.has("max_lat"))
            request.get("max_lat", max_lat);
        else
            return -1;
    }
    else
        max_lat = this->max_lat;
    return 0;
}
// --------------------------------------------------------------------------
int teca_latitude_range_filter::get_filtered_variables(
    std::vector<std::string> &vars)
{
    if (this->filtered_variables.empty())
        return -1;
    vars = this->filtered_variables;
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_latitude_range_filter::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    (void)port;
    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    const std::string &var_postfix = this->variable_postfix;
    if (!var_postfix.empty())
    {
        std::vector<std::string> &filtered_vars = this->filtered_variables;
        for (const auto &var : filtered_vars)
            out_md.append("variables", var + var_postfix);
    }
    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_latitude_range_filter::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;
    teca_metadata req(request);

    // get the name of the array to request
    std::vector<std::string> filtered_vars;
    if (this->get_filtered_variables(filtered_vars))
    {
        TECA_FATAL_ERROR("No variables to filter specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(filtered_vars.begin(), filtered_vars.end());

    // Cleaning off the postfix for arrays passed in the pipeline.
    // For ex a down stream could request "foo_damped" then we'd
    // need to request "foo". also remove "foo_damped" from the
    // request.
    const std::string &var_postfix = this->variable_postfix;
    if (!var_postfix.empty())
    {
        teca_string_util::remove_postfix(arrays, var_postfix);
    }

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_latitude_range_filter::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;
    std::chrono::high_resolution_clock::time_point t0, t1;
    t0 = std::chrono::high_resolution_clock::now();

    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->get_communicator(), &rank);
    } 
#endif

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("empty input, or not a mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh =
        std::dynamic_pointer_cast<teca_cartesian_mesh>(in_mesh->new_instance());

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array names
    std::vector<std::string> filtered_vars;
    if (this->get_filtered_variables(filtered_vars))
    {
        TECA_FATAL_ERROR("No variable specified to filter")
        return nullptr;
    }

// get the filter limits, if none were provided, these are the defaults
// that will be used.
    double min_lat = 0.0;
    double max_lat = 90.0;

    this->get_min_lat(request, min_lat);
    this->get_max_lat(request, max_lat);

    // get the coordinate axes
    const_p_teca_variant_array lat = in_mesh->get_y_coordinates();
    const_p_teca_variant_array lon = in_mesh->get_x_coordinates();

    size_t n_lat = lat->size();
    size_t n_lon = lon->size();

    int device_id = -1;
#if defined(TECA_HAS_CUDA)
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (teca_cuda_util::set_device(device_id))
            return nullptr;

        cudaError_t ierr = cudaSuccess;

        // domain decomp for the gpu
        int ntx = 32, nty = 4;
        dim3 thrs(ntx, nty);
        dim3 blks(n_lon / ntx + (n_lon % ntx ? 1 : 0),
                  n_lat / nty + (n_lat % nty ? 1 : 0));

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            lat.get(), _COORD,

            // get the lat array coordinates
            auto [sp_lat, p_lat] = get_cuda_accessible<TT_COORD>(lat);

            size_t n_arrays = filtered_vars.size();
            for (size_t i = 0; i < n_arrays; ++i)
            {
                // get the input array
                const_p_teca_variant_array input_array =
                    out_mesh->get_point_arrays()->get(filtered_vars[i]);

                if (!input_array)
                {
                    TECA_FATAL_ERROR("filtered variable \"" << filtered_vars[i] <<
                        "\" not present in the input")
                    return nullptr;
                }

                NESTED_VARIANT_ARRAY_DISPATCH(
                    input_array.get(), _DATA,

                    // read only access to the input
                    auto [sp_in, p_in] = get_cuda_accessible<CTT_DATA>(input_array);
                    // allocate the output
                    size_t n_elem = input_array->size();
                    auto [sp_out, p_out] = teca_variant_array_util::New<TT_DATA>(n_elem, teca_variant_array::allocator::cuda_async);

                    // launch range filter kernel
                    cuda_impl::filter_by_lat_range<<<blks,thrs>>>(p_out, p_in, p_lat,
                        int(n_lon), int(n_lat),
                        NT_COORD(min_lat), NT_COORD(max_lat));
                    ierr = cudaGetLastError();
                    if (ierr != cudaSuccess)
                    {
                        TECA_FATAL_ERROR("Failed to launch the range filter kernel")
                        return nullptr;
                    }

                    // set the damped array in the output
                    std::string out_var_name = filtered_vars[i] + this->variable_postfix;
                    out_mesh->get_point_arrays()->set(out_var_name, sp_out);
                )
            }
        )
    }
    else
    {
#endif
        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            lat.get(), _COORD,

            // construct the gaussian filter
            auto [sp_lat, p_lat] = get_host_accessible<TT_COORD>(lat);

            sync_host_access_any(lat);

            size_t n_arrays = filtered_vars.size();
            for (size_t i = 0; i < n_arrays; ++i)
            {
                // get the input array
                const_p_teca_variant_array input_array =
                    out_mesh->get_point_arrays()->get(filtered_vars[i]);

                if (!input_array)
                {
                    TECA_FATAL_ERROR("filter variable \"" << filtered_vars[i] <<
                        "\" not present in the input")
                    return nullptr;
                }

                // allocate the output
                size_t n_elem = input_array->size();
                p_teca_variant_array filtered_array = input_array->new_instance(n_elem);

                NESTED_VARIANT_ARRAY_DISPATCH(
                    input_array.get(), _DATA,

                    auto [sp_in, p_in] = get_host_accessible<CTT_DATA>(input_array);
                    auto [p_filtered_array] = data<TT_DATA>(filtered_array);
                    sync_host_access_any(input_array);

                    // apply the filter
                    host_impl::filter_by_lat_range(p_filtered_array, p_in, p_lat,
                        n_lat, n_lon, NT_COORD(min_lat), NT_COORD(max_lat));
                )

                // set the damped array in the output
                std::string out_var_name = filtered_vars[i] + this->variable_postfix;
                out_mesh->get_point_arrays()->set(out_var_name, filtered_array);
            }
        )
#if defined(TECA_HAS_CUDA)
    }
#endif

    teca_metadata &omd = out_mesh->get_metadata();
    omd.set("latitude_range_filter_min", min_lat);
    omd.set("latitude_range_filter_max", max_lat);

    t1 = std::chrono::high_resolution_clock::now();
    seconds_t dt(t1 - t0);

    if (this->get_verbose() && (rank == 0))
    {
        TECA_STATUS("latitude range filter computed on "
            << (device_id >= 0 ? "CUDA device" : "the host")
            << " (" << device_id << ") in " << dt.count() << " seconds");
    }

    return out_mesh;
}