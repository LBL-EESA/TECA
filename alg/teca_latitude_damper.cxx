#include "teca_latitude_damper.h"

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

//#define TECA_DEBUG
#define LDAMP_SINGLE_KERNEL

#if defined(TECA_HAS_CUDA)
namespace cuda_impl
{
#if defined(LDAMP_SINGLE_KERNEL)
template <typename coord_t, typename data_t>
__global__
void filter_by_lat(
    data_t * __restrict__ output,
    const data_t* __restrict__ input,
    const coord_t* __restrict__ lat,
    unsigned int nlon, unsigned int nlat, coord_t mu, coord_t sigma)
{
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (j >= nlat) return;

    coord_t two_sigma_sqr = coord_t(2)*sigma*sigma;
    coord_t x_min_mu = lat[j] - mu;
    coord_t neg_x_min_mu_sqr = -x_min_mu*x_min_mu;
    coord_t filter_j = coord_t(1) - exp(neg_x_min_mu_sqr/two_sigma_sqr);

    size_t jj = j*nlon;

    for (size_t i = threadIdx.x + blockIdx.x*blockDim.x; i < nlon;
         i += blockDim.x*gridDim.x)
    {
        output[jj + i] = filter_j * input[jj + i];
    }
}
#else
// get the filter ready to be applied in the next steps
template <typename coord_t>
__global__
void get_lat_filter(
    coord_t *filter, const coord_t *lat, size_t nlat,
    coord_t mu, coord_t sigma)
{
    coord_t two_sigma_sqr = coord_t(2)*sigma*sigma;

    for (size_t i = threadIdx.x + blockIdx.x*blockDim.x; i < nlat;
         i += blockDim.x*gridDim.x)
    {
        coord_t x_min_mu = lat[i] - mu;
        coord_t neg_x_min_mu_sqr = -x_min_mu*x_min_mu;
        filter[i] = coord_t(1) - exp(neg_x_min_mu_sqr/two_sigma_sqr);
    }
}

template <typename num_t, typename coord_t>
__global__
void apply_lat_filter(
    num_t * __restrict__ output,
    const num_t * __restrict__ input,
    const coord_t * __restrict__ filter,
    size_t nlat, size_t nlon)
{
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (j >= nlat) return;

    size_t jj = j*nlon;

    for (size_t i = threadIdx.x + blockIdx.x*blockDim.x; i < nlon;
         i += blockDim.x*gridDim.x)
    {
        output[jj + i] = filter[j] * input[jj + i];
    }
}
#endif
}
#endif

namespace host_impl
{

// get the filter ready to be applied in the next steps
template <typename coord_t>
void get_lat_filter(
    coord_t *filter, const coord_t *lat, size_t n_lat_vals,
    coord_t mu, coord_t sigma)
{
    coord_t two_sigma_sqr = 2.0*sigma*sigma;
    for (size_t i = 0; i < n_lat_vals; ++i)
    {
        coord_t x_min_mu = lat[i] - mu;
        coord_t neg_x_min_mu_sqr = -x_min_mu*x_min_mu;
        filter[i] = coord_t(1) - exp(neg_x_min_mu_sqr/two_sigma_sqr);
    }
}

// damp the input array using inverted gaussian
template <typename num_t, typename coord_t>
void apply_lat_filter(
    num_t *output, const num_t *input, const coord_t *filter,
    size_t n_lat_vals, size_t n_lon_vals)
{
    for (size_t j = 0; j < n_lat_vals; ++j)
    {
        size_t jj = j * n_lon_vals;
        for (size_t i = 0; i < n_lon_vals; ++i)
        {
            output[jj + i] = filter[j] * input[jj + i];
        }
    }
}
}

// --------------------------------------------------------------------------
teca_latitude_damper::teca_latitude_damper() :
    center(std::numeric_limits<double>::quiet_NaN()),
    half_width_at_half_max(std::numeric_limits<double>::quiet_NaN()),
    variable_postfix("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_latitude_damper::~teca_latitude_damper()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_latitude_damper::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_latitude_damper":prefix));

    opts.add_options()
        TECA_POPTS_GET(double, prefix, center,
            "set the center (mu) for the gaussian filter")
        TECA_POPTS_GET(double, prefix, half_width_at_half_max,
            "set the value of the half width at half maximum (HWHM) "
            "to calculate sigma from: sigma = HWHM/std::sqrt(2.0*std::log(2.0))")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, damped_variables,
            "set the variables that will be damped by the inverted "
            "gaussian filter")
        TECA_POPTS_GET(std::string, prefix, variable_postfix,
            "set the post-fix that will be attached to the variables "
            "that will be saved in the output")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}
// --------------------------------------------------------------------------
void teca_latitude_damper::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, double, prefix, center)
    TECA_POPTS_SET(opts, double, prefix, half_width_at_half_max)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, damped_variables)
    TECA_POPTS_SET(opts, std::string, prefix, variable_postfix)
}
#endif

// --------------------------------------------------------------------------
int teca_latitude_damper::get_sigma(const teca_metadata &request, double &sigma)
{
    double hwhm = 0.0;
    if (std::isnan(this->half_width_at_half_max))
    {
        if (request.has("half_width_at_half_max"))
            request.get("half_width_at_half_max", hwhm);
        else
            return -1;
    }
    else
    {
        hwhm = this->half_width_at_half_max;
    }

    sigma = hwhm/std::sqrt(2.0*std::log(2.0));

    return 0;
}

// --------------------------------------------------------------------------
int teca_latitude_damper::get_mu(const teca_metadata &request, double &mu)
{
    if (std::isnan(this->center))
    {
        if (request.has("center"))
            request.get("center", mu);
        else
            return -1;
    }
    else
    {
        mu = this->center;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_latitude_damper::get_damped_variables(std::vector<std::string> &vars)
{
    if (this->damped_variables.empty())
        return -1;
    else
        vars = this->damped_variables;

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_latitude_damper::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_latitude_damper::get_output_metadata" << std::endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    const std::string &var_postfix = this->variable_postfix;
    if (!var_postfix.empty())
    {
        std::vector<std::string> &damped_vars = this->damped_variables;

        size_t n_arrays = damped_vars.size();
        for (size_t i = 0; i < n_arrays; ++i)
        {
            out_md.append("variables", damped_vars[i] + var_postfix);
        }
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_latitude_damper::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_latitude_damper::get_upstream_request" << std::endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;
    teca_metadata req(request);

    // get the name of the array to request
    std::vector<std::string> damped_vars;
    if (this->get_damped_variables(damped_vars))
    {
        TECA_FATAL_ERROR("No variables to damp specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(damped_vars.begin(), damped_vars.end());

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
const_p_teca_dataset teca_latitude_damper::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_latitude_damper::execute" << std::endl;
#endif
    (void)port;

    std::chrono::high_resolution_clock::time_point t0, t1;
    t0 = std::chrono::high_resolution_clock::now();

    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
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
    std::vector<std::string> damped_vars;
    if (this->get_damped_variables(damped_vars))
    {
        TECA_FATAL_ERROR("No variable specified to damp")
        return nullptr;
    }

    // get Gaussian paramters. if none were provided, these are the defaults
    // that will be used.
    double mu = 0.0;
    double sigma = 45.0;

    this->get_mu(request, mu);
    this->get_sigma(request, sigma);

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

            // get the lat coordinates
            auto [sp_lat, p_lat] = get_cuda_accessible<TT_COORD>(lat);

#if !defined(LDAMP_SINGLE_KERNEL)
            // generate the filter
            hamr::buffer<NT_COORD> filter(allocator::cuda_async, n_lat);

            int nt1 = 32;
            int blk1 = ( n_lat / nt1 + ( n_lat % nt1 ? 1 : 0 ) );

            cuda_impl::get_lat_filter<<<blk1, nt1>>>(filter.data(), p_lat, n_lat,
                                                     NT_COORD(mu), NT_COORD(sigma));
            ierr = cudaGetLastError();
            if (ierr != cudaSuccess)
            {
                TECA_FATAL_ERROR("Failed to launch the filter kernel")
                return nullptr;
            }
#endif
            size_t n_arrays = damped_vars.size();
            for (size_t i = 0; i < n_arrays; ++i)
            {
                // get the input array
                const_p_teca_variant_array input_array
                    = out_mesh->get_point_arrays()->get(damped_vars[i]);

                if (!input_array)
                {
                    TECA_FATAL_ERROR("damper variable \"" << damped_vars[i]
                        << "\" is not in the input")
                    return nullptr;
                }

                NESTED_VARIANT_ARRAY_DISPATCH(
                    input_array.get(), _DATA,

                    // read only access to the input
                    auto [sp_in, p_in] = get_cuda_accessible<CTT_DATA>(input_array);

                    // allocate the output
                    size_t n_elem = input_array->size();
                    auto [sp_out, p_out] = teca_variant_array_util::New<TT_DATA>(n_elem, teca_variant_array::allocator::cuda_async);

                    // apply the filter
#if defined(LDAMP_SINGLE_KERNEL)
                    cuda_impl::filter_by_lat<<<blks,thrs>>>(p_out, p_in, p_lat,
                                                            int(n_lon), int(n_lat),
                                                            NT_COORD(mu), NT_COORD(sigma));
#else
                    cuda_impl::apply_lat_filter<<<blks,thrs>>>(p_out, p_in, filter.data(),
                                                               n_lat, n_lon);
#endif
                    ierr = cudaGetLastError();
                    if (ierr != cudaSuccess)
                    {
                        TECA_FATAL_ERROR("Failed to launch the filter kernel")
                        return nullptr;
                    }

                    // set the damped array in the output
                    std::string out_var_name = damped_vars[i] + this->variable_postfix;
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

            NT_COORD *filter = (NT_COORD*)malloc(n_lat*sizeof(NT_COORD));
            host_impl::get_lat_filter<NT_COORD>(filter, p_lat, n_lat, mu, sigma);

            size_t n_arrays = damped_vars.size();
            for (size_t i = 0; i < n_arrays; ++i)
            {
                // get the input array
                const_p_teca_variant_array input_array
                    = out_mesh->get_point_arrays()->get(damped_vars[i]);

                if (!input_array)
                {
                    TECA_FATAL_ERROR("damper variable \"" << damped_vars[i]
                        << "\" is not in the input")
                    return nullptr;
                }

                // allocate the output
                size_t n_elem = input_array->size();
                p_teca_variant_array damped_array = input_array->new_instance(n_elem);

                NESTED_VARIANT_ARRAY_DISPATCH(
                    input_array.get(), _DATA,

                    auto [sp_in, p_in] = get_host_accessible<CTT_DATA>(input_array);
                    auto [p_damped_array] = data<TT_DATA>(damped_array);

                    sync_host_access_any(input_array);

                    // apply the filter
                    host_impl::apply_lat_filter(p_damped_array, p_in, filter, n_lat, n_lon);
                )

                // set the damped array in the output
                std::string out_var_name = damped_vars[i] + this->variable_postfix;
                out_mesh->get_point_arrays()->set(out_var_name, damped_array);
            }

            free(filter);
        )
#if defined(TECA_HAS_CUDA)
    }
#endif

    teca_metadata &omd = out_mesh->get_metadata();
    omd.set("gaussian_filter_hwhm", sigma);
    omd.set("gaussian_filter_center_lat", mu);

    t1 = std::chrono::high_resolution_clock::now();
    seconds_t dt(t1 - t0);

    if (this->get_verbose() && (rank == 0))
    {
        TECA_STATUS("latitude damper computed on "
            << (device_id >= 0 ? "CUDA device" : "the host")
            << "(" << device_id << ") in " << dt.count()
            << " seconds")
    }

    return out_mesh;
}
