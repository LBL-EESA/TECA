#include "teca_2d_component_area.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <set>
#define _USE_MATH_DEFINES
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

#if defined(TECA_HAS_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace cuda_impl
{
template<typename coord_t, typename label_t>
__global__
void component_area(size_t nx, size_t ny,
    const coord_t * __restrict__ deg_lon, const coord_t * __restrict__ deg_lat,
    const label_t * __restrict__ labels, size_t n_labels, double *area)
{
    // initialize shared memory to use for the calculations
    extern __shared__ double sarea[];

    if (threadIdx.y == 0)
    {
        for (int q = threadIdx.x; q < n_labels; q += blockDim.x)
            sarea[q] = 0.0;
    }

    // wait for all threads to finish initialization
    __syncthreads();

    // get the coordinates of the labeled image for which we are to compute area
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // since we're using a guard cell halo to simplify the calculation, skip the guard halo
    bool threadActive = (i < nx - 1) && (i > 0) && (j < (ny - 1)) && (j > 0);

    if (threadActive)
    {
        // each node in the mesh is treated as a cell in the dual mesh
        // defined by mid points between nodes. the area of the cell is
        // added to the corresponding label.
        //
        // The exact area of the spherical rectangular patch A_i is given by:
        //
        // A_q = rho^2(cos(phi_0) - cos(phi_1))(theta_1 - theta_0)
        //
        //     = rho^2(sin(rad_lat_0) - sin(rad_lat_1))(theta_1 - theta_0)
        //
        // where:
        //   theta = deg_lon * pi/180
        //   phi = pi/2 - rad_lat
        //   rad_lat = deg_lat * pi/180
        //   sin(rad_lat) = cos(pi/2 - rad_lat)

        constexpr double R_e = 6378.1370; // km
        constexpr double R_e_sq = R_e*R_e;
        constexpr double rad_per_deg = M_PI/180.0;

        double cos_phi_1 = sin(0.5*(deg_lat[j - 1] + deg_lat[j    ])*rad_per_deg);
        double cos_phi_0 = sin(0.5*(deg_lat[j    ] + deg_lat[j + 1])*rad_per_deg);
        double d_cos_phi_j = cos_phi_0 - cos_phi_1;

        double rho_sq_d_theta_i = R_e_sq*0.5*(deg_lon[i + 1] - deg_lon[i - 1])*rad_per_deg;

        double A_q = rho_sq_d_theta_i*d_cos_phi_j;

        // get the label
        size_t q = j * nx + i;
        label_t lab = labels[q];

        // udpate the label's area
        atomicAdd(&sarea[lab], A_q);
    }

    // wait for all thread in the block to finish
    __syncthreads();

    // update this block's results in global memory
    if (threadIdx.y == 0)
    {
        for (int i = threadIdx.x; i < n_labels; i += blockDim.x)
            atomicAdd(&area[i], sarea[i]);
    }
}

/** Component area driver.
 *
 *  @param[in] strm the stream to issue work to
 *  @param[in] nlon the number of longitude points
 *  @param[in] nlat the number of latitude points
 *  @param[in] deg_lon the longitude coordinate axis values in decimal degrees
 *  @param[in] deg_lon the latitude coordinate axis values in decimal degrees
 *  @param[in] labels the labeled image
 *  @param[in] n_labels the number of labels
 *  @param[inout] area the array to store label areas in, must be initialized to 0
 *  @returns 0 if successful
 */
template<typename coord_t, typename component_t>
int component_area(cudaStream_t strm,
    size_t nlon, size_t nlat,
    const coord_t *deg_lon, const coord_t *deg_lat,
    const component_t *labels, size_t n_labels,
    double *area)
{
    int ntx = 32;
    int nty = 4;

    int nbx = nlon / ntx + ( nlon % ntx ? 1 : 0 );
    int nby = nlat / nty + ( nlat % nty ? 1 : 0 );

    dim3 blockDim(ntx, nty);
    dim3 gridDim(nbx, nby);

    int sharedBytes = n_labels * sizeof(double);

    component_area<<<gridDim, blockDim, sharedBytes, strm>>>(nlon, nlat, deg_lon, deg_lat,
                                                             labels, n_labels, area);

    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the component area kernel. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}
}
#endif

namespace host_impl
{
template <typename component_t>
component_t get_max_component_id(unsigned long n, const component_t *labels)
{
    component_t max_component_id = std::numeric_limits<component_t>::lowest();
    for (unsigned long i = 0; i < n; ++i)
    {
        component_t label = labels[i];
        max_component_id = label > max_component_id ? label : max_component_id;
    }
    return max_component_id;
}

// visit each node in the mesh, the node is treated as a cell in
// the dual mesh defined by mid points between nodes. the area of
// the cell is added to the corresponding label. this formulation
// requires a layer of ghost nodes
//
// The exact area of the sperical rectangular patch A_i is given by:
//
// A_i = rho^2(cos(phi_0) - cos(phi_1))(theta_1 - theta_0)
//
//     = rho^2(sin(rad_lat_0) - sin(rad_lat_1))(theta_1 - theta_0)
//
// where
//
//   theta = deg_lon * pi/180
//   phi = pi/2 - rad_lat
//   rad_lat = deg_lat * pi/180
//   sin(rad_lat) = cos(pi/2 - rad_lat)
//
template<typename coord_t, typename component_t, typename container_t>
void component_area(unsigned long nlon, unsigned long nlat,
    const coord_t * __restrict__ deg_lon, const coord_t * __restrict__ deg_lat,
    const component_t * __restrict__ labels, container_t &area)
{
    // This calculation is sensative to floating point precision and
    // should be done in double precision
    using calc_t = double;

    calc_t R_e = 6378.1370; // km
    calc_t R_e_sq = R_e*R_e;
    calc_t rad_per_deg = M_PI/180.0;

    unsigned long nlonm1 = nlon - 1;
    unsigned long nlatm1 = nlat - 1;

    // convert to spherical coordinates in units of radians,
    // move to the dual mesh, and pre-compute factors in A
    calc_t *rho_sq_d_theta = (calc_t*)malloc(nlon*sizeof(calc_t));
    rho_sq_d_theta[0] = calc_t();
    for (unsigned long i = 1; i < nlonm1; ++i)
        rho_sq_d_theta[i] = R_e_sq*calc_t(0.5)*(deg_lon[i + 1] - deg_lon[i - 1])*rad_per_deg;
    rho_sq_d_theta[nlonm1] = calc_t();

    calc_t *rad_lat = (calc_t*)malloc(nlat*sizeof(calc_t));
    for (unsigned long j = 0; j < nlat; ++j)
        rad_lat[j] = deg_lat[j]*rad_per_deg;

    calc_t *d_cos_phi = (calc_t*)malloc(nlat*sizeof(calc_t));
    for (unsigned long j = 1; j < nlatm1; ++j)
    {
        calc_t cos_phi_1 = sin(calc_t(0.5)*(rad_lat[j - 1] + rad_lat[j]));
        calc_t cos_phi_0 = sin(calc_t(0.5)*(rad_lat[j] + rad_lat[j + 1]));
        d_cos_phi[j] = cos_phi_0 - cos_phi_1;
    }
    d_cos_phi[0] = calc_t();
    d_cos_phi[nlatm1] = calc_t();

    // finish off the calc by multiplying the factors
    for (unsigned long j = 1; j < nlatm1; ++j)
    {
        calc_t d_cos_phi_j = d_cos_phi[j];
        unsigned long jj = j*nlon;
        for (unsigned long i = 1; i < nlonm1; ++i)
        {
            area[labels[jj + i]] += rho_sq_d_theta[i]*d_cos_phi_j;
        }
    }

    free(rad_lat);
    free(d_cos_phi);
    free(rho_sq_d_theta);
}

}



// --------------------------------------------------------------------------
teca_2d_component_area::teca_2d_component_area() :
    component_variable(""), contiguous_component_ids(0), background_id(-1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_2d_component_area::~teca_2d_component_area()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_2d_component_area::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_2d_component_area":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_variable,
            "name of the varibale containing region labels")
        TECA_POPTS_GET(int, prefix, contiguous_component_ids,
            "when the region label ids start at 0 and are consecutive "
            "this flag enables use of an optimization")
        TECA_POPTS_GET(long, prefix, background_id,
            "the label id that corresponds to the background")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_2d_component_area::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, component_variable)
    TECA_POPTS_SET(opts, int, prefix, contiguous_component_ids)
    TECA_POPTS_SET(opts, long, prefix, background_id)
}
#endif

// --------------------------------------------------------------------------
int teca_2d_component_area::get_component_variable(std::string &component_var)
{
    if (this->component_variable.empty())
        return -1;

    component_var = this->component_variable;
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_2d_component_area::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_2d_component_area::get_output_metadata" << endl;
#endif
    (void) port;

    teca_metadata md = input_md[0];
    return md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_2d_component_area::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_2d_component_area::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string component_var;
    if (this->get_component_variable(component_var))
    {
        TECA_FATAL_ERROR("component_variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(component_var);

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_2d_component_area::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_2d_component_area::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);
    if (!in_mesh)
    {
        TECA_FATAL_ERROR("empty input, or not a cartesian_mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array
    std::string component_var;
    if (this->get_component_variable(component_var))
    {
        TECA_FATAL_ERROR("component_variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array component_array
        = out_mesh->get_point_arrays()->get(component_var);
    if (!component_array)
    {
        TECA_FATAL_ERROR("label variable \"" << component_var
            << "\" is not in the input")
        return nullptr;
    }

    // get mesh dimension
    unsigned long extent[6];
    out_mesh->get_extent(extent);

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nxy = nx*ny;

    unsigned long nz = extent[5] - extent[4] + 1;
    if (nz != 1)
    {
        TECA_FATAL_ERROR("This calculation requires 2D data. The current dataset "
            "extents are [" << extent[0] << ", " << extent[1] << ", "
            << extent[2] << ", " << extent[3] << ", " << extent[4] << ", "
            << extent[5] << "]")
        return nullptr;
    }

    // get the coordinate axes
    const_p_teca_variant_array xc = in_mesh->get_x_coordinates();
    const_p_teca_variant_array yc = in_mesh->get_y_coordinates();

    // get the input and output metadata
    teca_metadata &in_metadata =
        const_cast<teca_metadata&>(in_mesh->get_metadata());

    teca_metadata &out_metadata = out_mesh->get_metadata();

    // get the background_id, and pass it through
    long bg_id = this->background_id;
    if (this->background_id == -1)
    {
        if (in_metadata.get("background_id", bg_id))
        {
            TECA_FATAL_ERROR("Metadata is missing the key \"background_id\". "
                "One should specify it via the \"background_id\" algorithm "
                "property")
            return nullptr;
        }
    }
    out_metadata.set("background_id", bg_id);

    // look for the list of components.
    bool has_component_ids = in_metadata.has("component_ids");

    // determine if the calculation runs on the cpu or gpu
#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);

    // our GPU implementation requires contiguous component ids
    if ((device_id >= 0) && !(has_component_ids || this->contiguous_component_ids))
    {
        TECA_WARNING("Requested execution on device " << device_id
            << ". Execution moved to host because of non-contiguous"
            " component ids.")
        device_id = -1;
    }
#endif

    // calculate area of components
    NESTED_VARIANT_ARRAY_DISPATCH_FP(
        xc.get(), _COORD,

        // the calculation is sensative to floating point precision
        // and should be made in double precision
        using NT_CALC = double;
        using TT_CALC = teca_double_array;

        // get the cooridnate arrays
        assert_type<CTT_COORD>(yc);

        NESTED_VARIANT_ARRAY_DISPATCH_I(
            component_array.get(), _LABEL,

#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                if (teca_cuda_util::set_device(device_id))
                    return nullptr;

                auto [sp_xc, p_xc, sp_yc, p_yc] = get_cuda_accessible<CTT_COORD>(xc, yc);
                auto [sp_labels, p_labels] = get_cuda_accessible<CTT_LABEL>(component_array);

                // use a contiguous buffer to hold the result, only for
                // contiguous lables that start at 0
                p_teca_variant_array component_id;
                unsigned int n_labels = 0;

                if (has_component_ids)
                {
                    component_id = in_metadata.get("component_ids");
                    in_metadata.get("number_of_components", int(0), n_labels);
                }
                else
                {
                    auto ep = thrust::cuda::par.on(cudaStreamPerThread);

                    NT_LABEL max_component_id = thrust::reduce(ep, p_labels, p_labels + nxy,
                                                               std::numeric_limits<NT_LABEL>::lowest(),
                                                               thrust::maximum<NT_LABEL>());
                    n_labels = max_component_id + 1;

                    auto [tmp, ptmp] = ::New<TT_LABEL>(n_labels, allocator::cuda_async);

                    thrust::sequence(ep, ptmp, ptmp + n_labels, NT_LABEL(0), NT_LABEL(1));

                    component_id = tmp;
                }

                auto [component_area, pcomponent_area] = ::New<teca_double_array>(n_labels, 0.0, allocator::cuda_async);

                cuda_impl::component_area(cudaStreamPerThread, nx,ny, p_xc,p_yc, p_labels, n_labels, pcomponent_area);

                // transfer the result to the output
                out_metadata.set("number_of_components", n_labels);
                out_metadata.set("component_ids", component_id);
                out_metadata.set("component_area", component_area);
            }
            else
            {
#endif
                unsigned int n_labels = 0;

                auto [sp_xc, p_xc, sp_yc, p_yc] = get_host_accessible<CTT_COORD>(xc, yc);
                auto [sp_labels, p_labels] = get_host_accessible<CTT_LABEL>(component_array);

                if (this->contiguous_component_ids || has_component_ids)
                {
                    // use a contiguous buffer to hold the result, only for
                    // contiguous lables that start at 0
                    p_teca_variant_array component_id;

                    if (has_component_ids)
                    {
                        component_id = in_metadata.get("component_ids");
                        in_metadata.get("number_of_components", int(0), n_labels);
                    }
                    else
                    {
                        NT_LABEL max_component_id = host_impl::get_max_component_id(nxy, p_labels);
                        n_labels = max_component_id + 1;

                        auto [tmp, ptmp] = ::New<TT_LABEL>(n_labels);

                        for (unsigned int i = 0; i < n_labels; ++i)
                            ptmp[i] = NT_LABEL(i);

                        component_id = tmp;
                    }

                    std::vector<NT_CALC> component_area(n_labels);

                    sync_host_access_any(xc, yc, component_array);

                    host_impl::component_area(nx,ny, p_xc,p_yc, p_labels, component_area);

                    // transfer the result to the output
                    out_metadata.set("number_of_components", n_labels);
                    out_metadata.set("component_ids", component_id);
                    out_metadata.set("component_area", component_area);
                }
                else
                {
                    // use an associative array to handle any labels
                    std::map<NT_LABEL, NT_CALC> result;

                    sync_host_access_any(xc, yc, component_array);

                    host_impl::component_area(nx,ny, p_xc,p_yc, p_labels, result);

                    // transfer the result to the output
                    unsigned int n_labels = result.size();

                    auto [component_id, pcomponent_id] = ::New<TT_LABEL>(n_labels);
                    auto [component_area, pcomponent_area] = ::New<TT_CALC>(n_labels);

                    auto it = result.begin();
                    for (unsigned int i = 0; i < n_labels; ++i,++it)
                    {
                        pcomponent_id[i] = it->first;
                        pcomponent_area[i] = it->second;
                    }

                    out_metadata.set("number_of_components", n_labels);
                    out_metadata.set("component_ids", component_id);
                    out_metadata.set("component_area", component_area);
                }
#if defined(TECA_HAS_CUDA)
            }
#endif
            )
        )

    return out_mesh;
}
