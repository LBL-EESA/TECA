#include "teca_regional_moisture_flux.h"

#include "teca_cartesian_mesh.h"
#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

namespace
{
// CPU codes
//

namespace cpu
{
// **************************************************************************
template <typename coord_t>
void transform_coordinates(coord_t *theta, coord_t *phi, coord_t *sin_phi,
    const coord_t *lon, const coord_t *lat, unsigned long nlon, unsigned long nlat)
{
    // transform the cooridnates
    coord_t rad_deg = M_PI / 180.0;

    for (unsigned long i = 0; i < nlon; ++i)
    {
        theta[i] = lon[i] * rad_deg;
    }

    for (unsigned long j = 0; j < nlat; ++j)
    {
        coord_t phi_j = (90.0 - lat[j]) * rad_deg;

        phi[j] = phi_j;

        sin_phi[j] = sin(phi_j);
    }
}


// **************************************************************************
template <typename coord_t, typename data_t>
void divergence(data_t *div, const data_t *ivt_u, const data_t *ivt_v,
    const coord_t *theta, const coord_t *phi, const coord_t *sin_phi,
    unsigned long nlon, unsigned long nlat)
{
    unsigned long nlatm1 = nlat - 1;
    unsigned long nlonm1 = nlon - 1;

    // compute the divergence at every point
    for (unsigned long j = 1; j < nlatm1; ++j)
    {
        for (unsigned long i = 1; i < nlonm1; ++i)
        {
            unsigned long q = j*nlon + i;

            // div = d / d theta ( ivt_theta )
            //     + d / d phi ( ivt_phi sin(phi) )

            // forward difference. using a centered difference here resulted in
            // a larger overall error.
            data_t dthe = theta[i + 1] - theta[i];
            data_t dphi = phi[j + 1] - phi[j];

            data_t divt_the = ivt_u[q + 1] - ivt_u[q];
            data_t divtsin_phi =  ivt_v[q + nlon] * sin_phi[j + 1] - ivt_v[q] * sin_phi[j];

            div[q] = divt_the / dthe + divtsin_phi / dphi;
        }
    }
}

// **************************************************************************
template <typename coord_t, typename data_t>
void integrate(data_t *cell_int, const data_t *div,
    const coord_t *theta, const coord_t *phi,
    unsigned long nlon, unsigned long nlat)
{
    unsigned long nlatm1 = nlat - 1;
    unsigned long nlonm1 = nlon - 1;

    // integrate cell by cell using the trapezoid rule
    for (unsigned long j = 1; j < nlatm1; ++j)
    {
        for (unsigned long i = 1; i < nlonm1; ++i)
        {
            unsigned long q = j*nlon + i;

            // area of the cell
            data_t dthe = theta[i + 1] - theta[i];
            data_t dphi = phi[j + 1] - phi[j];
            data_t dA = dthe * dphi;

            // height of the surface above the cell. using trapezoid here
            // resulted in a larger overall error.
            cell_int[q] = div[q] * dA;
        }
    }
}

// ***************************************************************************
template <typename data_t, typename mask_t>
void sum_if(double &moisture_flux, const data_t *cell_int,
    const mask_t *reg_mask, unsigned long nelem)
{
    moisture_flux = 0.0;
    for (unsigned long q = 0; q < nelem; ++q)
    {
        moisture_flux += reg_mask[q] ? cell_int[q] : data_t();
    }
}

// ***************************************************************************
int dispatch(double &moisture_flux,
    const const_p_teca_variant_array &lon, const const_p_teca_variant_array &lat,
    const const_p_teca_variant_array &ivt_u, const const_p_teca_variant_array &ivt_v,
    const const_p_teca_variant_array &region_mask,
    unsigned long nlon, unsigned long nlat)
{
    NESTED_VARIANT_ARRAY_DISPATCH_FP(lon.get(), _COORDS,
        assert_type<TT_COORDS>(lat);

        unsigned long nelem = nlat*nlon;

        auto [splon, plon] = get_host_accessible<CTT_COORDS>(lon);
        auto [splat, plat] = get_host_accessible<CTT_COORDS>(lat);

        auto [theta, ptheta] = New<TT_COORDS>(nlon);
        auto [phi, pphi] = New<CTT_COORDS>(nlat);
        auto [sin_phi, psin_phi] = New<TT_COORDS>(nlat);

        transform_coordinates(ptheta, pphi, psin_phi, plon, plat, nlon, nlat);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(ivt_u.get(), _DATA,
            assert_type<TT_DATA>(ivt_v);

            auto [spivt_u, pivt_u] = get_host_accessible<CTT_DATA>(ivt_u);
            auto [spivt_v, pivt_v] = get_host_accessible<CTT_DATA>(ivt_v);

            auto [div, pdiv] = New<TT_DATA>(nelem, NT_DATA());
            auto [cint, pcint] = New<TT_DATA>(nelem, NT_DATA());

            // divergance
            divergence(pdiv, pivt_u, pivt_v, ptheta, pphi, psin_phi, nlon, nlat);

            // integrate cell by cell
            integrate(pcint, pdiv, ptheta, pphi, nlon, nlat);

            NESTED_VARIANT_ARRAY_DISPATCH(region_mask.get(), _MASK,

                auto [spreg_mask, preg_mask] = get_host_accessible<CTT_MASK>(region_mask);

                // reduce/sum
                sum_if(moisture_flux, pcint, preg_mask, nelem);

                // note: the - sign here is because we flipped the integration limits in phi
                NT_DATA Re = 6.378100e6;
                moisture_flux *= -Re;

                return 0;
                )
                TECA_ERROR("Unsupported mask data type " << region_mask->get_class_name())
                return -1;
            )
            TECA_ERROR("Unsupported IVT data type " << ivt_u->get_class_name())
            return -1;
        )
        TECA_ERROR("Unsupported coordinate data type " << lon->get_class_name())
        return -1;
}
}




#if defined(TECA_HAS_CUDA)
// CUDA codes
namespace cuda_gpu
{


// **************************************************************************
template <typename coord_t>
__global__
void transform_coordinates(coord_t *theta, coord_t *phi, coord_t *sin_phi,
    const coord_t *lon, const coord_t *lat, unsigned long nlon, unsigned long nlat)
{
    coord_t rad_deg = M_PI / 180.0;

    unsigned long q = teca_cuda_util::thread_id_to_array_index();

    if (q < nlon)
    {
        theta[q] = lon[q] * rad_deg;
    }

    if (q < nlat)
    {
        coord_t phi_q = ( 90.0 - lat[q] ) * rad_deg;

        phi[q] = phi_q;
        sin_phi[q] = sin( phi_q );
    }
}

// **************************************************************************
template <typename coord_t, typename data_t>
__global__
void divergence(data_t *div, const data_t *ivt_u, const data_t *ivt_v,
    const coord_t *theta, const coord_t *phi, const coord_t *sin_phi,
    unsigned long nlon, unsigned long nlat)
{
    unsigned long q = teca_cuda_util::thread_id_to_array_index();

    unsigned long i = q % nlon;
    unsigned long j = q / nlon;

    if ((i < 1) || (i >= nlon - 1) || (j < 1) || (j >= nlat - 1))
        return;

    // div = d / d theta ( ivt_theta )
    //     + d / d phi ( ivt_phi sin(phi) )

    // forward difference. using a centered difference here resulted in
    // a larger overall error.
    data_t dthe = theta[i + 1] - theta[i];
    data_t dphi = phi[j + 1] - phi[j];

    data_t divt_the = ivt_u[q + 1] - ivt_u[q];
    data_t divtsin_phi =  ivt_v[q + nlon] * sin_phi[j + 1] - ivt_v[q] * sin_phi[j];

    div[q] = divt_the / dthe + divtsin_phi / dphi;
}

// **************************************************************************
template <typename coord_t, typename data_t>
__global__
void integrate(data_t *cell_int, const data_t *div,
    const coord_t *theta, const coord_t *phi,
    unsigned long nlon, unsigned long nlat)
{
    unsigned long q = teca_cuda_util::thread_id_to_array_index();

    unsigned long i = q % nlon;
    unsigned long j = q / nlon;

    if ((i < 1) || (i >= nlon - 1) || (j < 1) || (j >= nlat - 1))
        return;

    // cell by cell compute:
    //
    // \iint div dA
    //
    // where:
    //
    // div = d / d theta ( ivt_theta )
    //     + d / d phi ( ivt_phi sin(phi) )
    //
    // dA = d theta d phi
    //

    // area of the cell
    data_t dthe = theta[i + 1] - theta[i];
    data_t dphi = phi[j + 1] - phi[j];

    data_t dA = dthe * dphi;

    // height of the surface above the cell. using trapezoid here
    // resulted in a larger overall error.
    cell_int[q] = div[q] * dA;
}

// **************************************************************************
template <typename data_t, typename mask_t, typename tuple_t = thrust::tuple<data_t, mask_t>>
struct apply_mask : public thrust::unary_function<tuple_t, data_t>
{
    __host__ __device__ data_t operator()(const tuple_t & tup) const
    {
        return thrust::get<1>(tup) ? thrust::get<0>(tup) : data_t();
    }
};

// **************************************************************************
template <typename data_t, typename mask_t>
data_t sum_if(const data_t *data, const mask_t *mask, size_t n_elem)
{
    auto tpdata = thrust::device_pointer_cast(data);
    auto tpmask = thrust::device_pointer_cast(mask);

    return thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(tpdata, tpmask)),
        thrust::make_zip_iterator(thrust::make_tuple(tpdata + n_elem, tpmask + n_elem)),
        apply_mask<data_t, mask_t>(),
        data_t(),
        thrust::plus<data_t>());
        return 0;
}

// **************************************************************************
int dispatch(int device_id, double &moisture_flux,
    const const_p_teca_variant_array &lon, const const_p_teca_variant_array &lat,
    const const_p_teca_variant_array &ivt_u, const const_p_teca_variant_array &ivt_v,
    const const_p_teca_variant_array &region_mask,
    unsigned long nlon, unsigned long nlat)
{
    // set the CUDA device to run on
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
    {
        TECA_ERROR("Failed to set the CUDA device to " << device_id
            << ". " << cudaGetErrorString(ierr))
        return -1;
    }

    // decompose for coordinate transform
    unsigned long coord_nelem = std::max(nlon, nlat);
    dim3 coord_tgrid;
    dim3 coord_bgrid;
    int coord_nblk = 0;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        coord_nelem, 8, coord_bgrid, coord_nblk, coord_tgrid))
    {
        TECA_ERROR("Failed to partition for the coordinate transform")
        return -1;
    }

    // decompose for the divergence calc
    unsigned long data_nelem = nlon*nlat;
    dim3 data_tgrid;
    dim3 data_bgrid;
    int data_nblk = 0;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        data_nelem, 8, data_bgrid, data_nblk, data_tgrid))
    {
        TECA_ERROR("Failed to partition for the divergence calculation")
        return -1;
    }

    NESTED_VARIANT_ARRAY_DISPATCH_FP(lon.get(), _COORDS,
        assert_type<TT_COORDS>(lat);

        // transform the cooridnates
        auto [splon, plon] = get_cuda_accessible<CTT_COORDS>(lon);
        auto [splat, plat] = get_cuda_accessible<CTT_COORDS>(lat);

        auto [theta, ptheta] = New<TT_COORDS>(nlon, allocator::cuda_async);
        auto [phi, pphi] = New<CTT_COORDS>(nlat, allocator::cuda_async);
        auto [sin_phi, psin_phi] = New<TT_COORDS>(nlat, allocator::cuda_async);

        transform_coordinates<<<coord_bgrid,coord_tgrid>>>(ptheta,
             pphi, psin_phi, plon, plat, nlon, nlat);

        if ((ierr = cudaGetLastError()) != cudaSuccess)
        {
            TECA_ERROR("Failed to launch the transform_coordinates CUDA kernel"
                << cudaGetErrorString(ierr))
            return -1;
        }

        NESTED_VARIANT_ARRAY_DISPATCH_FP(ivt_u.get(), _DATA,
            assert_type<TT_DATA>(ivt_v);

            auto [spivt_u, pivt_u] = get_cuda_accessible<CTT_DATA>(ivt_u);
            auto [spivt_v, pivt_v] = get_cuda_accessible<CTT_DATA>(ivt_v);

            auto [div, pdiv] = New<TT_DATA>(data_nelem, NT_DATA(), allocator::cuda_async);
            auto [cint, pcint] = New<TT_DATA>(data_nelem, NT_DATA(), allocator::cuda_async);

            // divergence
            divergence<<<data_bgrid, data_tgrid>>>(pdiv,
                pivt_u, pivt_v, ptheta, pphi, psin_phi, nlon, nlat);

            if ((ierr = cudaGetLastError()) != cudaSuccess)
            {
                TECA_ERROR("Failed to launch the divergence CUDA kernel"
                    << cudaGetErrorString(ierr))
                return -1;
            }

            // integrate cell by cell
            integrate<<<data_bgrid, data_tgrid>>>(pcint, pdiv, ptheta, pphi, nlon, nlat);

            if ((ierr = cudaGetLastError()) != cudaSuccess)
            {
                TECA_ERROR("Failed to launch the integrate CUDA kernel"
                    << cudaGetErrorString(ierr))
                return -1;
            }

            NESTED_VARIANT_ARRAY_DISPATCH(region_mask.get(), _MASK,

                auto [spreg_mask, preg_mask] = get_cuda_accessible<CTT_MASK>(region_mask);

                // sum over the masked region
                moisture_flux = sum_if(pcint, preg_mask, data_nelem);

                // note: the - sign here is because we flipped the integration limits in phi
                NT_DATA Re = 6.378100e6;
                moisture_flux *= -Re;

                return 0;
                )
                TECA_ERROR("Unsupported mask data type " << region_mask->get_class_name())
                return -1;
            )
            TECA_ERROR("Unsupported IVT data type " << ivt_u->get_class_name())
            return -1;
        )
        TECA_ERROR("Unsupported coordinate data type " << lon->get_class_name())
        return -1;
}

}
#endif
}


// --------------------------------------------------------------------------
teca_regional_moisture_flux::teca_regional_moisture_flux() :
    ivt_u_variable(), ivt_v_variable(),
    region_mask_variable(), moisture_flux_variable("regional_moisture_flux")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_regional_moisture_flux::~teca_regional_moisture_flux()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_regional_moisture_flux::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_regional_moisture_flux":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, ivt_u_variable,
            "the name of the longitudinal component of IVT")
        TECA_POPTS_GET(std::string, prefix, ivt_v_variable,
            "the name of the latitudinal component of IVT")
        TECA_POPTS_GET(std::string, prefix, region_mask_variable,
            "the name of the region mask variable")
        TECA_POPTS_GET(std::string, prefix, moisture_flux_variable,
            "the name of the moisture flux variable (output)")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_regional_moisture_flux::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, ivt_u_variable)
    TECA_POPTS_SET(opts, std::string, prefix, ivt_v_variable)
    TECA_POPTS_SET(opts, std::string, prefix, region_mask_variable)
    TECA_POPTS_SET(opts, std::string, prefix, moisture_flux_variable)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_regional_moisture_flux::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_regional_moisture_flux::get_output_metadata" << std::endl;
#endif
    (void)port;

    // check that the class has been configured correctly
    if (this->ivt_u_variable.empty())
    {
        TECA_FATAL_ERROR("The ivt_u_variable was not set")
        return teca_metadata();
    }

    if (this->ivt_v_variable.empty())
    {
        TECA_FATAL_ERROR("The ivt_u_variable was not set")
        return teca_metadata();
    }

    if (this->region_mask_variable.empty())
    {
        TECA_FATAL_ERROR("The region_mask_variable was not set")
        return teca_metadata();
    }


    if (this->moisture_flux_variable.empty())
    {
        TECA_FATAL_ERROR("The moisture_fux_variable was not set")
        return {};
    }

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);
    out_md.append("variables", this->moisture_flux_variable);

    // add attributes for the netcdf writer
    teca_metadata attributes;
    out_md.get("attributes", attributes);

    if (this->moisture_flux_attributes.empty())
    {
        // generate and cache the attributes. they're also needed in execute
        std::ostringstream descr;
        descr << "moisture flux computed from IVT vector (" << this->ivt_u_variable
            << ", " << this->ivt_v_variable << ") over a region defined by the mask "
            << this->region_mask_variable;

        teca_array_attributes atts{teca_variant_array_code<double>::get(),
            teca_array_attributes::no_centering, 0, teca_array_attributes::none_active(),
            "kg s-1", this->moisture_flux_variable, descr.str(), 1, 1.e20};

        attributes.append(this->moisture_flux_variable, (teca_metadata)atts);

        this->moisture_flux_attributes = (teca_metadata)atts;
    }
    else
    {
        // use cached attributes
        attributes.append(this->moisture_flux_variable,
            this->moisture_flux_attributes);
    }

    out_md.set("attributes", attributes);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_regional_moisture_flux::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream requirements and
    // add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->ivt_u_variable);
    arrays.insert(this->ivt_v_variable);
    arrays.insert(this->region_mask_variable);

    // intercept request for our output
    arrays.erase(this->moisture_flux_variable);

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_regional_moisture_flux::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_regional_moisture_flux::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("The input dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // get the input arrays
    auto lon = in_mesh->get_x_coordinates();
    auto lat = in_mesh->get_y_coordinates();

    if (!lon || !lat)
    {
        TECA_FATAL_ERROR("Failed to get the input coordinates")
        return nullptr;
    }

    auto in_arrays = in_mesh->get_point_arrays();
    auto ivt_u = in_arrays->get(this->ivt_u_variable);
    auto ivt_v = in_arrays->get(this->ivt_v_variable);
    auto region_mask = in_arrays->get(this->region_mask_variable);

    if (!ivt_u || !ivt_v)
    {
        TECA_FATAL_ERROR("Failed to get IVT vector components \""
            << this->ivt_u_variable << (ivt_u ? "\" no " : "\" yes ")
            << this->ivt_v_variable << (ivt_v ? "\" no " : "\" yes "))
        return nullptr;
    }

    if (!region_mask)
    {
        TECA_FATAL_ERROR("Failed to get region_mask "
            << this->region_mask_variable)
    }

    auto [nlon, nlat, nlev, nt] = in_mesh->get_array_shape(this->ivt_u_variable);

    double moisture_flux = 0.0;

#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (::cuda_gpu::dispatch(device_id, moisture_flux, lon, lat, ivt_u, ivt_v, region_mask, nlon, nlat))
        {
            TECA_ERROR("Failed to compute the regional moisture flux using CUDA")
            return nullptr;
        }
    }
    else
    {
#endif
        if (::cpu::dispatch(moisture_flux, lon, lat, ivt_u, ivt_v, region_mask, nlon, nlat))
        {
            TECA_ERROR("Failed to compute the regional mositure flux on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif
    // create the output
    p_teca_table out_tab = teca_table::New();

    // pass the calendaring attributes
    std::string calendar;
    std::string time_units;
    unsigned long time_step = 0;
    double time = 0.0;

    in_mesh->get_calendar(calendar);
    in_mesh->get_time_units(time_units);
    in_mesh->get_time_step(time_step);
    in_mesh->get_time(time);

    out_tab->set_calendar(calendar);
    out_tab->set_time_units(time_units);

    // set collumn attributes
    teca_metadata out_atts;
    in_mesh->get_attributes(out_atts);

    out_atts.set(this->moisture_flux_variable, this->moisture_flux_attributes);

    out_tab->set_attributes(out_atts);

    // declare the columns
    out_tab->declare_columns("time_step", (unsigned long)0, "time", double(),
        this->moisture_flux_variable, double());

    // insert the data
    out_tab << time_step << time << moisture_flux;

    return out_tab;
}
