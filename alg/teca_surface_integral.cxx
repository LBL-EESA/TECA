#include "teca_surface_integral.h"

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
/// CPU implementation
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
void integrate(data_t *cell_int, const data_t *sflux,
    const coord_t *theta, const coord_t *phi, const coord_t *sin_phi,
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
            data_t dA = sin_phi[j] * dthe * dphi;

            // height of the surface above the cell.
            cell_int[q] = sflux[q] * dA;
        }
    }
}

// ***************************************************************************
template <typename data_t, typename mask_t>
void sum_if(double &net_flux, const data_t *cell_int,
    const mask_t *reg_mask, unsigned long nelem)
{
    net_flux = 0.0;
    for (unsigned long q = 0; q < nelem; ++q)
    {
        net_flux += reg_mask[q] ? cell_int[q] : data_t();
    }
}

// ***************************************************************************
int dispatch(std::vector<double> &net_flux,
    const const_p_teca_variant_array &lon, const const_p_teca_variant_array &lat,
    const std::vector<const_p_teca_variant_array> &surface_flux,
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

        unsigned int n_arrays = surface_flux.size();
        net_flux.resize(n_arrays);

        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            const const_p_teca_variant_array &sflux = surface_flux[i];

            NESTED_VARIANT_ARRAY_DISPATCH_FP(sflux.get(), _DATA,

                auto [spsflux, psflux] = get_host_accessible<CTT_DATA>(sflux);

                auto [cint, pcint] = New<TT_DATA>(nelem, NT_DATA());

                // integrate cell by cell
                integrate(pcint, psflux, ptheta, pphi, psin_phi, nlon, nlat);

                NESTED_VARIANT_ARRAY_DISPATCH(region_mask.get(), _MASK,

                    auto [spreg_mask, preg_mask] = get_host_accessible<CTT_MASK>(region_mask);

                    // reduce/sum
                    double reg_total;
                    sum_if(reg_total, pcint, preg_mask, nelem);

                    // - because limits of integration are flipped in our coordinate system
                    NT_DATA Re2 = 4.0680159610000e13;
                    net_flux[i] = -Re2 * reg_total;
                    )
                else
                {
                    TECA_ERROR("Unsupported mask data type " << region_mask->get_class_name())
                    return -1;
                }
                )
            else
            {
                TECA_ERROR("Unsupported surface flux data type " << sflux->get_class_name())
                return -1;
            }
        }
        )
    else
    {
        TECA_ERROR("Unsupported coordinate data type " << lon->get_class_name())
        return -1;
    }

    return 0;
}
}


#if defined(TECA_HAS_CUDA)
/// CUDA implementation
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
void integrate(data_t *cell_int, const data_t *sflux,
    const coord_t *theta, const coord_t *phi, const coord_t *sin_phi,
    unsigned long nlon, unsigned long nlat)
{
    unsigned long q = teca_cuda_util::thread_id_to_array_index();

    unsigned long i = q % nlon;
    unsigned long j = q / nlon;

    if ((i < 1) || (i >= nlon - 1) || (j < 1) || (j >= nlat - 1))
        return;

    // area of the cell
    data_t dthe = theta[i + 1] - theta[i];
    data_t dphi = phi[j + 1] - phi[j];

    data_t dA = sin_phi[j] * dthe * dphi;

    // height of the surface above the cell.
    cell_int[q] = sflux[q] * dA;
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
int dispatch(int device_id, std::vector<double> &net_flux,
    const const_p_teca_variant_array &lon, const const_p_teca_variant_array &lat,
    const std::vector<const_p_teca_variant_array> &surface_flux,
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

    // decompose for the integral
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

        unsigned int n_arrays = surface_flux.size();
        net_flux.resize(n_arrays);

        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            const const_p_teca_variant_array &sflux = surface_flux[i];

            NESTED_VARIANT_ARRAY_DISPATCH_FP(sflux.get(), _DATA,

                auto [spsflux, psflux] = get_cuda_accessible<CTT_DATA>(sflux);

                auto [cint, pcint] = New<TT_DATA>(data_nelem, NT_DATA(), allocator::cuda_async);

                // integrate cell by cell
                integrate<<<data_bgrid, data_tgrid>>>(pcint, psflux, ptheta, pphi, psin_phi, nlon, nlat);

                if ((ierr = cudaGetLastError()) != cudaSuccess)
                {
                    TECA_ERROR("Failed to launch the integrate CUDA kernel"
                        << cudaGetErrorString(ierr))
                    return -1;
                }

                NESTED_VARIANT_ARRAY_DISPATCH(region_mask.get(), _MASK,

                    auto [spreg_mask, preg_mask] = get_cuda_accessible<CTT_MASK>(region_mask);

                    // sum over the masked region
                    double reg_total = sum_if(pcint, preg_mask, data_nelem);

                    // - sign because we flipped the limits of integration
                    NT_DATA Re2 = 4.0680159610000e13;
                    net_flux[i] = -Re2 * reg_total;
                    )
                else
                {
                    TECA_ERROR("Unsupported mask data type " << region_mask->get_class_name())
                    return -1;
                }
                )
            else
            {
                TECA_ERROR("Unsupported surface flux data type " << sflux->get_class_name())
                return -1;
            }
        }
        )
    else
    {
        TECA_ERROR("Unsupported coordinate data type " << lon->get_class_name())
        return -1;
    }

    return 0;
}

}
#endif
}


// --------------------------------------------------------------------------
teca_surface_integral::teca_surface_integral() : input_variables(),
    output_variables(), region_mask_variable(), output_prefix("net_regional_")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_surface_integral::~teca_surface_integral()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_surface_integral::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_surface_integral":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, input_variables,
            "the list of surface flux variables to process")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, output_variables,
            "the list of names for the net flux over the region of each input")
        TECA_POPTS_GET(std::string, prefix, region_mask_variable,
            "the name of the region mask variable")
        TECA_POPTS_GET(std::string, prefix, output_prefix,
            "a string prepended to the output variable names")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_surface_integral::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, input_variables)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, output_variables)
    TECA_POPTS_SET(opts, std::string, prefix, region_mask_variable)
    TECA_POPTS_SET(opts, std::string, prefix, output_prefix)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_surface_integral::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_surface_integral::get_output_metadata" << std::endl;
#endif
    (void)port;

    // check that the class has been configured correctly
    if (this->input_variables.empty())
    {
        TECA_FATAL_ERROR("The surface flux variables were not set")
        return {};
    }

    if (!this->output_variables.empty() &&
        (this->output_variables.size() != this->input_variables.size()))
    {
        TECA_FATAL_ERROR("Each flux variable must have a coresponding"
            " output variable")
        return {};
    }

    if (this->region_mask_variable.empty())
    {
        TECA_FATAL_ERROR("The region_mask_variable was not set")
        return {};
    }

    // copy the report
    teca_metadata out_md(input_md[0]);

    const teca_metadata &md_in = input_md[0];

    teca_metadata attributes;
    if (md_in.get("attributes", attributes))
    {
        TECA_FATAL_ERROR("Metadata issue, missing attributes")
        return {};
    }

    unsigned int n_vars = this->input_variables.size();

    bool gen_attributes = this->output_attributes.empty();
    if (gen_attributes)
        this->output_attributes.resize(n_vars);

    for (unsigned int i = 0; i < n_vars; ++i)
    {
        const std::string &var = this->input_variables[i];

        std::string var_out;

        if (this->output_variables.empty())
            var_out = this->output_prefix + var;
        else
            var_out = this->output_variables[i];

        // report the variable we produce
        out_md.append("variables", var_out);

        if (gen_attributes)
        {
            // get the input variable's attributes
            teca_metadata atts;
            if (attributes.get(var, atts))
            {
                TECA_FATAL_ERROR("No attributes for \"" << var << "\"")
                continue;
            }

            // get the input units
            std::string units;
            atts.get("units", units);

            // remove area from the units
            size_t pos = units.find("m-2");
            if (pos != std::string::npos)
            {
                size_t len = 3;
                while (pos && (units[pos-1] == ' ')){ --pos; ++len; }
                units.erase(pos, len);
            }
            else if ((pos = units.find("m^{-2}")) != std::string::npos)
            {
                size_t len = 6;
                while (pos && (units[pos-1] == ' ')){ --pos; ++len; }
                units.erase(pos, len);
            }

            // add a decription
            std::ostringstream descr;
            descr << "The net flux of " << var
                << " over a region defined by the mask "
                << this->region_mask_variable;

            teca_array_attributes out_atts{teca_variant_array_code<double>::get(),
                teca_array_attributes::no_centering, 0,
                teca_array_attributes::none_active(), units, var_out, descr.str(),
                1, 1.e20};

            // set the output variable's attributes
            attributes.set(var_out, (teca_metadata)out_atts);

            // cache the output attributes since the writer needs them during execute.
            this->output_attributes[i] = out_atts;
        }
        else
        {
            // set the output variable's attributes
            attributes.set(var_out, (teca_metadata)this->output_attributes[i]);
        }
    }

    out_md.set("attributes", attributes);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_surface_integral::get_upstream_request(
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

    // intercept request for the arrays we produce
    unsigned int n_vars = this->input_variables.size();
    for (unsigned int i = 0; i < n_vars; ++i)
    {
        if (this->output_variables.empty())
            arrays.erase(this->output_prefix + this->input_variables[i]);
        else
            arrays.erase(this->output_variables[i]);
    }

    // reuest the arrays we need
    arrays.insert(this->region_mask_variable);
    for (unsigned int i = 0; i < n_vars; ++i)
        arrays.insert(this->input_variables[i]);

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_surface_integral::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_surface_integral::execute" << std::endl;
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

    out_tab->declare_columns("time_step", (unsigned long)0, "time", double());

    // get the mesh coordinates
    auto lon = in_mesh->get_x_coordinates();
    auto lat = in_mesh->get_y_coordinates();

    // get the region mask
    auto in_arrays = in_mesh->get_point_arrays();

    auto region_mask = in_arrays->get(this->region_mask_variable);
    if (!region_mask)
    {
        TECA_FATAL_ERROR("Region mask \"" << this->region_mask_variable
            << "\" was not found in the input")
        return nullptr;
    }

    teca_metadata attributes;

    // package up the flux inputs
    unsigned int n_arrays = this->input_variables.size();

    std::vector<double> net_flux(n_arrays);
    std::vector<const_p_teca_variant_array> surface_flux(n_arrays);

    for (unsigned int i = 0; i < n_arrays; ++i)
    {
        // get the input arrays
        const std::string &in_var = this->input_variables[i];

        surface_flux[i] = in_arrays->get(in_var);
        if (!surface_flux[i])
        {
            TECA_FATAL_ERROR("Array " << in_var << " was not found in the input")
            return nullptr;
        }

        // allocate the output columns, and pass the attributes
        if (this->output_variables.empty())
        {
            std::string out_var = this->output_prefix + in_var;
            out_tab->declare_column(out_var, double());

            attributes.set(out_var, (teca_metadata)this->output_attributes[i]);
        }
        else
        {
            out_tab->declare_column(this->output_variables[i], double());

            attributes.set(this->output_variables[i],
                (teca_metadata)this->output_attributes[i]);
        }
    }

    // set the array attributes
    out_tab->set_attributes(attributes);

    // get the mesh dimensions
    auto [nlon, nlat, nlev, nt] =
        in_mesh->get_array_shape(this->input_variables[0]);

#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (::cuda_gpu::dispatch(device_id, net_flux, lon, lat, surface_flux, region_mask, nlon, nlat))
        {
            TECA_ERROR("Failed to compute the regional moisture flux using CUDA")
            return nullptr;
        }
    }
    else
    {
#endif
        if (::cpu::dispatch(net_flux, lon, lat, surface_flux, region_mask, nlon, nlat))
        {
            TECA_ERROR("Failed to compute the regional mositure flux on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif

    // insert the data into the output
    out_tab << time_step << time;
    for (unsigned int i = 0; i < n_arrays; ++i)
        out_tab << net_flux[i];

    return out_tab;
}
