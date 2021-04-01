#include "teca_integrated_vapor_transport.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::cos;

//#define TECA_DEBUG

namespace {
template <typename coord_t, typename num_t>
void cartesian_ivt(unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const num_t *q, num_t *ivt)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    // compute the integrand
    num_t *f = (num_t*)malloc(nxyz*sizeof(num_t));
    for (unsigned long i = 0; i < nxyz; ++i)
        f[i] = wind[i]*q[i];

    // initialize the result
    memset(ivt, 0, nxy*sizeof(num_t));

    // work an x-y slice at  a time
    unsigned long nzm1 = nz - 1;
    for (unsigned long k = 0; k < nzm1; ++k)
    {
        // dp over the slice
        num_t h2 = num_t(0.5) * (plev[k+1] - plev[k]);

        // the current two x-y-planes of data
        unsigned long knxy = k*nxy;
        num_t *f_k0 = f + knxy;
        num_t *f_k1 = f_k0 + nxy;

        // accumulate this plane of data using trapazoid rule
        for (unsigned long q = 0; q < nxy; ++q)
        {
            ivt[q] += h2 * (f_k0[q] + f_k1[q]);
        }
    }

    // free up the integrand
    free(f);

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);
    for (unsigned long i = 0; i < nxy; ++i)
        ivt[i] *= m1g;
}

template <typename coord_t, typename num_t>
void cartesian_ivt(unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const char *wind_valid, const num_t *q, const char *q_valid,
    num_t *ivt)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    // compute the mask
    char *mask = (char*)malloc(nxyz);
    for (unsigned long i = 0; i < nxyz; ++i)
        mask[i] = wind_valid[i] && q_valid[i] ? 1 : 0;

    // compute the integrand
    num_t *f = (num_t*)malloc(nxyz*sizeof(num_t));
    for (unsigned long i = 0; i < nxyz; ++i)
        f[i] = wind[i]*q[i];

    // initialize the result
    memset(ivt, 0, nxy*sizeof(num_t));

    // work an x-y slice at a time
    unsigned long nzm1 = nz - 1;
    for (unsigned long k = 0; k < nzm1; ++k)
    {
        // dp over the slice
        num_t h2 = num_t(0.5) * (plev[k+1] - plev[k]);

        // the current two x-y-planes of data
        unsigned long knxy = k*nxy;
        num_t *f_k0 = f + knxy;
        num_t *f_k1 = f_k0 + nxy;

        char *mask_k0 = mask + knxy;
        char *mask_k1 = mask_k0 + nxy;

        // accumulate this plane of data using trapazoid rule
        for (unsigned long q = 0; q < nxy; ++q)
        {
            ivt[q] += ((mask_k0[q] && mask_k1[q]) ?
               h2 * (f_k0[q] + f_k1[q]) : num_t(0));
        }
    }

    // free up the integrand and mask
    free(mask);
    free(f);

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);
    for (unsigned long i = 0; i < nxy; ++i)
        ivt[i] *= m1g;
}
}

// --------------------------------------------------------------------------
teca_integrated_vapor_transport::teca_integrated_vapor_transport() :
    wind_u_variable("ua"), wind_v_variable("va"),
    specific_humidity_variable("hus"), ivt_u_variable("ivt_u"),
    ivt_v_variable("ivt_v"), fill_value(1.0e20)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_integrated_vapor_transport::~teca_integrated_vapor_transport()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_integrated_vapor_transport::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_integrated_vapor_transport":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, wind_u_variable,
            "name of the variable containg the lon component of the wind vector")
        TECA_POPTS_GET(std::string, prefix, wind_v_variable,
            "name of the variable containg the lat component of the wind vector")
        TECA_POPTS_GET(std::string, prefix, specific_humidity_variable,
            "name of the variable containg the specific humidity")
        TECA_POPTS_GET(double, prefix, fill_value,
            "the value of the NetCDF _FillValue attribute")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_integrated_vapor_transport::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, wind_u_variable)
    TECA_POPTS_SET(opts, std::string, prefix, wind_v_variable)
    TECA_POPTS_SET(opts, std::string, prefix, specific_humidity_variable)
    TECA_POPTS_SET(opts, double, prefix, fill_value)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_integrated_vapor_transport::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_integrated_vapor_transport::get_output_metadata" << std::endl;
#endif
    (void)port;

    // set things up in the first pass, and don't modify in subsequent passes
    // due to threading concerns

    if (this->get_number_of_derived_variables() == 0)
    {
        // the base class will handle dealing with the transformation of
        // mesh dimensions and reporting the array we produce, but we have
        // to determine the data type and tell the name of the produced array.
        const teca_metadata &md = input_md[0];

        teca_metadata attributes;
        if (md.get("attributes", attributes))
        {
            TECA_ERROR("Failed to determine output data type "
                "because attributes are misisng")
            return teca_metadata();
        }

        teca_metadata u_atts;
        if (attributes.get(this->wind_u_variable, u_atts))
        {
            TECA_ERROR("Failed to determine output data type "
                "because attributes for \"" << this->wind_u_variable
                << "\" are misisng")
            return teca_metadata();
        }

        int type_code = 0;
        if (u_atts.get("type_code", type_code))
        {
            TECA_ERROR("Failed to determine output data type "
                "because attributes for \"" << this->wind_u_variable
                << "\" is misisng a \"type_code\"")
            return teca_metadata();
        }

        teca_array_attributes ivt_u_atts(
            type_code, teca_array_attributes::point_centering,
            0, "kg m^{-1} s^{-1}", "longitudinal integrated vapor transport",
            "the longitudinal component of integrated vapor transport",
            1, this->fill_value);

        teca_array_attributes ivt_v_atts(
            type_code, teca_array_attributes::point_centering,
            0, "kg m^{-1} s^{-1}", "latitudinal integrated vapor transport",
            "the latitudinal component of integrated vapor transport",
            this->fill_value);

        // install name and attributes of the output variables in the base classs
        this->append_derived_variable(this->ivt_u_variable);
        this->append_derived_variable(this->ivt_v_variable);

        this->append_derived_variable_attribute(ivt_u_atts);
        this->append_derived_variable_attribute(ivt_v_atts);

    }

    if (this->get_number_of_dependent_variables() == 0)
    {
        // install the names of the input variables in the base class
        this->append_dependent_variable(this->wind_u_variable);
        this->append_dependent_variable(this->wind_v_variable);
        this->append_dependent_variable(this->specific_humidity_variable);
    }

    // invoke the base class method, which does the work of transforming
    // the mesh and reporting the variables and their attributes.
    return teca_vertical_reduction::get_output_metadata(port, input_md);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_integrated_vapor_transport::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    // invoke the base class method
    return teca_vertical_reduction::get_upstream_request(port, input_md, request);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_integrated_vapor_transport::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_integrated_vapor_transport::execute" << std::endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("Failed to compute IVT because a cartesian mesh is required.")
        return nullptr;
    }

    // get the input dimensions
    unsigned long extent[6] = {0};
    if (in_mesh->get_extent(extent))
    {
        TECA_ERROR("Failed to compute IVT because mesh extent is missing.")
        return nullptr;
    }

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

    // get the pressure coordinates
    const_p_teca_variant_array p = in_mesh->get_z_coordinates();
    if (!p)
    {
        TECA_ERROR("Failed to compute IVT because pressure coordinates are missing")
        return nullptr;
    }

    if (p->size() < 2)
    {
        TECA_ERROR("Failed to compute IVT because z dimensions "
            << p->size() << " < 2 as required by the integration method")
        return nullptr;
    }

    // gather the input arrays
    const_p_teca_variant_array wind_u =
        in_mesh->get_point_arrays()->get(this->wind_u_variable);

    if (!wind_u)
    {
        TECA_ERROR("Failed to compute IVT because longitudinal wind \""
            << this->wind_u_variable << "\" is missing")
        return nullptr;
    }

    const_p_teca_variant_array wind_u_valid =
           in_mesh->get_point_arrays()->get(this->wind_u_variable + "_valid");

    const_p_teca_variant_array wind_v =
        in_mesh->get_point_arrays()->get(this->wind_v_variable);

    if (!wind_v)
    {
        TECA_ERROR("Failed to compute IVT because latitudinal wind \""
            << this->wind_v_variable << "\" is missing")
        return nullptr;
    }

    const_p_teca_variant_array wind_v_valid =
           in_mesh->get_point_arrays()->get(this->wind_v_variable + "_valid");

    const_p_teca_variant_array q =
        in_mesh->get_point_arrays()->get(this->specific_humidity_variable);

    if (!q)
    {
        TECA_ERROR("Failed to compute IVT because specific humidity \""
            << this->specific_humidity_variable << "\" is missing")
        return nullptr;
    }

    const_p_teca_variant_array q_valid =
           in_mesh->get_point_arrays()->get(this->specific_humidity_variable + "_valid");

    // the base class will construct the output mesh
    p_teca_cartesian_mesh out_mesh
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(
                teca_vertical_reduction::execute(port, input_data, request)));

    if (!out_mesh)
    {
        TECA_ERROR("Failed to compute IVT because the output mesh was "
            "not constructed")
        return nullptr;
    }

    // allocate the output arrays
    unsigned long nxy = nx*ny;
    p_teca_variant_array ivt_u = wind_u->new_instance(nxy);
    p_teca_variant_array ivt_v = wind_u->new_instance(nxy);

    // store the result
    out_mesh->get_point_arrays()->set(this->ivt_u_variable, ivt_u);
    out_mesh->get_point_arrays()->set(this->ivt_v_variable, ivt_v);

    // calculate IVT
    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        p.get(), _COORDS,

        const NT_COORDS *p_p = static_cast<TT_COORDS*>(p.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
            ivt_u.get(), _DATA,

            NT_DATA *p_ivt_u = static_cast<TT_DATA*>(ivt_u.get())->get();
            NT_DATA *p_ivt_v = static_cast<TT_DATA*>(ivt_v.get())->get();

            const NT_DATA *p_wind_u = static_cast<const TT_DATA*>(wind_u.get())->get();
            const NT_DATA *p_wind_v = static_cast<const TT_DATA*>(wind_v.get())->get();
            const NT_DATA *p_q = static_cast<const TT_DATA*>(q.get())->get();

            const char *p_wind_u_valid = nullptr;
            const char *p_wind_v_valid = nullptr;
            const char *p_q_valid = nullptr;
            if (wind_u_valid)
            {
                using TT_MASK = teca_char_array;

                p_wind_u_valid = dynamic_cast<const TT_MASK*>(wind_u_valid.get())->get();
                p_wind_v_valid = dynamic_cast<const TT_MASK*>(wind_v_valid.get())->get();
                p_q_valid = dynamic_cast<const TT_MASK*>(q_valid.get())->get();

                ::cartesian_ivt(nx, ny, nz, p_p, p_wind_u, p_wind_u_valid, p_q, p_q_valid, p_ivt_u);
                ::cartesian_ivt(nx, ny, nz, p_p, p_wind_v, p_wind_v_valid, p_q, p_q_valid, p_ivt_v);
            }
            else
            {
                ::cartesian_ivt(nx, ny, nz, p_p, p_wind_u, p_q, p_ivt_u);
                ::cartesian_ivt(nx, ny, nz, p_p, p_wind_v, p_q, p_ivt_v);
            }
            )
        )

    // pass 2D arrays through.
    p_teca_array_collection in_arrays =
        std::const_pointer_cast<teca_array_collection>(in_mesh->get_point_arrays());

    p_teca_array_collection out_arrays = out_mesh->get_point_arrays();

    int n_arrays = in_arrays->size();
    for (int i = 0; i < n_arrays; ++i)
    {
        p_teca_variant_array array = in_arrays->get(i);
        if (array->size() == nxy)
        {
            // pass the array.
            out_arrays->append(in_arrays->get_name(i), array);
        }
    }

    return out_mesh;
}
