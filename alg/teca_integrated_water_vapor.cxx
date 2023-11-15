#include "teca_integrated_water_vapor.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"
#include "teca_valid_value_mask.h"
#include "teca_array_attributes.h"

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

using namespace teca_variant_array_util;

//#define TECA_DEBUG

namespace {
template <typename coord_t, typename num_t>
void cartesian_iwv(unsigned long nx, unsigned long ny, unsigned long nz,
    const coord_t *plev, const num_t *q, num_t *iwv)
{
    unsigned long nxy = nx*ny;

    // initialize the result
    memset(iwv, 0, nxy*sizeof(num_t));

    // work an x-y slice at  a time
    unsigned long nzm1 = nz - 1;
    for (unsigned long k = 0; k < nzm1; ++k)
    {
        // dp over the slice
        num_t h2 = num_t(0.5) * (plev[k+1] - plev[k]);

        // the current two x-y-planes of data
        unsigned long knxy = k*nxy;
        const num_t *q_k0 = q + knxy;
        const num_t *q_k1 = q_k0 + nxy;

        // accumulate this plane of data using trapazoid rule
        for (unsigned long i = 0; i < nxy; ++i)
        {
            iwv[i] += h2 * (q_k0[i] + q_k1[i]);
        }
    }

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);
    for (unsigned long i = 0; i < nxy; ++i)
        iwv[i] *= m1g;
}

template <typename coord_t, typename num_t>
void cartesian_iwv(unsigned long nx, unsigned long ny, unsigned long nz,
    const coord_t *plev, const num_t *q, const char *q_valid, num_t *iwv)
{
    unsigned long nxy = nx*ny;

    // initialize the result
    memset(iwv, 0, nxy*sizeof(num_t));

    // work an x-y slice at a time
    unsigned long nzm1 = nz - 1;
    for (unsigned long k = 0; k < nzm1; ++k)
    {
        // dp over the slice
        num_t h2 = num_t(0.5) * (plev[k+1] - plev[k]);

        // the current two x-y-planes of data
        unsigned long knxy = k*nxy;
        const num_t *q_k0 = q + knxy;
        const num_t *q_k1 = q_k0 + nxy;

        const char *q_valid_k0 = q_valid + knxy;
        const char *q_valid_k1 = q_valid_k0 + nxy;

        // accumulate this plane of data using trapazoid rule
        for (unsigned long i = 0; i < nxy; ++i)
        {
            iwv[i] += ((q_valid_k0[i] && q_valid_k1[i]) ?
               h2 * (q_k0[i] + q_k1[i]) : num_t(0));
        }
    }

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);
    for (unsigned long i = 0; i < nxy; ++i)
        iwv[i] *= m1g;
}
}

// --------------------------------------------------------------------------
teca_integrated_water_vapor::teca_integrated_water_vapor() :
    specific_humidity_variable("Q"), iwv_variable("IWV"),
    fill_value(1.0e20)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_integrated_water_vapor::~teca_integrated_water_vapor()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_integrated_water_vapor::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_integrated_water_vapor":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, specific_humidity_variable,
            "name of the variable containg the specific humidity")
        TECA_POPTS_GET(double, prefix, fill_value,
            "the value of the NetCDF _FillValue attribute")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_integrated_water_vapor::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, specific_humidity_variable)
    TECA_POPTS_SET(opts, double, prefix, fill_value)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_integrated_water_vapor::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_integrated_water_vapor::get_output_metadata" << std::endl;
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
            TECA_FATAL_ERROR("Failed to determine output data type "
                "because attributes are misisng")
            return teca_metadata();
        }

        teca_metadata hus_atts;
        if (attributes.get(this->specific_humidity_variable, hus_atts))
        {
            TECA_FATAL_ERROR("Failed to determine output data type "
                "because attributes for \"" << this->specific_humidity_variable
                << "\" are misisng")
            return teca_metadata();
        }

        int type_code = 0;
        if (hus_atts.get("type_code", type_code))
        {
            TECA_FATAL_ERROR("Failed to determine output data type "
                "because attributes for \"" << this->specific_humidity_variable
                << "\" is misisng a \"type_code\"")
            return teca_metadata();
        }

        teca_array_attributes iwv_atts(
            type_code, teca_array_attributes::point_centering,
            0, teca_array_attributes::xyt_active(), "kg m-2",
            "integrated water vapor",
            "vertically integrated " + this->specific_humidity_variable,
            1, this->fill_value);


        // install name and attributes of the output variables in the base classs
        this->append_derived_variable(this->iwv_variable);
        this->append_derived_variable_attribute(iwv_atts);
    }

    if (this->get_number_of_dependent_variables() == 0)
    {
        // install the names of the input variables in the base class
        this->append_dependent_variable(this->specific_humidity_variable);
    }

    // invoke the base class method, which does the work of transforming
    // the mesh and reporting the variables and their attributes.
    return teca_vertical_reduction::get_output_metadata(port, input_md);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_integrated_water_vapor::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    // invoke the base class method
    return teca_vertical_reduction::get_upstream_request(port, input_md, request);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_integrated_water_vapor::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_integrated_water_vapor::execute" << std::endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to compute IWV because a cartesian mesh is required.")
        return nullptr;
    }

    // get the input dimensions
    unsigned long extent[6] = {0};
    if (in_mesh->get_extent(extent))
    {
        TECA_FATAL_ERROR("Failed to compute IWV because mesh extent is missing.")
        return nullptr;
    }

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

    // get the pressure coordinates
    const_p_teca_variant_array p = in_mesh->get_z_coordinates();
    if (!p)
    {
        TECA_FATAL_ERROR("Failed to compute IWV because pressure coordinates are missing")
        return nullptr;
    }

    if (p->size() < 2)
    {
        TECA_FATAL_ERROR("Failed to compute IWV because z dimensions "
            << p->size() << " < 2 as required by the integration method")
        return nullptr;
    }

    // gather the input arrays
    const_p_teca_variant_array q =
        in_mesh->get_point_arrays()->get(this->specific_humidity_variable);

    if (!q)
    {
        TECA_FATAL_ERROR("Failed to compute IWV because specific humidity \""
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
        TECA_FATAL_ERROR("Failed to compute IWV because the output mesh was "
            "not constructed")
        return nullptr;
    }

    // allocate the output arrays
    unsigned long nxy = nx*ny;
    p_teca_variant_array iwv = q->new_instance(nxy);

    // store the result
    out_mesh->get_point_arrays()->set(this->iwv_variable, iwv);

    // calculate IWV
    NESTED_VARIANT_ARRAY_DISPATCH_FP(
        p.get(), _COORDS,

        auto [sp_p, p_p] = get_host_accessible<CTT_COORDS>(p);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            iwv.get(), _DATA,

            auto [sp_q, p_q] = get_host_accessible<CTT_DATA>(q);
            auto [p_iwv] = data<TT_DATA>(iwv);

            if (q_valid)
            {
                auto [spqv, p_q_valid] = get_host_accessible<CTT_MASK>(q_valid);
                sync_host_access_any(p, q, q_valid);
                ::cartesian_iwv(nx, ny, nz, p_p, p_q, p_q_valid, p_iwv);
            }
            else
            {
                sync_host_access_any(p, q);
                ::cartesian_iwv(nx, ny, nz, p_p, p_q, p_iwv);
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
