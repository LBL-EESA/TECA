#include "teca_surface_pressure_estimate.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
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

//#define TECA_DEBUG

namespace internal
{
template <typename num_t, typename coord_t>
void compute_surface_pressure(size_t n_elem, const num_t *tas,
    const num_t *psl, coord_t *z, num_t *ps)
{
    num_t cp = 1.0; // kJ/kg.K
    num_t cv = 0.718; // kJ/kg.K
    num_t g = 9.81;
    num_t GAM = -g / cp;
    num_t gam = cp / cv;
    num_t gam_o_gam_m_1 = gam / (gam - num_t(1));

    for (size_t i = 0; i < n_elem; ++i)
    {
        num_t zi = z[i];
        ps[i] = psl[i] * pow( num_t(1) +
            GAM * zi / ( tas[i] - GAM *zi ), gam_o_gam_m_1 );
    }
}
}


// --------------------------------------------------------------------------
teca_surface_pressure_estimate::teca_surface_pressure_estimate() :
    surface_temperature_variable("tas"), sea_level_pressure_variable("psl"),
    surface_elevation_variable("z"), surface_pressure_variable("ps")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_surface_pressure_estimate::~teca_surface_pressure_estimate()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_surface_pressure_estimate::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_surface_pressure_estimate":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, surface_temperature_variable,
            "array containg the surface temperature field")
        TECA_POPTS_GET(std::string, prefix, sea_level_pressure_variable,
            "array containg the mean sea level pressure field")
        TECA_POPTS_GET(std::string, prefix, surface_elevation_variable,
            "array containg the surface elevation field")
        TECA_POPTS_GET(std::string, prefix, surface_pressure_variable,
            "array to store the computed surface pressure in")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_surface_pressure_estimate::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, surface_temperature_variable)
    TECA_POPTS_SET(opts, std::string, prefix, sea_level_pressure_variable)
    TECA_POPTS_SET(opts, std::string, prefix, surface_elevation_variable)
    TECA_POPTS_SET(opts, std::string, prefix, surface_pressure_variable)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_surface_pressure_estimate::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_surface_pressure_estimate::get_output_metadata" << std::endl;
#endif
    (void)port;

    // pass existing metadata
    teca_metadata out_md(input_md[0]);

    // add in the array we will generate
    out_md.append("variables", this->surface_pressure_variable);

    // insert attributes to enable this to be written by the CF writer
    teca_metadata attributes;
    out_md.get("attributes", attributes);

    teca_metadata psl_atts;
    if (attributes.get(this->sea_level_pressure_variable, psl_atts))
    {
        TECA_WARNING("Failed to get seal level pressure \""
            << this->sea_level_pressure_variable
            << "\" attrbibutes. Writing the result will not be possible")
    }
    else
    {
        // copy the attributes from the input. this will capture the
        // data type, size, units, etc.
        teca_array_attributes ps_atts(psl_atts);

        // update name, long_name, and description.
        ps_atts.long_name = this->surface_pressure_variable;

        ps_atts.description = std::string("Surface pressure estimated from " +
            this->sea_level_pressure_variable + ", " + this->surface_temperature_variable +
            ", and " + this->surface_elevation_variable);

        attributes.set(this->surface_pressure_variable, (teca_metadata)ps_atts);
        out_md.set("attributes", attributes);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_surface_pressure_estimate::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->surface_temperature_variable);
    arrays.insert(this->sea_level_pressure_variable);
    arrays.insert(this->surface_elevation_variable);

    // intercept request for our output
    arrays.erase(this->surface_pressure_variable);

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_surface_pressure_estimate::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_surface_pressure_estimate::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("Failed to compute surface pressure. The dataset is"
            " not a teca_mesh")
        return nullptr;
    }

    // get the surface temperature
    const_p_teca_variant_array tas
        = in_mesh->get_point_arrays()->get(this->surface_temperature_variable);
    if (!tas)
    {
        TECA_ERROR("surface temperature array \""
            << this->surface_temperature_variable << "\" was not found.")
        return nullptr;
    }

    // get the sea level pressure
    const_p_teca_variant_array psl = nullptr;
    if (!this->sea_level_pressure_variable.empty() &&
        !(psl = in_mesh->get_point_arrays()->get(this->sea_level_pressure_variable)))
    {
        TECA_ERROR("sea level pressure array \""
            << this->sea_level_pressure_variable << "\" was not found.")
        return nullptr;
    }

    // get the surface elevation
    const_p_teca_variant_array z = nullptr;
    if (!this->surface_elevation_variable.empty() &&
        !(z = in_mesh->get_point_arrays()->get(this->surface_elevation_variable)))
    {
        TECA_ERROR("surface elevation array \""
            << this->surface_elevation_variable << "\" was not found.")
        return nullptr;
    }

    // allocate the output array
    unsigned long n = psl->size();
    p_teca_variant_array ps = psl->new_instance();
    ps->resize(n);

    // compute the surface pressure
    NESTED_TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        ps.get(),
        _DATA,

        const NT_DATA *p_tas = static_cast<const TT_DATA*>(tas.get())->get();
        const NT_DATA *p_psl = static_cast<const TT_DATA*>(psl.get())->get();
        NT_DATA *p_ps = static_cast<TT_DATA*>(ps.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(
            const teca_variant_array_impl,
            z.get(),
            _COORD,

            const NT_COORD *p_z = static_cast<const TT_COORD*>(z.get())->get();

            internal::compute_surface_pressure(n, p_tas, p_psl, p_z, p_ps);
            )
        )

    // create the output mesh, pass everything through, and
    // add the l2 norm array
    p_teca_mesh out_mesh = std::static_pointer_cast<teca_mesh>
        (std::const_pointer_cast<teca_mesh>(in_mesh)->new_shallow_copy());

    out_mesh->get_point_arrays()->set(
        this->surface_pressure_variable, ps);

    return out_mesh;
}
