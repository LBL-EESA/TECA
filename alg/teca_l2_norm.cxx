#include "teca_l2_norm.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

namespace internal
{
template <typename num_t>
void square(num_t * __restrict__ s,
    const num_t * __restrict__ c, unsigned long n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        num_t ci = c[i];
        s[i] = ci*ci;
    }
}

template <typename num_t>
void sum_square(num_t * __restrict__ ss,
    const num_t * __restrict__ c, unsigned long n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        num_t ci = c[i];
        ss[i] += ci*ci;
    }
}

template <typename num_t>
void square_root(num_t * __restrict__ rt,
    const num_t * __restrict__ c, unsigned long n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        rt[i] = std::sqrt(c[i]);
    }
}
};


// --------------------------------------------------------------------------
teca_l2_norm::teca_l2_norm() :
    component_0_variable(), component_1_variable(),
    component_2_variable(), l2_norm_variable("l2_norm")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_l2_norm::~teca_l2_norm()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_l2_norm::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_l2_norm":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_0_variable,
            "array containg the first component")
        TECA_POPTS_GET(std::string, prefix, component_1_variable,
            "array containg the second component")
        TECA_POPTS_GET(std::string, prefix, component_2_variable,
            "array containg the third component")
        TECA_POPTS_GET(std::string, prefix, l2_norm_variable,
            "array to store the computed norm in")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_l2_norm::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, component_0_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_1_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_2_variable)
    TECA_POPTS_SET(opts, std::string, prefix, l2_norm_variable)
}
#endif

// --------------------------------------------------------------------------
std::string teca_l2_norm::get_component_0_variable(const teca_metadata &request)
{
    std::string comp_0_var = this->component_0_variable;

    if (comp_0_var.empty() &&
        request.has("teca_l2_norm::component_0_variable"))
            request.get("teca_l2_norm::component_0_variable", comp_0_var);

    return comp_0_var;
}

// --------------------------------------------------------------------------
std::string teca_l2_norm::get_component_1_variable(
    const teca_metadata &request)
{
    std::string comp_1_var = this->component_1_variable;

    if (comp_1_var.empty() &&
        request.has("teca_l2_norm::component_1_variable"))
            request.get("teca_l2_norm::component_1_variable", comp_1_var);

    return comp_1_var;
}

// --------------------------------------------------------------------------
std::string teca_l2_norm::get_component_2_variable(const teca_metadata &request)
{
    std::string comp_2_var = this->component_2_variable;

    if (comp_2_var.empty() &&
        request.has("teca_l2_norm::component_2_variable"))
            request.get("teca_l2_norm::component_2_variable", comp_2_var);

    return comp_2_var;
}

// --------------------------------------------------------------------------
std::string teca_l2_norm::get_l2_norm_variable(const teca_metadata &request)
{
    std::string norm_var = this->l2_norm_variable;

    if (norm_var.empty())
    {
        if (request.has("teca_l2_norm::l2_norm_variable"))
            request.get("teca_l2_norm::l2_norm_variable", norm_var);
        else
            norm_var = "l2_norm";
    }

    return norm_var;
}

// --------------------------------------------------------------------------
teca_metadata teca_l2_norm::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_l2_norm::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    std::string norm_var = this->l2_norm_variable;
    if (norm_var.empty())
        norm_var = "l2_norm";

    out_md.append("variables", norm_var);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_l2_norm::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // get the names of the arrays we need to request
    std::string comp_0_var = this->get_component_0_variable(request);
    if (comp_0_var.empty())
    {
        TECA_ERROR("component 0 array was not specified")
        return up_reqs;
    }

    // optional arrays to request
    std::string comp_1_var = this->get_component_1_variable(request);
    std::string comp_2_var = this->get_component_2_variable(request);

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(comp_0_var);

    if (!comp_1_var.empty())
        arrays.insert(comp_1_var);

    if (!comp_2_var.empty())
        arrays.insert(comp_2_var);

    // intercept request for our output
    arrays.erase(this->get_l2_norm_variable(request));

    req.insert("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_l2_norm::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_l2_norm::execute" << endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("Failed to compute l2 norm. dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // get the name of the first component
    std::string comp_0_var = this->get_component_0_variable(request);
    if (comp_0_var.empty())
    {
        TECA_ERROR("component 0 array was not specified")
        return nullptr;
    }

    // optional component array names
    std::string comp_1_var = this->get_component_1_variable(request);
    std::string comp_2_var = this->get_component_2_variable(request);

    // get the first component array
    const_p_teca_variant_array c0
        = in_mesh->get_point_arrays()->get(comp_0_var);
    if (!c0)
    {
        TECA_ERROR("component 0 array \"" << comp_0_var
            << "\" not present.")
        return nullptr;
    }

    // get the optional component arrays
    const_p_teca_variant_array c1 = nullptr;
    if (!comp_1_var.empty() &&
        !(c1 = in_mesh->get_point_arrays()->get(comp_1_var)))
    {
        TECA_ERROR("component 1 array \"" << comp_1_var
            << "\" requested but not present.")
        return nullptr;
    }

    const_p_teca_variant_array c2 = nullptr;
    if (!comp_2_var.empty() &&
        !(c2 = in_mesh->get_point_arrays()->get(comp_2_var)))
    {
        TECA_ERROR("component 1 array \"" << comp_2_var
            << "\" requested but not present.")
        return nullptr;
    }

    // allocate the output array
    unsigned long n = c0->size();
    p_teca_variant_array l2_norm = c0->new_instance();
    l2_norm->resize(n);

    // compute l2 norm
    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        l2_norm.get(),

        NT *tmp = static_cast<NT*>(malloc(sizeof(NT)*n));
        internal::square(tmp, static_cast<const TT*>(c0.get())->get(), n);

        if (c1)
            internal::sum_square(tmp, dynamic_cast<const TT*>(c1.get())->get(), n);

        if (c2)
            internal::sum_square(tmp, dynamic_cast<const TT*>(c2.get())->get(), n);

        internal::square_root(static_cast<TT*>(l2_norm.get())->get(), tmp, n);

        free(tmp);
        )

    // create the output mesh, pass everything through, and
    // add the l2 norm array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    out_mesh->get_point_arrays()->append(
        this->get_l2_norm_variable(request), l2_norm);

    return out_mesh;
}
