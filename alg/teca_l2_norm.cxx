#include "teca_l2_norm.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

namespace {

template <typename num_t>
void sum_square(num_t *ss, const num_t *c, unsigned long n)
{
    // TODO -- verify that this is being vectorized
    for (unsigned long i = 0; i < n; ++i)
    {
        num_t ci = c[i];
        ss[i] += ci*ci;
    }
}

template <typename num_t>
void square_root(num_t *rt, const num_t *c, unsigned long n)
{
    // TODO -- verify that this is vectorized, it might not be
    // because of aliasing
    for (unsigned long i = 0; i < n; ++i)
    {
        rt[i] = sqrt(c[i]);
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
    options_description opts("Options for " + prefix + "(teca_l2_norm)");

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_0_variable, "array containg the first component")
        TECA_POPTS_GET(std::string, prefix, component_1_variable, "array containg the second component")
        TECA_POPTS_GET(std::string, prefix, component_2_variable, "array containg the third component")
        TECA_POPTS_GET(std::string, prefix, l2_norm_variable, "array to store the computed norm in")
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
    out_md.append("variables", this->l2_norm_variable);

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

    // do some error checking.
    if (this->component_0_variable.empty())
    {
        TECA_ERROR("at least one component array is needed")
        return up_reqs;
    }

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::vector<std::string> arrays;
    req.get("arrays", arrays);

    arrays.push_back(this->component_0_variable);

    if (!this->component_1_variable.empty())
        arrays.push_back(this->component_1_variable);

    if (!this->component_2_variable.empty())
        arrays.push_back(this->component_2_variable);

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
    (void)request;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("Failed to compute l2 norm. dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // get the input component arrays
    const_p_teca_variant_array c0
        = in_mesh->get_point_arrays()->get(this->component_0_variable);

    const_p_teca_variant_array c1 = nullptr;
    if (!this->component_1_variable.empty())
    {
        c1 = in_mesh->get_point_arrays()->get(this->component_1_variable);
    }

    const_p_teca_variant_array c2 = nullptr;
    if (!this->component_2_variable.empty())
    {
        c2 = in_mesh->get_point_arrays()->get(this->component_2_variable);
    }

    // allocate the output array
    unsigned long n = c0->size();
    p_teca_variant_array l2_norm = c0->new_instance();
    l2_norm->resize(n);

    // compute l2 norm
    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        l2_norm.get(),

        const NT *pc0 = dynamic_cast<const TT*>(c0.get())->get();
        NT *pl2 = dynamic_cast<TT*>(l2_norm.get())->get();

        sum_square(pl2, pc0, n);

        if (c1)
        {
            const NT *pc1 = dynamic_cast<const TT*>(c1.get())->get();
            sum_square(pl2, pc1, n);
        }

        if (c2)
        {
            const NT *pc2 = dynamic_cast<const TT*>(c2.get())->get();
            sum_square(pl2, pc2, n);
        }

        square_root(pl2, pl2, n);
        )

    // create the output mesh, pass everything through, and
    // add the l2 norm array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
    out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));
    out_mesh->get_point_arrays()->append(this->l2_norm_variable, l2_norm);

    return out_mesh;
}
