#include "teca_descriptive_statistics.h"

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

using namespace teca_variant_array_util;

using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

namespace internal
{
template <typename num_t>
void quartiles(const num_t *ptr, size_t n, num_t &lq, num_t &med, num_t &uq)
{
    size_t nb = n*sizeof(num_t);
    num_t *tmp = static_cast<num_t*>(malloc(nb));
    memcpy(tmp, ptr, nb);
    size_t n25 = n/4;
    size_t n50 = n/2;
    size_t n75 = (n*3)/4;
    std::partial_sort(tmp, tmp+n75+1, tmp+n);
    lq = tmp[n25];
    med = tmp[n50];
    uq = tmp[n75];
    free(tmp);
}

template <typename num_t>
void quartiles2(const num_t *ptr, size_t n, num_t &lq, num_t &med, num_t &uq)
{
    size_t nb = n*sizeof(num_t);
    num_t *tmp = static_cast<num_t*>(malloc(nb));
    memcpy(tmp, ptr, nb);

    size_t n25 = n/4;
    std::nth_element(tmp, tmp+n25, tmp+n);
    lq = tmp[n25];

    size_t n50 = n/2;
    std::nth_element(tmp, tmp+n50, tmp+n);
    med = tmp[n50];

    size_t n75 = (n*3)/4;
    std::nth_element(tmp, tmp+n75, tmp+n);
    uq = tmp[n75];

    free(tmp);
}

template <typename num_t>
num_t min(const num_t *ptr, size_t n)
{
    num_t min = std::numeric_limits<num_t>::max();
    for (size_t i = 0; i < n; ++i)
        min = min > ptr[i] ? ptr[i] : min;
    return min;
}

template <typename num_t>
num_t max(const num_t *ptr, size_t n)
{
    num_t max = std::numeric_limits<num_t>::lowest();
    for (size_t i = 0; i < n; ++i)
        max = max < ptr[i] ? ptr[i] : max;
    return max;
}

template <typename num_t>
num_t sum(const num_t *ptr, size_t n)
{
    num_t s = num_t();
    for (size_t i = 0; i < n; ++i)
        s += ptr[i];
    return s;
}

template <typename num_t>
num_t var(const num_t *ptr, size_t n, num_t av)
{
    if (!n) return num_t();
    num_t v = num_t();
    for (size_t i = 0; i < n; ++i)
    {
        num_t d = ptr[i] - av;
        v += d*d;
    }
    v /= num_t(n);
    return v;
}
};


// --------------------------------------------------------------------------
teca_descriptive_statistics::teca_descriptive_statistics()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_descriptive_statistics::~teca_descriptive_statistics()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_descriptive_statistics::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_descriptive_statistics":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, dependent_variables,
            "list of arrays to compute statistics for")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_descriptive_statistics::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, dependent_variables)
}
#endif

// --------------------------------------------------------------------------
void teca_descriptive_statistics::get_dependent_variables(
    const teca_metadata &request, std::vector<std::string> &dep_vars)
{
    dep_vars = this->dependent_variables;

    if (dep_vars.empty())
    {
        std::string key = "teca_descriptive_statistics::dependent_variables";
        if (request.has(key))
            request.get(key, dep_vars);
    }
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_descriptive_statistics::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_descriptive_statistics::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    // intercept request for our output
    //arrays.erase(this->get_derived_variable(request));

    // get the names of the arrays we need to request
    std::vector<std::string> dep_vars;
    this->get_dependent_variables(req, dep_vars);

    size_t n = dependent_variables.size();
    for (size_t i = 0; i < n; ++i)
        arrays.insert(dep_vars[i]);

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_descriptive_statistics::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_descriptive_statistics::execute" << endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("dataset is not a teca_cartesian_mesh")
        return nullptr;
    }
    // dependent variables
    std::vector<std::string> dep_var_names;
    this->get_dependent_variables(request, dep_var_names);
    size_t n_dep_vars = dep_var_names.size();

    // set up the output
    p_teca_table table = teca_table::New();
    table->declare_columns("step", long(), "time", double());

    // pass calendaring metadata
    std::string calendar;
    in_mesh->get_calendar(calendar);
    table->set_calendar(calendar);

    std::string time_units;
    in_mesh->get_time_units(time_units);
    table->set_time_units(time_units);

    unsigned long step;
    in_mesh->get_time_step(step);
    table << step;

    double time;
    in_mesh->get_time(time);
    table << time;

    // pass attributes and add the new variables we consrtuct. this is for
    // NetCDF output.
    teca_metadata atrs;
    if (in_mesh->get_metadata().get("attributes", atrs) == 0)
    {
        for (size_t i = 0; i < n_dep_vars; ++i)
        {
            const std::string &dep_var_name = dep_var_names[i];

            teca_metadata var_atts;
            if (atrs.get(dep_var_names[i], var_atts) == 0)
            {
                teca_array_attributes out_atts(var_atts);

                std::string out_var_name = "min_" + dep_var_name;
                out_atts.long_name = out_var_name;
                out_atts.description = "minimum of " + dep_var_name;
                atrs.set(out_var_name, (teca_metadata)out_atts);

                out_var_name = "max_" + dep_var_name;
                out_atts.long_name = out_var_name;
                out_atts.description = "maximum of " + dep_var_name;
                atrs.set(out_var_name, (teca_metadata)out_atts);

                out_var_name = "avg_" + dep_var_name;
                out_atts.long_name = out_var_name;
                out_atts.description = "average of " + dep_var_name;
                atrs.set(out_var_name, (teca_metadata)out_atts);

                out_var_name = "var_" + dep_var_name;
                out_atts.long_name = out_var_name;
                out_atts.description = "variance of " + dep_var_name;
                atrs.set(out_var_name, (teca_metadata)out_atts);

                out_var_name = "med_" + dep_var_name;
                out_atts.long_name = out_var_name;
                out_atts.description = "median of " + dep_var_name;
                atrs.set(out_var_name, (teca_metadata)out_atts);

                out_var_name = "low_q_" + dep_var_name;
                out_atts.long_name = out_var_name;
                out_atts.description = "lower quartile of " + dep_var_name;
                atrs.set(out_var_name, (teca_metadata)out_atts);

                out_var_name = "up_q_" + dep_var_name;
                out_atts.long_name = out_var_name;
                out_atts.description = "upper quartile of " + dep_var_name;
                atrs.set(out_var_name, (teca_metadata)out_atts);
            }
        }

        teca_array_attributes step_atts;
        step_atts.long_name = "time step";
        step_atts.description = "input dataset time step";
        step_atts.units = "unitless";
        atrs.set("step", (teca_metadata)step_atts);

        table->get_metadata().set("attributes", atrs);
    }

    // for each variable
    for (size_t i = 0; i < n_dep_vars; ++i)
    {
        const std::string &dep_var_name = dep_var_names[i];

        // get the array
        const_p_teca_variant_array dep_var
            = in_mesh->get_point_arrays()->get(dep_var_name);
        if (!dep_var)
        {
            TECA_FATAL_ERROR("dependent variable " << i << " \""
                << dep_var_name << "\" not present.")
            return nullptr;
        }

        VARIANT_ARRAY_DISPATCH(dep_var.get(),

            size_t n = dep_var->size();

            auto [spv, pv] = get_host_accessible<CTT>(dep_var);

            sync_host_access_any(dep_var);

            // compute stats
            NT mn = internal::min(pv, n);
            NT mx = internal::max(pv, n);
            NT av = internal::sum(pv, n)/NT(n);
            NT vr = internal::var(pv, n, av);
            NT lq; NT med; NT uq;
            internal::quartiles2(pv, n, lq, med, uq);

            // add to output table
            table->declare_columns(
                "min_" + dep_var_name, NT(), "max_" + dep_var_name, NT(),
                "avg_" + dep_var_name, NT(), "var_" + dep_var_name, NT(),
                "low_q_" + dep_var_name, NT(), "med_" + dep_var_name, NT(),
                "up_q_" + dep_var_name, NT());

            table << mn << mx << av << vr << lq << med << uq;
            )
    }

    return table;
}
