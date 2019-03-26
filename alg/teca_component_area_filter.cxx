#include "teca_component_area_filter.h"

#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"

#include <iostream>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

namespace {

template <typename label_t, typename area_t, typename container_t>
void get_filtered_labels(
    const label_t *unique_labels, const area_t *areas,
    container_t &filter_map, label_t replace_value, size_t n,
    double low_threshold_value, double high_threshold_value)
{
    for (size_t i = 0; i < n; ++i)
    {
        if ((areas[i] < low_threshold_value) ||
            (areas[i] > high_threshold_value))
            filter_map[unique_labels[i]] = replace_value;
        else
            filter_map[unique_labels[i]] = unique_labels[i];
    }
}

template <typename label_t, typename container_t>
void apply_filter(
    label_t *labels, const label_t *labels_in,
    container_t &filter_map, size_t n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        labels[i] = filter_map[labels_in[i]];
    }
}
}



// --------------------------------------------------------------------------
teca_component_area_filter::teca_component_area_filter() :
    component_variable(""), component_ids_key("label_id"),
    component_area_key("area"), mask_value(0),
    low_threshold_value(std::numeric_limits<double>::lowest()),
    high_threshold_value(std::numeric_limits<double>::max()),
    variable_post_fix("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_component_area_filter::~teca_component_area_filter()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_component_area_filter::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_component_area_filter":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_variable,
            "name of the varibale containing connected component labeling")
        TECA_POPTS_GET(std::string, prefix, component_ids_key,
            "name of the key that contains the list of component ids "
            "\"component_ids\")")
        TECA_POPTS_GET(std::string, prefix, component_area_key,
            "name of the key that contains the list of component areas "
            "(\"component_area\")")
        TECA_POPTS_GET(int, prefix, mask_value,
            "components with area outside of the range will be replaced "
            "by this label value (0)")
        TECA_POPTS_GET(double, prefix, low_threshold_value,
            "set the lower end of the range of areas to pass through. "
            "components smaller than this are masked out. (-inf)")
        TECA_POPTS_GET(double, prefix, high_threshold_value,
            "set the higher end of the range of areas to pass through. "
            "components larger than this are masked out. (+inf)")
        TECA_POPTS_GET(std::string, prefix, variable_post_fix,
            "set a string that will be appended to variable names and "
            "metadata keys in the filter's output (\"\")")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_component_area_filter::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, component_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_ids_key)
    TECA_POPTS_SET(opts, std::string, prefix, component_area_key)
    TECA_POPTS_SET(opts, int, prefix, mask_value)
    TECA_POPTS_SET(opts, double, prefix, low_threshold_value)
    TECA_POPTS_SET(opts, double, prefix, high_threshold_value)
    TECA_POPTS_SET(opts, std::string, prefix, variable_post_fix)
}
#endif

// --------------------------------------------------------------------------
std::string teca_component_area_filter::get_labels_variable(
    const teca_metadata &request)
{
    std::string labels_var = this->component_variable;
    if (labels_var.empty())
    {
        if (request.has("teca_component_area_filter::component_variable"))
            request.get("teca_component_area_filter::component_variable", labels_var);
        else if (request.has("teca_2d_component_area::label_variable"))
            request.get("teca_2d_component_area::label_variable", labels_var);
        else
            labels_var = "labels";
    }
    return labels_var;
}

// --------------------------------------------------------------------------
teca_metadata teca_component_area_filter::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_component_area_filter::get_output_metadata" << endl;
#endif
    (void) port;

    teca_metadata md = input_md[0];

    // TODO -- add the name of the output array, this is only
    // needed if the post fix is set

    return md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_component_area_filter::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_component_area_filter::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string labels_var = this->get_labels_variable(request);
    if (labels_var.empty())
    {
        TECA_ERROR("labels variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(labels_var);

    // TODO -- remove the arrays we produce. this is only needed
    // if the post-fix is set.

    req.insert("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_component_area_filter::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_component_area_filter::execute" << endl;
#endif
    (void)port;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);
    if (!in_mesh)
    {
        TECA_ERROR("empty input, or not a cartesian_mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array
    std::string labels_var = this->get_labels_variable(request);
    if (labels_var.empty())
    {
        TECA_ERROR("labels variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array labels_array
        = out_mesh->get_point_arrays()->get(labels_var);

    if (!labels_array)
    {
        TECA_ERROR("labels variable \"" << labels_var
            << "\" is not in the input")
        return nullptr;
    }

    // get threshold values
    double low_val = this->low_threshold_value;
    if (low_val == std::numeric_limits<double>::lowest()
        && request.has("teca_component_area_filter::low_threshold_value"))
        request.get("teca_component_area_filter::low_threshold_value", low_val);

    double high_val = this->high_threshold_value;
    if (high_val == std::numeric_limits<double>::max()
        && request.has("teca_component_area_filter::high_threshold_value"))
        request.get("teca_component_area_filter::high_threshold_value", high_val);


    // get the input and output metadata
    teca_metadata &in_metadata =
        const_cast<teca_metadata&>(in_mesh->get_metadata());
    teca_metadata &out_metadata = out_mesh->get_metadata();

    const_p_teca_variant_array label_id_array = in_metadata.get(this->component_ids_key);
    const_p_teca_variant_array area_array = in_metadata.get(this->component_area_key);

    size_t n_elem = labels_array->size();
    p_teca_variant_array filtered_labels_array = labels_array->new_instance(n_elem);

    // calculate area of components
    NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl,
        filtered_labels_array.get(),
        _LABEL,

        const NT_LABEL *p_labels_in = static_cast<const TT_LABEL*>(labels_array.get())->get();
        const NT_LABEL *p_unique_labels = static_cast<const TT_LABEL*>(label_id_array.get())->get();

        NT_LABEL *p_labels = static_cast<TT_LABEL*>(filtered_labels_array.get())->get();

        NT_LABEL replace_val = this->mask_value;

        NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            area_array.get(),
            _AREA,

            const NT_AREA *p_areas = static_cast<TT_AREA*>(area_array.get())->get();

            decltype(std::map<NT_LABEL, NT_LABEL>()) filter_map;

            size_t n_label_id = label_id_array->size();

            ::get_filtered_labels(
                    p_unique_labels, p_areas, filter_map, replace_val,
                    n_label_id, low_val, high_val);

            ::apply_filter(p_labels, p_labels_in, filter_map, n_elem);


            std::vector<NT_LABEL> label_id;
            std::vector<NT_AREA> area;
            for (size_t i = 0; i < n_label_id; ++i)
            {
                if (filter_map[p_unique_labels[i]] == p_unique_labels[i]){
                    label_id.push_back(p_unique_labels[i]);
                    area.push_back(p_areas[i]);
                }
            }

            std::string labels_var_post_fix = labels_var + this->variable_post_fix;
            out_mesh->get_point_arrays()->set(labels_var_post_fix, filtered_labels_array);

            out_metadata.insert("label_id" + this->variable_post_fix, label_id);
            out_metadata.insert("area" + this->variable_post_fix, area);

            )
        )

    return out_mesh;
}
