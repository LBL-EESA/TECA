#include "teca_component_area_filter.h"

#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_metadata_util.h"

#include <iostream>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

namespace {

// this constructs a map from input label id to the output label id, the list
// of ids that survive and their respective areas. if the area of the label is
// outside of the range it will be replaced in the output.
template <typename label_t, typename area_t, typename container_t>
void build_label_map(const label_t *comp_ids, const area_t *areas,
    size_t n, double low_area_threshold, double high_area_threshold,
    label_t mask_value, container_t &label_map,
    std::vector<label_t> &ids_out, std::vector<area_t> &areas_out)
{
    for (size_t i = 0; i < n; ++i)
    {
        if ((areas[i] < low_area_threshold) || (areas[i] > high_area_threshold))
        {
            // outside the range, mask this label
            label_map[comp_ids[i]] = mask_value;
        }
        else
        {
            // inside the range, pass it through
            label_map[comp_ids[i]] = comp_ids[i];
            ids_out.push_back(comp_ids[i]);
            areas_out.push_back(areas[i]);
        }
    }
}

// visit every point in the data, apply the map. The map is such that labels
// ouside of the specified range are replaced
template <typename label_t, typename container_t>
void apply_label_map(label_t *labels, const label_t *labels_in,
    container_t &label_map, size_t n)
{
    for (unsigned long i = 0; i < n; ++i)
        labels[i] = label_map[labels_in[i]];
}

}



// --------------------------------------------------------------------------
teca_component_area_filter::teca_component_area_filter() :
    component_variable(""), number_of_components_key("number_of_components"),
    component_ids_key("component_ids"), component_area_key("component_area"),
    mask_value(-1), low_area_threshold(std::numeric_limits<double>::lowest()),
    high_area_threshold(std::numeric_limits<double>::max()),
    variable_post_fix(""), contiguous_component_ids(0)
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
        TECA_POPTS_GET(std::string, prefix, number_of_components_key,
            "name of the key that contains the number of components"
            "\"number_of_components\")")
        TECA_POPTS_GET(std::string, prefix, component_ids_key,
            "name of the key that contains the list of component ids "
            "\"component_ids\")")
        TECA_POPTS_GET(std::string, prefix, component_area_key,
            "name of the key that contains the list of component areas "
            "(\"component_area\")")
        TECA_POPTS_GET(int, prefix, mask_value,
            "components with area outside of the range will be replaced "
            "by this label value (-1)")
        TECA_POPTS_GET(double, prefix, low_area_threshold,
            "set the lower end of the range of areas to pass through. "
            "components smaller than this are masked out. (-inf)")
        TECA_POPTS_GET(double, prefix, high_area_threshold,
            "set the higher end of the range of areas to pass through. "
            "components larger than this are masked out. (+inf)")
        TECA_POPTS_GET(std::string, prefix, variable_post_fix,
            "set a string that will be appended to variable names and "
            "metadata keys in the filter's output (\"\")")
        TECA_POPTS_GET(int, prefix, contiguous_component_ids,
            "when the region label ids start at 0 and are consecutive "
            "this flag enables use of an optimization (0)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_component_area_filter::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, component_variable)
    TECA_POPTS_SET(opts, std::string, prefix, number_of_components_key)
    TECA_POPTS_SET(opts, std::string, prefix, component_ids_key)
    TECA_POPTS_SET(opts, std::string, prefix, component_area_key)
    TECA_POPTS_SET(opts, int, prefix, mask_value)
    TECA_POPTS_SET(opts, double, prefix, low_area_threshold)
    TECA_POPTS_SET(opts, double, prefix, high_area_threshold)
    TECA_POPTS_SET(opts, std::string, prefix, variable_post_fix)
}
#endif

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

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    const std::string &var_post_fix = this->variable_post_fix;
    if (!var_post_fix.empty())
    {
        std::string component_var = this->component_variable;
        out_md.append("variables", component_var + var_post_fix);
    }

    return out_md;
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
    if (this->component_variable.empty())
    {
        TECA_ERROR("The component variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(this->component_variable);

    // remove the arrays we produce if the post-fix is set,
    // and replace it with the actual requested array.
    const std::string &var_post_fix = this->variable_post_fix;
    if (!var_post_fix.empty())
    {
        teca_metadata_util::remove_post_fix(arrays, var_post_fix);
    }

    req.set("arrays", arrays);

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
    if (this->component_variable.empty())
    {
        TECA_ERROR("The component variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array labels_in
        = out_mesh->get_point_arrays()->get(this->component_variable);

    if (!labels_in)
    {
        TECA_ERROR("labels variable \"" << this->component_variable
            << "\" is not in the input")
        return nullptr;
    }

    // get the list of component ids, and their corresponding areas
    teca_metadata &in_metadata =
        const_cast<teca_metadata&>(in_mesh->get_metadata());

    const_p_teca_variant_array ids_in
        = in_metadata.get(this->component_ids_key);

    if (!ids_in)
    {
        TECA_ERROR("Metadata missing component ids")
        return nullptr;
    }

    size_t n_ids_in = ids_in->size();

    const_p_teca_variant_array areas_in
        = in_metadata.get(this->component_area_key);

    if (!areas_in)
    {
        TECA_ERROR("Metadata missing component areas")
        return nullptr;
    }

    // get threshold values
    double low_val = this->low_area_threshold;
    if (low_val == std::numeric_limits<double>::lowest()
        && request.has("low_area_threshold"))
        request.get("low_area_threshold", low_val);

    double high_val = this->high_area_threshold;
    if (high_val == std::numeric_limits<double>::max()
        && request.has("high_area_threshold"))
        request.get("high_area_threshold", high_val);

    // allocate the array to store the output with labels outside the requested
    // range removed.
    size_t n_elem = labels_in->size();
    p_teca_variant_array labels_out = labels_in->new_instance(n_elem);

    // pass to the output
    std::string labels_var_post_fix = this->component_variable + this->variable_post_fix;
    out_mesh->get_point_arrays()->set(labels_var_post_fix, labels_out);

    // get the output metadata to add results to after the filter is applied
    teca_metadata &out_metadata = out_mesh->get_metadata();

    long mask_value = this->mask_value;
    if (this->mask_value == -1)
    {
        if (in_metadata.get("background_id", mask_value))
        {
            TECA_ERROR("Metadata is missing the key \"background_id\". "
                "One should specify it via the \"mask_value\" algorithm "
                "property")
            return nullptr;
        }
    }

    // apply the filter
    NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl,
        labels_out.get(),
        _LABEL,

        // pointer to input/output labels
        const NT_LABEL *p_labels_in =
            static_cast<const TT_LABEL*>(labels_in.get())->get();

        NT_LABEL *p_labels_out =
            static_cast<TT_LABEL*>(labels_out.get())->get();

        // pointer to input ids and a container to hold ids which remain
        // after the filtering operation
        const NT_LABEL *p_ids_in =
            static_cast<const TT_LABEL*>(ids_in.get())->get();

        std::vector<NT_LABEL> ids_out;

        NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            areas_in.get(),
            _AREA,

            // pointer to the areas in and a container to hold areas which
            // remain after the filtering operation
            const NT_AREA *p_areas = static_cast<TT_AREA*>(areas_in.get())->get();

            std::vector<NT_AREA> areas_out;

            // if we labels with small values we can speed the calculation by
            // using a contiguous buffer to hold the map. otherwise we need to
            // use an associative container
            if (this->contiguous_component_ids)
            {
                // find the max label id, used to size the map buffer
                NT_LABEL max_id = std::numeric_limits<NT_LABEL>::lowest();
                for (unsigned int i = 0; i < n_ids_in; ++i)
                    max_id = std::max(max_id, p_ids_in[i]);

                // allocate the map
                std::vector<NT_LABEL> label_map(max_id+1, NT_LABEL(mask_value));

                // construct the map from input label to output label.
                // removing a lable from the output ammounts to applying
                // the mask value to the labels
                ::build_label_map(p_ids_in, p_areas, n_ids_in,
                        low_val, high_val, NT_LABEL(mask_value),
                        label_map, ids_out, areas_out);

                // use the map to mask out removed labels
                ::apply_label_map(p_labels_out, p_labels_in, label_map, n_elem);
            }
            else
            {
                decltype(std::map<NT_LABEL, NT_LABEL>()) label_map;

                // construct the map from input label to output label.
                // removing a lable from the output ammounts to applying
                // the mask value to the labels
                ::build_label_map(p_ids_in, p_areas, n_ids_in,
                        low_val, high_val, NT_LABEL(mask_value),
                        label_map, ids_out, areas_out);

                // use the map to mask out removed labels
                ::apply_label_map(p_labels_out, p_labels_in, label_map, n_elem);
            }

            // pass the updated set of component ids and their coresponding areas
            // to the output
            out_metadata.set(this->number_of_components_key + this->variable_post_fix, ids_out.size());
            out_metadata.set(this->component_ids_key + this->variable_post_fix, ids_out);
            out_metadata.set(this->component_area_key + this->variable_post_fix, areas_out);
            out_metadata.set("background_id" + this->variable_post_fix, mask_value);

            // pass the threshold values used
            out_metadata.set("low_area_threshold_km", low_val);
            out_metadata.set("high_area_threshold_km", high_val);
            )
        )

    return out_mesh;
}
