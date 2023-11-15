#include "teca_apply_tempest_remap.h"

#include "teca_cartesian_mesh.h"
#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_valid_value_mask.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_apply_tempest_remap::teca_apply_tempest_remap() :
    weights_variable("S"), row_variable("row"), column_variable("col"),
    target_mask_variable(""), static_target_mesh(0)
{
    this->set_number_of_input_connections(3);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_apply_tempest_remap::~teca_apply_tempest_remap()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_apply_tempest_remap::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_apply_tempest_remap":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, weights_variable,
            "The name of the variable with remap weights")
        TECA_POPTS_GET(std::string, prefix, row_variable,
            "The name of the variable with the row indices")
        TECA_POPTS_GET(std::string, prefix, column_variable,
            "The name of the variable with the column indices")
        TECA_POPTS_GET(std::string, prefix, target_mask_variable,
            "The name of the variable with valid target mesh locations")
        TECA_POPTS_GET(int, prefix, static_target_mesh,
            "If set time is ignored in the target mesh input.")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_apply_tempest_remap::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, weights_variable)
    TECA_POPTS_SET(opts, std::string, prefix, row_variable)
    TECA_POPTS_SET(opts, std::string, prefix, column_variable)
    TECA_POPTS_SET(opts, std::string, prefix, target_mask_variable)
    TECA_POPTS_SET(opts, int, prefix, static_target_mesh)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_apply_tempest_remap::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_apply_tempest_remap::get_output_metadata" << std::endl;
#endif
    (void)port;

    const teca_metadata &src_md = input_md[0];
    const teca_metadata &tgt_md = input_md[1];

    // start with a copy of metadata from the source
    teca_metadata out_md(tgt_md);

    // pull the pipeline controls from the source
    std::string initializer_key;
    src_md.get("index_initializer_key", initializer_key);

    unsigned long n_indices = 0;
    src_md.get(initializer_key, n_indices);

    std::string request_key;
    src_md.get("index_request_key", request_key);

    out_md.set("index_initializer_key", initializer_key);
    out_md.set("index_request_key", request_key);
    out_md.set(initializer_key, n_indices);

    // pull the time axis from the source
    teca_metadata src_coords;
    src_md.get("coordinates", src_coords);

    std::string t_var;
    src_coords.get("t_variable", t_var);

    teca_metadata out_coords;
    out_md.get("coordinates", out_coords);

    out_coords.set("t", src_coords.get("t"));
    out_coords.set("t_variable", t_var);

    out_md.set("coordinates", out_coords);

    // get target metadata
    std::set<std::string> target_vars;
    tgt_md.get("variables", target_vars);

    teca_metadata target_atts;
    tgt_md.get("attributes", target_atts);

    // get source metadata
    std::vector<std::string> source_vars;
    src_md.get("variables", source_vars);

    teca_metadata source_atts;
    src_md.get("attributes", source_atts);

    // merge metadata from source and target variables should be unique lists.
    // attributes are indexed by variable names in the case of collisions, the
    // target variable is kept, the source variable is ignored
    size_t n_source_vars = source_vars.size();
    for (size_t i = 0; i < n_source_vars; ++i)
    {
        const std::string &src_array = source_vars[i];

        // check that there's not a variable of that same name in target
        // the request phase is going to route stuff to the target first
        // making these arrays in the source inaccessible.
        if (target_vars.find(src_array) != target_vars.end())
        {
            if (this->get_verbose())
            {
                TECA_WARNING("The src_array and target mesh both have an array"
                    " named \"" << src_array << "\". The data from the target"
                    " mesh will be used. To prevent this issue use"
                    " teca_rename_variables upstream.")
            }
        }
        else
        {
            // this array is not present in the target, ok to add it
            // report as an available variable on the target mesh
            target_vars.insert(src_array);

            // get the source array attributes
            teca_metadata src_array_atts;
            source_atts.get(src_array, src_array_atts);

            // if the src_array doesn't specify a _FillValue and the target mask is active
            // use the _FillValue specified from the target mask
            int type_code = 0;
            if (!this->target_mask_variable.empty() && !src_array_atts.has("_FillValue") &&
                !src_array_atts.get("type_code", type_code))
            {
                // make sure the _FillValue is specified with the matching precision
                // and don't add a _FillValue for integer arrays
                CODE_DISPATCH_FP(type_code,

                    NT tgt_mask_fill_value = NT();

                    teca_metadata tgt_mask_atts;
                    if (target_atts.get(this->target_mask_variable, tgt_mask_atts) ||
                        tgt_mask_atts.get("_FillValue", tgt_mask_fill_value))
                    {
                        // _FillValue must be present in this case, if not you can't
                        // use a target_mask_variable
                        TECA_FATAL_ERROR("Failed to get _FillValue for the target mask \""
                            << this->target_mask_variable << "\"")
                        return teca_metadata();
                    }

                    // pass the _FillValue
                    src_array_atts.set("_FillValue", tgt_mask_fill_value);
                    )
            }

            // put source array attributes to the target mesh attributes
            target_atts.set(src_array, src_array_atts);
        }
    }

    // update with merged lists
    out_md.set("variables", target_vars);
    out_md.set("attributes", target_atts);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_apply_tempest_remap::get_upstream_request(unsigned int port,
    const std::vector<teca_metadata> &input_md, const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_apply_tempest_remap::get_upstream_request" << std::endl;
#endif
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // copy the incoming request
    teca_metadata req = request;

    // get the list of requested arrays
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);

    // get the list of source variables
    const teca_metadata &src_md = input_md[0];

    std::set<std::string> src_vars;
    src_md.get("variables", src_vars);

    // get the whole extent (TempestExpects the whole mesh to be present)
    unsigned long src_whole_extent[6] = {0};
    src_md.get("whole_extent", src_whole_extent);

    // get the list of target variables
    const teca_metadata &tgt_md = input_md[1];

    std::set<std::string> tgt_vars;
    tgt_md.get("variables", tgt_vars);

    // get the whole extent (TempestExpects the whole mesh to be present)
    unsigned long tgt_whole_extent[6] = {0};
    tgt_md.get("whole_extent", tgt_whole_extent);

    // route the requested arrays to source or target. it is assumed that the
    // names are unique, use teca_rename_variables upstream if they are not.
    std::vector<std::string> src_arrays;
    std::vector<std::string> tgt_arrays;
    unsigned int n_arrays = req_arrays.size();
    for (unsigned int i = 0; i < n_arrays; ++i)
    {
        const std::string &array = req_arrays[i];

        if (src_vars.count(array))
        {
            // found the requested array in the source dataset
            src_arrays.push_back(array);
            src_arrays.push_back(array + "_valid");
        }
        else if (tgt_vars.count(array))
        {
            // found the requested array in the target dataset
            tgt_arrays.push_back(array);
        }
        else
        {
            TECA_FATAL_ERROR("Neither the source or target data has "
                "a variable named \"" << array << "\"")
            return up_reqs;
        }
    }

    // request the targte coordinate mask
    if (!this->target_mask_variable.empty())
        tgt_arrays.push_back(this->target_mask_variable + "_valid");

    // copy the incoming request and update the requested arrays
    teca_metadata src_req = request;
    src_req.set("arrays", src_arrays);
    src_req.set("extent", src_whole_extent);

    teca_metadata tgt_req = request;
    tgt_req.set("arrays", tgt_arrays);
    tgt_req.set("extent", tgt_whole_extent);

    if (static_target_mesh)
    {
        std::string tgt_idx_req_key;
        tgt_md.get("index_request_key", tgt_idx_req_key);

        tgt_req.set("index_request_key", tgt_idx_req_key);
        tgt_req.set(tgt_idx_req_key, {0ul, 0ul});
    }

    // request the remap weights and indices
    std::vector<std::string> rmp_arrays({this->weights_variable,
        this->row_variable, this->column_variable});

    teca_metadata rmp_req;
    rmp_req.set("arrays", rmp_arrays);

    std::string rmp_idx_req_key;
    const teca_metadata &rmp_md = input_md[2];
    rmp_md.get("index_request_key", rmp_idx_req_key);

    rmp_req.set("index_request_key", rmp_idx_req_key);
    rmp_req.set(rmp_idx_req_key, {0ul, 0ul});

    // package the requests and send upstream
    up_reqs.push_back(src_req);
    up_reqs.push_back(tgt_req);
    up_reqs.push_back(rmp_req);
    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_apply_tempest_remap::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_apply_tempest_remap::execute" << std::endl;
#endif
    (void)port;

    // get the source mesh
    const_p_teca_mesh src_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!src_mesh)
    {
        TECA_FATAL_ERROR("The source dataset is not a teca_mesh")
        return nullptr;
    }

    teca_metadata src_atts;
    src_mesh->get_attributes(src_atts);

    // get the target mesh
    const_p_teca_mesh tgt_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[1]);

    if (!tgt_mesh)
    {
        TECA_FATAL_ERROR("The target dataset is not a teca_mesh")
        return nullptr;
    }

    // get the target coordinate mask
    teca_metadata tgt_atts;
    tgt_mesh->get_attributes(tgt_atts);

    const_p_teca_variant_array tgt_valid;
    if (!this->target_mask_variable.empty())
    {
        tgt_valid =
            tgt_mesh->get_point_arrays()->get(this->target_mask_variable + "_valid");

        if (!tgt_valid)
        {
            TECA_FATAL_ERROR("Failed to get the target mask \"" << this->target_mask_variable
                << "_valid\". Include teca_valid_value_mask in the upstream pipeline.")
            return nullptr;
        }
    }

    // get the remap weights
    const_p_teca_array_collection rmp_coll
        = std::dynamic_pointer_cast<const teca_array_collection>(input_data[2]);

    if (!rmp_coll)
    {
        TECA_FATAL_ERROR("dataset is not a teca_mesh")
        return nullptr;
    }

    const_p_teca_variant_array row = rmp_coll->get(this->row_variable);
    if (!row)
    {
        TECA_FATAL_ERROR("Failed to get row indices no array named \""
            << this->row_variable << "\"")
        return nullptr;
    }

    const_p_teca_variant_array col = rmp_coll->get(this->column_variable);
    if (!col)
    {
        TECA_FATAL_ERROR("Failed to get column indices no array named \""
            << this->column_variable << "\"")
        return nullptr;
    }

    const_p_teca_variant_array weights = rmp_coll->get(this->weights_variable);
    if (!weights)
    {
        TECA_FATAL_ERROR("Failed to get the weights no array names \""
            << this->weights_variable << "\"")
        return nullptr;
    }

    unsigned long n_weights = weights->size();

    // shallow copy the target for the output, passing target data
    // through
    p_teca_mesh out_mesh =
        std::static_pointer_cast<teca_mesh>(tgt_mesh->new_instance());

    out_mesh->shallow_copy(std::const_pointer_cast<teca_mesh>(tgt_mesh));

    unsigned long tgt_nxyz = out_mesh->get_number_of_points();

    // update the time info, which comes from the source
    // create output dataset
    double time = 0.0;
    src_mesh->get_time(time);
    out_mesh->set_time(time);

    unsigned long time_step = 0;
    src_mesh->get_time_step(time_step);
    out_mesh->set_time_step(time_step);

    std::string calendar;
    src_mesh->get_calendar(calendar);
    out_mesh->set_calendar(calendar);

    std::string units;
    src_mesh->get_time_units(units);
    out_mesh->set_time_units(units);

    // get the list of arrays requested
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);

    // get the list of arrays that are simply passed through
    const std::vector<std::string> &tmp = out_mesh->get_point_arrays()->get_names();
    std::set<std::string> tgt_arrays(tmp.begin(), tmp.end());
    std::set<std::string>::iterator tgt_arrays_end = tgt_arrays.end();

    // for each requested array that is not already present in the target,
    // remap from the source
    unsigned int n_arrays = req_arrays.size();
    for (unsigned int i = 0; i < n_arrays; ++i)
    {
        const std::string &array = req_arrays[i];

        if (tgt_arrays.find(array) == tgt_arrays_end)
        {
            // the requested array is not already in the target
            // move it from the source
            const_p_teca_variant_array src_data = src_mesh->get_point_arrays()->get(array);
            if (!src_data)
            {
                TECA_FATAL_ERROR("Failed to get \"" << array << "\" from the source mesh")
                return nullptr;
            }

            const_p_teca_variant_array src_valid =
                src_mesh->get_point_arrays()->get(array + "_valid");

            /*if (!src_valid)
            {
                TECA_FATAL_ERROR("Failed to get \"" << array << "_valid\" from the source mesh."
                    " Include teca_valid_value_mask in the upstream pipeline")
                return nullptr;
            }*/

            // copy the array attributes
            teca_metadata src_array_atts;
            src_atts.get(array, src_array_atts);
            tgt_atts.set(array, src_array_atts);

            p_teca_variant_array tgt_data = src_data->new_instance(tgt_nxyz);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wchar-subscripts"
            NESTED_VARIANT_ARRAY_DISPATCH_I(
                row.get(), _IDX,

                // get the source and target mesh indices
                auto [spr, pr] = get_host_accessible<CTT_IDX>(row);
                auto [spc, pc] = get_host_accessible<CTT_IDX>(col);

                NESTED_VARIANT_ARRAY_DISPATCH_FP(
                    weights.get(), _WGT,

                    // get the weight matrix
                    auto [spw, pw] = get_host_accessible<CTT_WGT>(weights);

                    NESTED_VARIANT_ARRAY_DISPATCH(
                        src_data.get(), _DATA,

                        // get the source and target data
                        auto [ptgt] = data<TT_DATA>(tgt_data);
                        auto [spsrc, psrc] = get_host_accessible<CTT_DATA>(src_data);

                        if (src_valid)
                        {
                            // the source array has a _FillValue and valid
                            // value mask.  move the data only where the source
                            // is valid. assign _FillValue elsewhere.
                            NT_DATA src_fill_value = NT_DATA(0);
                            if (src_array_atts.get("_FillValue", src_fill_value))
                            {
                                TECA_FATAL_ERROR("Failed to get the _FillValue for \"" << array << "\"")
                                return nullptr;
                            }
                            auto [spsv, psrc_valid] = get_host_accessible<CTT_MASK>(src_valid);

                            sync_host_access_any(row, col, weights, src_data, src_valid);

                            for (unsigned long q = 0; q < n_weights; ++q)
                                ptgt[pr[q]] = (psrc_valid[pc[q]] ?
                                    (ptgt[pr[q]] + pw[q] * psrc[pc[q]]) : src_fill_value);
                        }
                        else
                        {
                            sync_host_access_any(row, col, weights, src_data);

                            // the source array didn't provide a _FillValue or
                            // valid value mask. move all of the data
                            for (unsigned long q = 0; q < n_weights; ++q)
                                ptgt[pr[q]] = ptgt[pr[q]] + pw[q] * psrc[pc[q]];
                        }
                        )
                    )
                )
#pragma GCC diagnostic pop
            // un-pack the result. TempestRemap packs the result one value per valid
            // mesh point. In the GOES data some mesh points are invalid. To put the
            // mapped data back into the GOES mesh we'll need to un-pack the result.
            if (tgt_valid)
            {
                teca_metadata tgt_mask_atts;
                if (!src_valid)
                    tgt_atts.get(this->target_mask_variable, tgt_mask_atts);

                auto [sptgt_valid, ptgt_valid] = get_host_accessible<CTT_MASK>(tgt_valid);

                NESTED_VARIANT_ARRAY_DISPATCH(
                    tgt_data.get(), _DATA,

                    // get the _FillValue from the source (that's what will be declared in the
                    // NetCDF variable attriibutes), but if it is not present get it from the
                    // mask variable's attributes.
                    NT_DATA fill_value = NT_DATA(0);
                    if (src_array_atts.get("_FillValue", fill_value) &&
                        tgt_mask_atts.get("_FillValue", fill_value))
                    {
                        TECA_FATAL_ERROR("Failed to get the _FillValue from \"" << array
                            << "\" and \"" << this->target_mask_variable << "\"")
                        return nullptr;
                    }

                    // get the target data, and allocate the output
                    auto [ptgt] = data<TT_DATA>(tgt_data);
                    auto [tgt_out, ptgt_out] = ::New<TT_DATA>(tgt_nxyz, fill_value);

                    sync_host_access_any(tgt_valid);

                    for (unsigned long q = 0, p = 0; q < tgt_nxyz; ++q)
                    {
                        if (ptgt_valid[q])
                        {
                            // copy the packed data into the correct location in the output
                            ptgt_out[q] = ptgt[p];

                            // queue up the next packed location
                            ++p;
                        }
                    }

                    tgt_data = tgt_out;
                    )
            }

            // store the result in the output mesh
            out_mesh->get_point_arrays()->append(array, tgt_data);
        }
    }

    out_mesh->set_attributes(tgt_atts);

    return out_mesh;
}
