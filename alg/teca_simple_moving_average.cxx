#include "teca_simple_moving_average.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_metadata_util.h"

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

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_simple_moving_average::teca_simple_moving_average()
    : filter_width(3), filter_type(backward)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_simple_moving_average::~teca_simple_moving_average()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_simple_moving_average::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_simple_moving_average":prefix));

    opts.add_options()
        TECA_POPTS_GET(unsigned int, prefix, filter_width,
            "number of steps to average over")
        TECA_POPTS_GET(int, prefix, filter_type,
            "use a backward(0), forward(1) or centered(2) stencil")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_simple_moving_average::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, unsigned int, prefix, filter_width)
    TECA_POPTS_SET(opts, int, prefix, filter_type)
}
#endif

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_simple_moving_average::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::string type = "unknown";
    switch(this->filter_type)
    {
        case backward:
            type = "backward";
            break;
        case centered:
            type = "centered";
            break;
        case forward:
            type = "forward";
            break;
    }
    cerr << teca_parallel_id()
        << "teca_simple_moving_average::get_upstream_request filter_type="
        << type << endl;
#endif
    (void) port;

    vector<teca_metadata> up_reqs;

    // get the time values required to compute the average centered on the
    // requested time
    long active_step = 0;
    std::string request_key;
    if (teca_metadata_util::get_requested_index(request, request_key, active_step))
    {
        TECA_FATAL_ERROR("Failed to determine the requested index")
        return up_reqs;
    }

    // get the number of time steps available
    long num_steps;
    std::string initializer_key;
    if (input_md[0].get("index_initializer_key", initializer_key) ||
        input_md[0].get(initializer_key, num_steps))
    {
        TECA_FATAL_ERROR("Failed to determine the number of time steps available")
        return up_reqs;
    }

    long first = 0;
    long last = 0;
    switch(this->filter_type)
    {
        case backward:
            first = active_step - this->filter_width + 1;
            last = active_step;
            break;
        case centered:
            {
            if (this->filter_width % 2 == 0)
                TECA_FATAL_ERROR("\"filter_width\" should be odd for centered calculation")
            long delta = this->filter_width/2;
            first = active_step - delta;
            last = active_step + delta;
            }
            break;
        case forward:
            first = active_step;
            last = active_step + this->filter_width - 1;
            break;
        default:
            TECA_FATAL_ERROR("Invalid \"filter_type\" " << this->filter_type)
            return up_reqs;
    }
    first = std::max(0l, first);
    last = std::min(num_steps - 1, last);

    // make a request for each time that will be used in the average
    for (long i = first; i <= last; ++i)
    {
        teca_metadata up_req(request);
        up_req.set(request_key, {i, i});
        up_reqs.push_back(up_req);
    }

#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "processing " << active_step
        << " request " << first << " - " << last << endl;
#endif

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_simple_moving_average::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_simple_moving_average::execute" << endl;
#endif
    (void)port;

    // nothing to do
    if ((input_data.size() < 1) || !input_data[0])
        return nullptr;

    // create output and copy metadata, coordinates, etc
    p_teca_mesh out_mesh
        = std::dynamic_pointer_cast<teca_mesh>(input_data[0]->new_instance());

    if (!out_mesh)
    {
        TECA_FATAL_ERROR("input data[0] is not a teca_mesh")
        return nullptr;
    }

    // initialize the output array collections from the
    // first input
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to average. dataset is not a teca_mesh")
        return nullptr;
    }

    size_t n_meshes = input_data.size();

    // TODO -- handle cell, edge, face arrays
    p_teca_array_collection out_arrays = out_mesh->get_point_arrays();

    // initialize with a copy of the first dataset
    out_arrays->copy(in_mesh->get_point_arrays(), allocator::malloc);

    size_t n_arrays = out_arrays->size();
    size_t n_elem = n_arrays ? out_arrays->get(0)->size() : 0;

    // accumulate each array from remaining datasets
    for (size_t i = 1; i < n_meshes; ++i)
    {
        in_mesh = std::dynamic_pointer_cast<const teca_mesh>(input_data[i]);

        const_p_teca_array_collection  in_arrays = in_mesh->get_point_arrays();

        for (size_t j = 0; j < n_arrays; ++j)
        {
            const_p_teca_variant_array in_a = in_arrays->get(j);
            p_teca_variant_array out_a = out_arrays->get(j);

            VARIANT_ARRAY_DISPATCH(in_a.get(),

                auto [sp_in_a, p_in_a] = get_host_accessible<CTT>(in_a);
                auto [p_out_a] = data<TT>(out_a);

                sync_host_access_any(in_a);

                for (size_t q = 0; q < n_elem; ++q)
                    p_out_a[q] += p_in_a[q];
                )
        }
    }

    // scale result by the filter width
    for (size_t j = 0; j < n_arrays; ++j)
    {
        p_teca_variant_array out_a = out_arrays->get(j);

        VARIANT_ARRAY_DISPATCH(out_a.get(),

            auto [p_out_a] = data<TT>(out_a);

            NT fac = static_cast<NT>(n_meshes);

            for (size_t q = 0; q < n_elem; ++q)
                p_out_a[q] /= fac;
            )
    }

    // get active time step
    std::string request_key;
    unsigned long active_step;
    if (teca_metadata_util::get_requested_index(request,
        request_key, active_step))
    {
        TECA_FATAL_ERROR("Failed to determine the requested time step")
        return nullptr;
    }

    // copy metadata and information arrays from the
    // active step
    for (size_t i = 0; i < n_meshes; ++i)
    {
        in_mesh = std::dynamic_pointer_cast<const teca_mesh>(input_data[i]);

        unsigned long step;
        if (in_mesh->get_time_step(step))
        {
            TECA_FATAL_ERROR("input dataset metadata missing \"time_step\"")
            return nullptr;
        }

        if (step == active_step)
        {
            out_mesh->copy_metadata(in_mesh);
            out_mesh->get_information_arrays()->copy(in_mesh->get_information_arrays());
        }
    }

    return out_mesh;
}
