#include "teca_temporal_average.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>

using std::vector;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_temporal_average::teca_temporal_average() : filter_width(3)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_temporal_average::~teca_temporal_average()
{}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_temporal_average::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void) port;

    vector<teca_metadata> up_reqs;

    // get the time values required to compute the average
    // centered on the requested time
    long active_step;
    if (request.get("time_step", active_step))
    {
        TECA_ERROR("request is missing \"time_step\"")
        return up_reqs;
    }

    long num_steps;
    if (input_md[0].get("number_of_time_steps", num_steps))
    {
        TECA_ERROR("input is missing \"number_of_time_steps\"")
        return up_reqs;
    }

    long first = active_step - this->filter_width/2;
    long last = active_step + this->filter_width/2;
    for (long i = first; i <= last; ++i)
    {
        // make a request for each time that will be used in the
        // average
        if ((i >= 0) && (i < num_steps))
        {
            teca_metadata up_req(request);
            up_req.insert("time_step", i);
            up_reqs.push_back(up_req);
        }
    }

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_temporal_average::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "teca_temporal_average::execute" << endl;
#endif
    (void)port;

    // create output and copy metadata, coordinates, etc
    p_teca_mesh out_mesh
        = std::dynamic_pointer_cast<teca_mesh>(input_data[0]->new_instance());

    if (!out_mesh)
    {
        // TODO -- add support for other datatypes
        TECA_ERROR("input data[0] is not a teca_mesh")
        return nullptr;
    }

    // initialize the output array collections from the
    // first input
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("Failed to average. dataset is not a teca_mesh")
        return nullptr;
    }

    size_t n_meshes = input_data.size();

    // TODO -- handle cell, edge, face arrays
    p_teca_array_collection out_arrays = out_mesh->get_point_arrays();

    // initialize with a copy of the first dataset
    out_arrays->copy(in_mesh->get_point_arrays());
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

            TEMPLATE_DISPATCH(
                teca_variant_array_impl,
                out_a.get(),

                const NT *p_in_a = dynamic_cast<const TT*>(in_a.get())->get();
                NT *p_out_a = dynamic_cast<TT*>(out_a.get())->get();

                for (size_t q = 0; q < n_elem; ++q)
                    p_out_a[q] += p_in_a[q];
                )
        }
    }

    // scale by the filter width
    for (size_t i = 0; i < n_meshes; ++i)
    {
        for (size_t j = 0; j < n_arrays; ++j)
        {
            p_teca_variant_array out_a = out_arrays->get(j);

            TEMPLATE_DISPATCH(
                teca_variant_array_impl,
                out_a.get(),

                NT *p_out_a = dynamic_cast<TT*>(out_a.get())->get();
                NT fac = static_cast<NT>(n_meshes);

                for (size_t q = 0; q < n_elem; ++q)
                    p_out_a[q] /= fac;
                )
        }
    }

    // get active time step
    unsigned long active_step;
    if (request.get("time_step", active_step))
    {
        TECA_ERROR("request is missing \"time_step\"")
        return nullptr;
    }

    // copy metadata and information arrays from the
    // active step
    for (size_t i = 0; i < n_meshes; ++i)
    {
        in_mesh = std::dynamic_pointer_cast<const teca_mesh>(input_data[i]);

        unsigned long step;
        if (in_mesh->get_metadata().get("time_step", step))
        {
            TECA_ERROR("input dataset metadata missing \"time_step\"")
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
