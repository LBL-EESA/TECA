#include "teca_temporal_index_select.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG


// --------------------------------------------------------------------------
teca_temporal_index_select::teca_temporal_index_select()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_temporal_index_select::~teca_temporal_index_select()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_temporal_index_select::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_temporal_index_select":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<long long>, prefix, indices,
            "a list of the time indices to select")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_temporal_index_select::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<long long>, prefix, indices)
}
#endif


// --------------------------------------------------------------------------
teca_metadata teca_temporal_index_select::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &md_in)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_temporal_index_select::get_output_metadata" << std::endl;
#endif
    (void) port;

    // copy the incoming metadata
    teca_metadata md_out = teca_metadata(md_in[0]);

    size_t n_elem = this->indices.size();

    // skip modifying the coordinates if no indices have been requested
    if (n_elem)
    {
        // revise the time coordinate to be the subset that was requested
        teca_metadata coords;
        p_teca_variant_array t_in;
        if ( md_out.get("coordinates", coords) || !(t_in = coords.get("t")) )
        {
            TECA_FATAL_ERROR("failed to locate target mesh coordinates")
            return md_out;
        }

        // create the output times by selecting times at the given indices
        p_teca_variant_array t_out;
        VARIANT_ARRAY_DISPATCH_FP(t_in.get(),

            auto [tmp, p_tmp] = ::New<TT>(n_elem);
            auto [sp_t_in, p_t_in] = get_host_accessible<CTT>(t_in);

            sync_host_access_any(t_in);

            for ( size_t i = 0; i < n_elem; ++i )
            {
                // set this time value
                p_tmp[i] = p_t_in[this->indices[i]];
            }

            t_out = tmp;
        )

        // overwrite the coordinates
        coords.set("t", t_out);
        md_out.set("coordinates", coords);

        // get the key for the # of timesteps
        std::string init_key;
        if (md_out.get("index_initializer_key", init_key))
        {
            TECA_FATAL_ERROR("failed to get index initializer key")
            return md_out;
        }

        // reset the length of the time coordinate
        md_out.set(init_key, n_elem);
    }

    // return the output metadata
    return md_out;

}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_temporal_index_select::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &md_in,
    const teca_metadata &request_in)
{
    (void)port;
    (void)md_in;

    std::vector<teca_metadata> up_reqs;

    // skip modifying the request if no indices have been requested
    if (! (this->indices.size() == 0) )
    {
        // get the key for the time indices requested
        std::string key;
        if (request_in.get("index_request_key", key))
        {
            TECA_FATAL_ERROR("failed to get index request key")
            return up_reqs;
        }
        std::vector<size_t> index_extent;
        request_in.get(key, index_extent);

        // create a request for each index in the request
        for ( auto i = index_extent[0]; i <= index_extent[1]; ++i)
        {
            // check that this index is valid
            if ( i >= this->indices.size() )
            {
                TECA_FATAL_ERROR(
                    << "index_request_key: " <<
                    index_extent[0] << ":" << index_extent[1]
                    << std::endl
                    << "Bad request: list index " << i
                    << " is out of bounds of the index list of size "
                    << this->indices.size())
            }

            // copy the incoming request
            teca_metadata req = teca_metadata(request_in);

            // map the incoming request index to the list of temporal indices provided
            std::vector<size_t> new_indices = {0,0};
            new_indices[0] = this->indices[i];
            new_indices[1] = this->indices[i];
            req.set(key, new_indices);

            // append this request to the list
            up_reqs.push_back(req);
        }
    }
    else
    {
        // just pass along the input request
        up_reqs.push_back(request_in);
    }

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_temporal_index_select::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_temporal_index_select::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    std::ostringstream oss;

    // check for valid incoming data
    if (input_data.size() == 0)
    {
        TECA_FATAL_ERROR("Failed to obtain indices; incoming dataset is empty")
        return nullptr;
    }

    // get the input mesh
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to obtain indices; dataset is not a teca_mesh")
        return nullptr;
    }

    // default to CPU operations
    int device_id = -1;
    allocator alloc = allocator::malloc;
#if defined(TECA_HAS_CUDA)
    // if we are assigned a gpu, put the output on that device
    request.get("device_id",device_id);
    if ( device_id >= 0 )
    {
        if (teca_cuda_util::set_device(device_id))
            return nullptr;

        alloc = allocator::cuda_async;
    }
#endif

    // get the key for the time indices requested
    std::string key;
    if (request.get("index_request_key", key))
    {
        TECA_FATAL_ERROR("failed to get index request key")
        return nullptr;
    }

    size_t index_extent[2];
    request.get(key, index_extent);
    if ( this->get_verbose() )
    {
        std::vector<long long> ind_request;
        for (size_t i = index_extent[0]; i <= index_extent[1]; ++i)
        {
            ind_request.push_back(this->indices[i]);
        }
        oss << "(device_id=" << device_id << "): "
            << "Mapping request " << index_extent[0] << ":" << index_extent[1]
            << " to indices " << ind_request;

        TECA_STATUS(<< oss.str())
    }

    // create the output mesh
    p_teca_mesh out_mesh = std::static_pointer_cast<teca_mesh>
        (std::const_pointer_cast<teca_mesh>(in_mesh)->new_shallow_copy());


    // proceed if there are indices to remap; otherwise pass data along
    if ( this->indices.size() > 0 )
    {
        // get the arrays and their names
        p_teca_array_collection arrays = out_mesh->get_point_arrays();
        std::vector<std::string> array_names = arrays->get_names();

        // pass the output mesh through if there are no arrays
        if ( array_names.size() == 0 )
            return out_mesh;

        // loop over the point arrays
        for ( auto& array_name : array_names)
        {
            // get the shape of one timestep of this array
            const_p_teca_variant_array array_in = arrays->get(array_name);
            size_t nxyz = array_in->size();

            // set the shape of the output array
            size_t nxyzt = nxyz * input_data.size();

            // pre-allocate the output array on the specified device sized so
            // it can accommodate all timesteps.
            p_teca_variant_array array_out = array_in->new_instance(nxyzt, alloc);

            // loop over the requests
            size_t n_steps = input_data.size();
            for ( size_t i = 0; i < n_steps; ++i )
            {
                // get the current timestep's data
                p_teca_mesh tmp_data = std::static_pointer_cast<teca_mesh>
                    (std::const_pointer_cast<teca_mesh>(
                    std::dynamic_pointer_cast<const teca_mesh>(input_data[i])));

                // get the current timestep's array
                p_teca_variant_array tmp_array =
                    tmp_data->get_point_arrays()->get(array_name);

                // insert data into the output array
                array_out->set(i*nxyz, tmp_array, 0, nxyz);
            }
            // insert the new array into the output dataset
            arrays->set(array_name, array_out);
        }

        // set the reported time extent to be the same as the request
        out_mesh->set_request_indices(key, index_extent);
    }

    TECA_STATUS("Mapping complete.")

    // return the modifed dataset
    return out_mesh;
}
