#ifndef teca_metadata_util_h
#define teca_metadata_util_h

/// @file

#include "teca_config.h"
#include "teca_common.h"
#include "teca_metadata.h"

#include <string>


/// Codes for dealing with teca_metadata
namespace teca_metadata_util
{
/** Given a collection of array attributes (following the conventions used by
 * the teca_cf_reader) and a mesh extent, compute and return the valid extent
 * of the array. This takes into account 1d and 2d arrays on a 3d mesh. Return
 * zero if successful. The mesh_dims_active key is required, if not found 1
 * is returned and the array_extent is set to the mesh_extent.
 */
TECA_EXPORT
int get_array_extent(const teca_metadata &array_attributes,
    const unsigned long mesh_extent[8], unsigned long array_extent[8]);

/** Get the requested index extent from the request.
 *
 * @param[in] request the request
 * @param[out] request_key the name of the key holding the index
 * @apram[out] indices the requested index extent
 * @returns zero if the index was successfully obtained.
 */
template <typename index_t>
int get_requested_indices(const teca_metadata &request,
    std::string &request_key, index_t indices[2])
{
    // get the requested index
    if (request.get("index_request_key", request_key))
    {
        TECA_ERROR("Failed to locate the index_request_key")
        return -1;
    }

    if (request.get(request_key, indices))
    {
        TECA_ERROR("Failed to get the requested indices using the"
            " index_request_key \"" << request_key << "\"")
        return -1;
    }

    return 0;
}

/** Get the requested index from the request. Use this function when the
 * algorithm can not handle requests for multiple indices. A check is performed
 * to ensure that only one index was requested.
 *
 * @param[in] request the request
 * @param[out] request_key the name of the key holding the index
 * @apram[out] index the requested index
 * @returns zero if the index was successfully obtained.
 */
template <typename index_t>
int get_requested_index(const teca_metadata &request,
    std::string &request_key, index_t &index)
{
    // get the requested index
    if (request.get("index_request_key", request_key))
    {
        TECA_ERROR("Failed to locate the index_request_key")
        return -1;
    }

    index_t indices[2];
    if (request.get(request_key, indices))
    {
        TECA_ERROR("Failed to get the requested index using the"
            " index_request_key \"" << request_key << "\"")
        return -1;
    }

    index_t n_indices = indices[1] - indices[0] + 1;
    if (n_indices != 1)
    {
        TECA_ERROR(<< n_indices << " requested when one was required")
        return -1;
    }

    index = indices[0];

    return 0;
}

};

#endif
