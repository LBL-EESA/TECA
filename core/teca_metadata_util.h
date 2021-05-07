#ifndef teca_metadata_util_h
#define teca_metadata_util_h

/// @file

class teca_metadata;

/// Codes for dealing with teca_metadata
namespace teca_metadata_util
{
/** Given a collection of array attributes (following the conventions used by
 * the teca_cf_reader) and a mesh extent, compute and return the valid extent
 * of the array. This takes into account 1d and 2d arrays on a 3d mesh. Return
 * zero if successful. The mesh_dims_active key is required, if not found 1
 * is returned and the array_extent is set to the mesh_extent.
 */
int get_array_extent(const teca_metadata &array_attributes,
    const unsigned long mesh_extent[6], unsigned long array_extent[6]);
};

#endif
