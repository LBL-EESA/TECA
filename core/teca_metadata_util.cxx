#include "teca_metadata_util.h"

#include "teca_metadata.h"
#include "teca_common.h"

namespace teca_metadata_util
{

// **************************************************************************
int get_array_extent(const teca_metadata &array_attributes,
    const unsigned long mesh_extent[8], unsigned long array_extent[8])
{
    for (int i = 0; i < 8; ++i)
        array_extent[i] = mesh_extent[i];

    unsigned long dim_active[4] = {0ul};
    if (array_attributes.get("mesh_dim_active", dim_active, 4))
    {
        //TECA_ERROR("metadata issue. The array attributes collection is"
        //    " missing the mesh_dim_active key")
        return -1;
    }

    // make the extent 1 in any direction that this array is undefined in
    if (!dim_active[0])
        array_extent[1] = array_extent[0] = 0;

    if (!dim_active[1])
        array_extent[3] = array_extent[2] = 0;

    if (!dim_active[2])
        array_extent[5] = array_extent[4] = 0;

    if (!dim_active[3])
        array_extent[7] = array_extent[8] = 0;

    return 0;
}

};
