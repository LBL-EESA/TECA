#ifndef teca_cartesian_mesh_util_h
#define teca_cartesian_mesh_util_h

#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"

#include <vector>

// binary search that will locate index bounding the value
// above or below such that data[i] <= val or val <= data[i+1]
// depending on the value of lower. return 0 if the value is
// found.
template <typename T>
int index_of(const T *data, size_t l, size_t r, T val, bool lower, unsigned long &id)
{
    unsigned long m_0 = (r + l)/2;
    unsigned long m_1 = m_0 + 1;

    if (m_0 == r)
    {
        // not found
        return -1;
    }
    else
    if ((val >= data[m_0]) && (val <= data[m_1]))
    {
        // found the value!
        if (lower)
            id = m_0;
        else
            id = m_1;
        return 0;
    }
    else
    if (val < data[m_0])
    {
        // split range to the left
        return index_of(data, l, m_0, val, lower, id);
    }
    else
    {
        // split the range to the right
        return index_of(data, m_1, r, val, lower, id);
    }

    // not found
    return -1;
}

// return the extent corresponding to the
// supplied bounding box. when cover_bbox is true the
// bounds of the returned extent covers the supplied
// bounding box. when cover_bbox is false the returned
// extent is the largest extent contained by the bounding
// box. return 0 if successful.
template <typename T>
int bounds_to_extent(
    T low_x,  T high_x, T low_y, T high_y, T low_z, T high_z,
    const T *p_x, const T *p_y, const T *p_z,
    unsigned long high_i, unsigned long high_j, unsigned long high_k,
    bool cover_bbox, std::vector<unsigned long> &extent)
{
    extent.resize(6, 0l);
    if ((high_i && (index_of(p_x, 0, high_i, low_x, cover_bbox, extent[0])
        || index_of(p_x, 0, high_i, high_x, !cover_bbox, extent[1])))
        || (high_j && (index_of(p_y, 0, high_j, low_y, cover_bbox, extent[2])
        || index_of(p_y, 0, high_j, high_y, !cover_bbox, extent[3])))
        || (high_k && (index_of(p_z, 0, high_k, low_z, cover_bbox, extent[4])
        || index_of(p_z, 0, high_k, high_z, !cover_bbox, extent[5]))))
    {
        return -1;
    }
    return 0;
}

// get the i,j,k cell index of point x,y,z in the given mesh.
// return 0 if successful.
template<typename T>
int index_of(const const_p_teca_cartesian_mesh &mesh, T x, T y, T z,
        unsigned long &i, unsigned long &j, unsigned long &k)
{
    const_p_teca_variant_array xc = mesh->get_x_coordinates();
    const_p_teca_variant_array yc = mesh->get_y_coordinates();
    const_p_teca_variant_array zc = mesh->get_z_coordinates();

    TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        xc.get(),

        NT *p_xc = std::dynamic_pointer_cast<TT>(xc)->get();
        NT *p_yc = std::dynamic_pointer_cast<TT>(yc)->get();
        NT *p_zc = std::dynamic_pointer_cast<TT>(zc)->get();

        unsigned long nx = xc->size();
        unsigned long ny = yc->size();
        unsigned long nz = zc->size();

        if (index_of(p_xc, 0, nx-1, x, true, i)
            || index_of(p_yc, 0, ny-1, y, true, j)
            || index_of(p_zc, 0, nz-1, z, true, k))
        {
            // out of bounds
            return -1;
        }

        // success
        return 0;
        )

    // coords are not a floating point type
    return -1;
}

#endif
