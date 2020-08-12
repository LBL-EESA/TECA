#ifndef teca_cartesian_mesh_util_h
#define teca_cartesian_mesh_util_h

#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <vector>
#include <cmath>

namespace teca_coordinate_util
{
template <typename T>
bool equal(T a, T b, T tol)
{
    T diff = std::abs(a - b);
    a = std::abs(a);
    b = std::abs(b);
    b = (b > a) ? b : a;
    if (diff <= (b*tol))
        return true;
    return false;
}

// comparators implementing bracket for ascending and
// descending input arrays
template<typename data_t>
struct leq
{ static bool eval(const data_t &l, const data_t &r) { return l <= r; } };

template<typename data_t>
struct geq
{ static bool eval(const data_t &l, const data_t &r) { return l >= r; } };

template<typename data_t>
struct lt
{ static bool eval(const data_t &l, const data_t &r) { return l < r; } };

template<typename data_t>
struct gt
{ static bool eval(const data_t &l, const data_t &r) { return l > r; } };

template<typename data_t>
struct ascend_bracket
{
    using comp0_t = geq<data_t>;
    using comp1_t = leq<data_t>;
};

template<typename data_t>
struct descend_bracket
{
    using comp0_t = leq<data_t>;
    using comp1_t = geq<data_t>;
};

// binary search that will locate index bounding the value
// above or below such that data[i] <= val or val <= data[i+1]
// depending on the value of lower. return 0 if the value is
// found. the comp0 and comp1 template parameters let us
// operate on both ascending and descending input. defaults
// are set for ascending inputs.
template <typename data_t, typename bracket_t = ascend_bracket<data_t>>
int index_of(const data_t *data, unsigned long l, unsigned long r,
    data_t val, bool lower, unsigned long &id)
{
    unsigned long m_0 = (r + l)/2;
    unsigned long m_1 = m_0 + 1;

    if (m_0 == r)
    {
        data_t eps8 = data_t(8)*std::numeric_limits<data_t>::epsilon();
        if (equal(val, data[m_0], eps8))
        {
            id = m_0;
            return 0;
        }
        // not found
        return -1;
    }
    else
    if (bracket_t::comp0_t::eval(val, data[m_0]) &&
         bracket_t::comp1_t::eval(val, data[m_1]))
    {
        data_t eps8 = data_t(8)*std::numeric_limits<data_t>::epsilon();
        // found a bracket around the value
        if (equal(val, data[m_0], eps8))
            id = m_0;
        else
        if (equal(val, data[m_1], eps8))
            id = m_1;
        else
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
        return teca_coordinate_util::index_of<data_t, bracket_t>(
            data, l, m_0, val, lower, id);
    }
    else
    {
        // split the range to the right
        return teca_coordinate_util::index_of<data_t, bracket_t>(
            data, m_1, r, val, lower, id);
    }

    // not found
    return -1;
}

// binary search that will locate index of the given value.
// return 0 if the value is found.
template <typename T>
int index_of(const T *data, size_t l, size_t r, T val, unsigned long &id)
{
    unsigned long m_0 = (r + l)/2;
    unsigned long m_1 = m_0 + 1;

    if (m_0 == r)
    {
        // need this test when len of data is 1
        if (equal(val, data[m_0], T(8)*std::numeric_limits<T>::epsilon()))
        {
            id = m_0;
            return 0;
        }
        // not found
        return -1;
    }
    else
    if (equal(val, data[m_0], T(8)*std::numeric_limits<T>::epsilon()))
	{
        id = m_0;
        return 0;
    }
    else
    if (equal(val, data[m_1], T(8)*std::numeric_limits<T>::epsilon()))
	{
        id = m_1;
        return 0;
    }
    else
    if (val < data[m_0])
    {
        // split range to the left
        return teca_coordinate_util::index_of(data, l, m_0, val, id);
    }
    else
    {
        // split the range to the right
        return teca_coordinate_util::index_of(data, m_1, r, val, id);
    }

    // not found
    return -1;
}

// convert bounds to extents
// return non-zero if the requested bounds are not in
// the given coordinate arrays. coordinate arrays must
// not be empty.
int bounds_to_extent(const double *bounds,
    const_p_teca_variant_array x, const_p_teca_variant_array y,
    const_p_teca_variant_array z, unsigned long *extent);

int bounds_to_extent(const double *bounds, const teca_metadata &md,
    unsigned long *extent);

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

        const NT *p_xc = std::dynamic_pointer_cast<TT>(xc)->get();
        const NT *p_yc = std::dynamic_pointer_cast<TT>(yc)->get();
        const NT *p_zc = std::dynamic_pointer_cast<TT>(zc)->get();

        unsigned long nx = xc->size();
        unsigned long ny = yc->size();
        unsigned long nz = zc->size();

        if (teca_coordinate_util::index_of(p_xc, 0, nx-1, x, true, i)
            || teca_coordinate_util::index_of(p_yc, 0, ny-1, y, true, j)
            || teca_coordinate_util::index_of(p_zc, 0, nz-1, z, true, k))
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

// given a human readable date string in YYYY-MM-DD hh:mm:ss format
// amd a list of floating point offset times inthe specified calendar
// and units find the closest time step. return 0 if successful
// see index_of for a description of lower, if clamp is true then
// when the date falls outside of the time values either the first
// or last time step is returned.
int time_step_of(p_teca_variant_array time, bool lower, bool clamp,
    const std::string &calendar, const std::string &units,
    const std::string &date, unsigned long &step);

// given a time value (val), associated time units (units), and calendar
// (calendar), return a human-readable rendering of the date (date) in a
// strftime-format (format).  return 0 if successful.
int time_to_string(double val, const std::string &calendar,
    const std::string &units, const std::string &format, std::string &date);

// build random access data structures for an indexed table.
// the index column gives each entity a unique id. the index is
// used to identify rows that belong in the entity. it is assumed
// that an entity ocupies consecutive rows. the returns are:
// n_entities, the number of entities found; counts, the number of
// rows used by each entity; offsets, the starting row of each
// entity; ids, a new set of ids for the entities starting from 0
template <typename int_t>
void get_table_offsets(const int_t *index, unsigned long n_rows,
    unsigned long &n_entities, std::vector<unsigned long> &counts,
    std::vector<unsigned long> &offsets, std::vector<unsigned long> &ids)
{
    // count unique number of steps and compute an array index
    // from each time step.
    n_entities = 1;
    ids.resize(n_rows);
    unsigned long n_m1 = n_rows - 1;
    for (unsigned long i = 0; i < n_m1; ++i)
    {
        ids[i] = n_entities - 1;
        if (index[i] != index[i+1])
            ++n_entities;
    }
    ids[n_m1] = n_entities - 1;

    // compute num storms in each step
    counts.resize(n_entities);
    unsigned long q = 0;
    for (unsigned long i = 0; i < n_entities; ++i)
    {
        counts[i] = 1;
        while ((q < n_m1) && (index[q] == index[q+1]))
        {
          ++counts[i];
          ++q;
        }
        ++q;
    }

    // compute the offset to the first storm in each step
    offsets.resize(n_entities);
    offsets[0] = 0;
    for (unsigned long i = 1; i < n_entities; ++i)
        offsets[i] = offsets[i-1] + counts[i-1];
}

// 0 order (nearest neighbor) interpolation
// for nodal data on stretched cartesian mesh.
template<typename CT, typename DT>
int interpolate_nearest(CT cx, CT cy, CT cz,
    const CT *p_x, const CT *p_y, const CT *p_z,
    const DT *p_data, unsigned long ihi, unsigned long jhi,
    unsigned long khi, unsigned long nx, unsigned long nxy,
    DT &val)
{
    // get i,j of node less than cx,cy
    unsigned long i = 0;
    unsigned long j = 0;
    unsigned long k = 0;

    if ((ihi && teca_coordinate_util::index_of(p_x, 0, ihi, cx, true, i))
        || (jhi && teca_coordinate_util::index_of(p_y, 0, jhi, cy, true, j))
        || (khi && teca_coordinate_util::index_of(p_z, 0, khi, cz, true, k)))
    {
        // cx,cy,cz is outside the coordinate axes
        return -1;
    }

    // get i,j of node greater than cx,cy
    unsigned long ii = std::min(i + 1, ihi);
    unsigned long jj = std::min(j + 1, jhi);
    unsigned long kk = std::min(k + 1, khi);

    // get index of nearest node
    unsigned long p = (cx - p_x[i]) <= (p_x[ii] - cx) ? i : ii;
    unsigned long q = (cy - p_y[j]) <= (p_y[jj] - cy) ? j : jj;
    unsigned long r = (cz - p_z[k]) <= (p_z[kk] - cz) ? k : kk;

    // assign value from nearest node
    val = p_data[p + nx*q + nxy*r];
    return 0;
}

// 1 order (linear) interpolation
// for nodal data on stretched cartesian mesh.
template<typename CT, typename DT>
int interpolate_linear(CT cx, CT cy, CT cz,
    const CT *p_x, const CT *p_y, const CT *p_z,
    const DT *p_data, unsigned long ihi, unsigned long jhi,
    unsigned long khi, unsigned long nx, unsigned long nxy,
    DT &val)
{
    // get i,j of node less than cx,cy
    unsigned long i = 0;
    unsigned long j = 0;
    unsigned long k = 0;

    if ((ihi && teca_coordinate_util::index_of(p_x, 0, ihi, cx, true, i))
        || (jhi && teca_coordinate_util::index_of(p_y, 0, jhi, cy, true, j))
        || (khi && teca_coordinate_util::index_of(p_z, 0, khi, cz, true, k)))
    {
        // cx,cy,cz is outside the coordinate axes
        return -1;
    }

    // get i,j of node greater than cx,cy
    unsigned long ii = std::min(i + 1, ihi);
    unsigned long jj = std::min(j + 1, jhi);
    unsigned long kk = std::min(k + 1, khi);

    // compute weights
    CT wx = (cx - p_x[i])/(p_x[ii] - p_x[i]);
    CT wy = (cy - p_y[i])/(p_y[ii] - p_y[i]);
    CT wz = (cz - p_z[i])/(p_z[ii] - p_z[i]);

    CT vx = CT(1) - wx;
    CT vy = CT(1) - wy;
    CT vz = CT(1) - wz;

    // interpolate
    val = vx*vy*vz*p_data[ i +  j*nx +  k*nxy]
        + wx*vy*vz*p_data[ii +  j*nx +  k*nxy]
        + wx*wy*vz*p_data[ii + jj*nx +  k*nxy]
        + vx*wy*vz*p_data[ i + jj*nx +  k*nxy]
        + vx*vy*wz*p_data[ i +  j*nx + kk*nxy]
        + wx*vy*wz*p_data[ii +  j*nx + kk*nxy]
        + wx*wy*wz*p_data[ii + jj*nx + kk*nxy]
        + vx*wy*wz*p_data[ i + jj*nx + kk*nxy];

    return 0;
}

// functor templated on order of accuracy for above Cartesian mesh interpolants
template<int> struct interpolate_t;

template<> struct interpolate_t<0>
{
    template<typename CT, typename DT>
    int operator()(CT tx, CT ty, CT tz, const CT *sx, const CT *sy,
        const CT *sz, const DT *sa, unsigned long ihi, unsigned long jhi,
        unsigned long khi, unsigned long nx, unsigned long nxy, DT &ta)
    {
        return teca_coordinate_util::interpolate_nearest(tx,ty,tz,
             sx,sy,sz,sa, ihi,jhi,khi, nx,nxy, ta);
    }
};

template<> struct interpolate_t<1>
{
    template<typename CT, typename DT>
    int operator()(CT tx, CT ty, CT tz, const CT *sx, const CT *sy,
        const CT *sz, const DT *sa, unsigned long ihi, unsigned long jhi,
        unsigned long khi, unsigned long nx,unsigned long nxy, DT &ta)
    {
        return teca_coordinate_util::interpolate_linear(tx,ty,tz,
             sx,sy,sz,sa, ihi,jhi,khi, nx,nxy, ta);
    }
};

};
#endif
