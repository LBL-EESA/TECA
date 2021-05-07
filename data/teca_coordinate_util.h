#ifndef teca_cartesian_mesh_util_h
#define teca_cartesian_mesh_util_h

/// @file

#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <vector>
#include <cmath>
#include <type_traits>
#include <typeinfo>
#include <iomanip>

/// For printing data as ASCII with the maximum supported numerical precision
#define max_prec(T) \
    std::setprecision(std::numeric_limits<T>::digits10 + 1)

/// Codes dealing with operations on coordinate systems
namespace teca_coordinate_util
{
/** @brief
 *  traits classes used to get default tolerances for comparing numbers
 *  of a given precision.
 *
 *  @details
 *  A relative tolerance is used for comparing large
 *  numbers and an absolute tolerance is used for comparing small numbers.
 *  these defaults are not universal and will not work well in all situations.
 *
 *  see also:
 *  https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 */
template <typename n_t>
struct equal_tt {};

#define declare_equal_tt(cpp_t, atol, rtol)                                 \
/** Specialization for cpp_t with default absTol and relTol */              \
template <>                                                                 \
struct equal_tt<cpp_t>                                                      \
{                                                                           \
    static cpp_t absTol() { return atol; }                                  \
    static cpp_t relTol() { return rtol; }                                  \
};

declare_equal_tt(float, 10.0f*std::numeric_limits<float>::epsilon(),
    std::numeric_limits<float>::epsilon())

declare_equal_tt(double, 10.0*std::numeric_limits<double>::epsilon(),
    std::numeric_limits<float>::epsilon())

declare_equal_tt(long double, std::numeric_limits<double>::epsilon(),
    std::numeric_limits<double>::epsilon())

/** Compare two floating point numbers.  absTol handles comparing numbers very
 * close to zero.  relTol handles comparing larger values.
 */
template <typename T>
bool equal(T a, T b,
    T relTol = equal_tt<T>::relTol(), T absTol = equal_tt<T>::absTol(),
    typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
{
    // for numbers close to zero
    T diff = std::abs(a - b);
    if (diff <= absTol)
        return true;
    // realtive difference for larger values
    a = std::abs(a);
    b = std::abs(b);
    b = (b > a) ? b : a;
    b *= relTol;
    if (diff <= b)
        return true;
    return false;
}

/// Compare two integral numbers.
template <typename T>
bool equal(T a, T b, T relTol = 0, T absTol = 0,
    typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    (void)relTol;
    (void)absTol;
    return a == b;
}

/** Compare two floating point numbers. This overload may be used in regression
 * tests or other contexts where a diagnostic error message should be reported
 * if the numbers are not equal.
 */
template <typename T>
bool equal(T a, T b, std::string &diagnostic,
    T relTol = equal_tt<T>::relTol(), T absTol = equal_tt<T>::absTol(),
    typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
{
    // for numbers close to zero
    T diff = std::abs(a - b);
    if (diff <= absTol)
        return true;
    // realtive difference for larger values
    T aa = std::abs(a);
    T bb = std::abs(b);
    bb = (bb > aa) ? bb : aa;
    T cc = bb*relTol;
    if (diff <= cc)
        return true;
    // a and b are not equal format the diagnostic
    T eps = std::numeric_limits<T>::epsilon();
    std::ostringstream os;
    os  << max_prec(T) << a << " != " << max_prec(T) << b
        << " relTol=" << max_prec(T) << relTol
        << " absTol=" << max_prec(T) << absTol
        << " |a-b|=" << max_prec(T) << diff
        << " |a-b|/eps=" << max_prec(T) << diff/eps
        << " max(|a|,|b|)*relTol=" << max_prec(T) << cc
        << " |a-b|/max(|a|,|b|)=" << max_prec(T) << diff/bb
        << " eps(" << typeid(a).name() << sizeof(T) << ")="
        << max_prec(T) << eps;
    diagnostic = os.str();
    return false;
}

/** Compare two integral numbers. This overload may be used in regression
 * tests or other contexts where a diagnostic error message should be reported
 * if the numbers are not equal.
 */
template <typename T>
bool equal(T a, T b, std::string &diagnostic, T relTol = 0, T absTol = 0,
    typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    (void)relTol;
    (void)absTol;
    if (a == b)
        return true;
    // a and b are not equal format the diagnostic
    std::ostringstream os;
    os  << typeid(a).name() << sizeof(T) << " " << a << " != " << b;
    diagnostic = os.str();
    return false;
}


/// Less than or equal to predicate
template<typename data_t>
struct leq
{ static bool eval(const data_t &l, const data_t &r) { return l <= r; } };

/// Greater than or equal to predicate
template<typename data_t>
struct geq
{ static bool eval(const data_t &l, const data_t &r) { return l >= r; } };

/// Less than predicate
template<typename data_t>
struct lt
{ static bool eval(const data_t &l, const data_t &r) { return l < r; } };

/// Greater than predicate
template<typename data_t>
struct gt
{ static bool eval(const data_t &l, const data_t &r) { return l > r; } };

/// comparator implementing bracket for ascending input arrays
template<typename data_t>
struct ascend_bracket
{
    // m_0 is an index into the data, m_1 = m_0 + 1
    // comparitors defining the bracket orientation. for data in
    // ascending order:  val >= data[m_0] && val <= data[m_1]
    using comp0_t = geq<data_t>;
    using comp1_t = leq<data_t>;

    // m_0 is an index into the data, m_1 = m_0 + 1
    // get the id of the smaller value (lower == true)
    // or the larger value (lower == false)
    static unsigned long get_id(bool lower,
        unsigned long m_0, unsigned long m_1)
    {
        if (lower)
            return m_0;
        return m_1;
    }
};

/// comparator implementing bracket for descending input arrays
template<typename data_t>
struct descend_bracket
{
    // m_0 is an index into the data, m_1 = m_0 + 1
    // comparitors defining the bracket orientation. for data in
    // descending order:  val <= data[m_0] && val >= data[m_1]
    using comp0_t = leq<data_t>;
    using comp1_t = geq<data_t>;

    // m_0 is an index into the data, m_1 = m_0 + 1
    // get the id of the smaller value (lower == true)
    // or the larger value (lower == false)
    static unsigned long get_id(bool lower,
        unsigned long m_0, unsigned long m_1)
    {
        if (lower)
            return m_1;
        return m_0;
    }
};

/** binary search that will locate index bounding the value above or below
 * such that data[i] <= val or val <= data[i+1] depending on the value of
 * lower. return 0 if the value is found. the comp0 and comp1 template
 * parameters let us operate on both ascending and descending input. defaults
 * are set for ascending inputs.
 */
template <typename data_t, typename bracket_t = ascend_bracket<data_t>>
int index_of(const data_t *data, unsigned long l, unsigned long r,
    data_t val, bool lower, unsigned long &id)
{
    unsigned long m_0 = (r + l)/2;
    unsigned long m_1 = m_0 + 1;

    if (m_0 == r)
    {
        if (equal(val, data[m_0]))
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
        // found a bracket around the value
        if (equal(val, data[m_0]))
            id = m_0;
        else
        if (equal(val, data[m_1]))
            id = m_1;
        else
            id = bracket_t::get_id(lower, m_0, m_1);
        return 0;
    }
    else
    if (bracket_t::comp1_t::eval(val, data[m_0]))
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

/** binary search that will locate index of the given value. return 0 if the
 * value is found.
 */
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

/** Convert bounds to extents.  return non-zero if the requested bounds are
 * not in the given coordinate arrays. coordinate arrays must not be empty.
 */
int bounds_to_extent(const double *bounds,
    const const_p_teca_variant_array &x, const const_p_teca_variant_array &y,
    const const_p_teca_variant_array &z, unsigned long *extent);

int bounds_to_extent(const double *bounds,
    const const_p_teca_variant_array &x, unsigned long *extent);

int bounds_to_extent(const double *bounds, const teca_metadata &md,
    unsigned long *extent);

/** Get the i,j,k cell index of point x,y,z in the given mesh.  return 0 if
 * successful.
 */
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

/**  given a human readable date string in YYYY-MM-DD hh:mm:ss format and a
 * list of floating point offset times in the specified calendar and units find
 * the closest time step. return 0 if successful see index_of for a description
 * of lower, if clamp is true then when the date falls outside of the time
 * values either the first or last time step is returned.
 */
int time_step_of(const const_p_teca_variant_array &time,
    bool lower, bool clamp, const std::string &calendar,
    const std::string &units, const std::string &date,
    unsigned long &step);

/**  given a time value (val), associated time units (units), and calendar
 * (calendar), return a human-readable rendering of the date (date) in a
 * strftime-format (format).  return 0 if successful.
 */
int time_to_string(double val, const std::string &calendar,
    const std::string &units, const std::string &format, std::string &date);

/** build random access data structures for an indexed table.  the index column
 * gives each entity a unique id. the index is used to identify rows that
 * belong in the entity. it is assumed that an entity occupies consecutive rows.
 * the returns are: n_entities, the number of entities found; counts, the
 * number of rows used by each entity; offsets, the starting row of each
 * entity; ids, a new set of ids for the entities starting from 0
 */
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

/** 0th order (nearest neighbor) interpolation for nodal data on a stretched
 * Cartesian mesh. This overload implements the general 3D case.
 * cx, cy, cz is the location to interpolate to
 * p_x, p_y, p_z array arrays containing the source coordinates with extents
 * [0, ihi, 0, jhi, 0, khi]
 * p_data is the field to interpolate from
 * val is the result
 * returns 0 if successful, an error occurs if cx, cy, cz is outside of the
 * source coordinate system
 */
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

/** 0th order (nearest neighbor) interpolation for nodal data on a stretched
 * Cartesian mesh.  This overload implements the special case where both source
 * and target mesh data are in a 2D x-y plane using fewer operations than the
 * general 3D implementation.
 * cx, cy, cz is the location to interpolate to
 * p_x, p_y, p_z array arrays containing the source coordinates with extents
 * [0, ihi, 0, jhi, 0, khi]
 * p_data is the field to interpolate from
 * val is the result
 * returns 0 if successful, an error occurs if cx, cy, cz is outside of the
 * source coordinate system
 */
template<typename coord_t, typename data_t>
int interpolate_nearest(coord_t cx, coord_t cy, const coord_t *p_x,
    const coord_t *p_y, const data_t *p_data, unsigned long ihi,
    unsigned long jhi, unsigned long nx, data_t &val)
{
    // get i,j of node less than cx,cy
    unsigned long i = 0;
    unsigned long j = 0;

    if ((ihi && teca_coordinate_util::index_of(p_x, 0, ihi, cx, true, i))
        || (jhi && teca_coordinate_util::index_of(p_y, 0, jhi, cy, true, j)))
    {
        // cx,cy is outside the coordinate axes
        return -1;
    }

    // get i,j of node greater than cx,cy
    unsigned long ii = std::min(i + 1, ihi);
    unsigned long jj = std::min(j + 1, jhi);

    // get index of nearest node
    unsigned long p = (cx - p_x[i]) <= (p_x[ii] - cx) ? i : ii;
    unsigned long q = (cy - p_y[j]) <= (p_y[jj] - cy) ? j : jj;

    // assign value from nearest node
    val = p_data[p + nx*q];

    return 0;
}

/** 1st order (linear) interpolation for nodal data on stretched Cartesian
 * mesh. This overload implements the general 3D case.
 * cx, cy, cz is the location to interpolate to
 * p_x, p_y, p_z array arrays containing the source coordinates with extents
 * [0, ihi, 0, jhi, 0, khi]
 * p_data is the field to interpolate from
 * val is the result
 * returns 0 if successful, an error occurs if cx, cy, cz is outside of the
 * source coordinate system
 */
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

    // get i,j of node greater than cx,cy,cz
    unsigned long ii = std::min(i + 1, ihi);
    unsigned long jj = std::min(j + 1, jhi);
    unsigned long kk = std::min(k + 1, khi);

    // compute weights
    CT wx = ii == i ? 0 : (cx - p_x[i])/(p_x[ii] - p_x[i]);
    CT wy = jj == j ? 0 : (cy - p_y[j])/(p_y[jj] - p_y[j]);
    CT wz = kk == k ? 0 : (cz - p_z[k])/(p_z[kk] - p_z[k]);

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

/** 1st order (linear) interpolation for nodal data on stretched Cartesian mesh.
 * This overload implements the special case where both source and target data
 * are in a 2D x-y plane using fewer operations than the general 3D
 * implementation.
 * cx, cy, cz is the location to interpolate to
 * p_x, p_y, p_z array arrays containing the source coordinates with extents
 * [0, ihi, 0, jhi, 0, khi]
 * p_data is the field to interpolate from
 * val is the result
 * returns 0 if successful, an error occurs if cx, cy, cz is outside of the
 * source coordinate system
 */
template<typename CT, typename DT>
int interpolate_linear(CT cx, CT cy, const CT *p_x, const CT *p_y,
    const DT *p_data, unsigned long ihi, unsigned long jhi,
    unsigned long nx, DT &val)
{
    // get i,j of node less than cx,cy
    unsigned long i = 0;
    unsigned long j = 0;

    if ((ihi && teca_coordinate_util::index_of(p_x, 0, ihi, cx, true, i))
        || (jhi && teca_coordinate_util::index_of(p_y, 0, jhi, cy, true, j)))
    {
        // cx,cy is outside the coordinate axes
        return -1;
    }

    // get i,j of node greater than cx,cy
    unsigned long ii = std::min(i + 1, ihi);
    unsigned long jj = std::min(j + 1, jhi);

    // compute weights
    CT wx = ii == i ? 0 : (cx - p_x[i])/(p_x[ii] - p_x[i]);
    CT wy = jj == j ? 0 : (cy - p_y[j])/(p_y[jj] - p_y[j]);

    CT vx = CT(1) - wx;
    CT vy = CT(1) - wy;

    // interpolate
    val = vx*vy*p_data[ i +  j*nx]
        + wx*vy*p_data[ii +  j*nx]
        + wx*wy*p_data[ii + jj*nx]
        + vx*wy*p_data[ i + jj*nx];

    return 0;
}

/// A functor templated on order of accuracy for above Cartesian mesh interpolants
template<int> struct interpolate_t;

/// Zero'th order interpolant specialization
template<> struct interpolate_t<0>
{
    // 3D
    template<typename CT, typename DT>
    int operator()(CT tx, CT ty, CT tz, const CT *sx, const CT *sy,
        const CT *sz, const DT *sa, unsigned long ihi, unsigned long jhi,
        unsigned long khi, unsigned long nx, unsigned long nxy, DT &ta)
    {
        return teca_coordinate_util::interpolate_nearest(tx,ty,tz,
             sx,sy,sz,sa, ihi,jhi,khi, nx,nxy, ta);
    }

    // 2D x-y plane
    template<typename CT, typename DT>
    int operator()(CT tx, CT ty, const CT *sx, const CT *sy,
        const DT *sa, unsigned long ihi, unsigned long jhi,
        unsigned long nx, DT &ta)
    {
        return teca_coordinate_util::interpolate_nearest(tx,ty,
             sx,sy,sa, ihi,jhi, nx, ta);
    }
};

/// First order interpolant specialization
template<> struct interpolate_t<1>
{
    // 3D
    template<typename CT, typename DT>
    int operator()(CT tx, CT ty, CT tz, const CT *sx, const CT *sy,
        const CT *sz, const DT *sa, unsigned long ihi, unsigned long jhi,
        unsigned long khi, unsigned long nx,unsigned long nxy, DT &ta)
    {
        return teca_coordinate_util::interpolate_linear(tx,ty,tz,
             sx,sy,sz,sa, ihi,jhi,khi, nx,nxy, ta);
    }

    // 2D x-y plane
    template<typename CT, typename DT>
    int operator()(CT tx, CT ty, const CT *sx, const CT *sy,
        const DT *sa, unsigned long ihi, unsigned long jhi,
        unsigned long nx,  DT &ta)
    {
        return teca_coordinate_util::interpolate_linear(tx,ty,
             sx,sy,sa, ihi,jhi, nx, ta);
    }
};

/// return 0 if the centering is one of the values defined in teca_array_attributes
int validate_centering(int centering);

/// convert from a cell extent to a face, edge or point centered extent
template <typename num_t>
int convert_cell_extent(num_t *extent, int centering)
{
    switch (centering)
    {
        case teca_array_attributes::invalid_value:
            TECA_ERROR("detected invalid_value in centering")
            return -1;
            break;
        case teca_array_attributes::cell_centering:
            break;
        case teca_array_attributes::x_face_centering:
            extent[1] += 1;
            break;
        case teca_array_attributes::y_face_centering:
            extent[3] += 1;
            break;
        case teca_array_attributes::z_face_centering:
            extent[5] += 1;
            break;
        case teca_array_attributes::x_edge_centering:
            extent[3] += 1;
            extent[5] += 1;
            break;
        case teca_array_attributes::y_edge_centering:
            extent[1] += 1;
            extent[5] += 1;
            break;
        case teca_array_attributes::z_edge_centering:
            extent[1] += 1;
            extent[3] += 1;
            break;
        case teca_array_attributes::point_centering:
            extent[1] += 1;
            extent[3] += 1;
            extent[5] += 1;
            break;
        case teca_array_attributes::no_centering:
            break;
        default:
            TECA_ERROR("this centering is undefined " << centering)
            return -1;
    }
    return 0;
}

/** Given Cartesian mesh metadata extract whole_extent and bounds
 *  if bounds metadata is not already present then it is initialized
 *  from coordinate arrays. It's an error if whole_extent or coordinate
 *  arrays are not present. return zero if successful.
 */
int get_cartesian_mesh_extent(const teca_metadata &md,
    unsigned long *whole_extent, double *bounds);

/// get the mesh's bounds from the coordinate axis arrays
int get_cartesian_mesh_bounds(const const_p_teca_variant_array x,
    const const_p_teca_variant_array y, const const_p_teca_variant_array z,
    double *bounds);

/** Check that one Cartesian region covers the other coordinates must be in
 *  ascending order. assumes that both regions are specified in ascending order.
 */
template <typename num_t>
int covers_ascending(const num_t *whole, const num_t *part)
{
    if ((part[0] >= whole[0]) && (part[0] <= whole[1]) &&
        (part[1] >= whole[0]) && (part[1] <= whole[1]) &&
        (part[2] >= whole[2]) && (part[2] <= whole[3]) &&
        (part[3] >= whole[2]) && (part[3] <= whole[3]) &&
        (part[4] >= whole[4]) && (part[4] <= whole[5]) &&
        (part[5] >= whole[4]) && (part[5] <= whole[5]))
        return 1;
    return 0;
}

/** Check that one Cartesian region covers the other, taking into account the
 * order of the coordinates. assumes that the regions are specified in the same
 * orientation.
 */
template <typename num_t>
int covers(const num_t *whole, const num_t *part)
{
    bool x_ascend = whole[0] <= whole[1];
    bool y_ascend = whole[2] <= whole[3];
    bool z_ascend = whole[4] <= whole[5];
    if (((x_ascend &&
        (part[0] >= whole[0]) && (part[0] <= whole[1]) &&
        (part[1] >= whole[0]) && (part[1] <= whole[1])) ||
        (!x_ascend &&
        (part[0] <= whole[0]) && (part[0] >= whole[1]) &&
        (part[1] <= whole[0]) && (part[1] >= whole[1]))) &&
        ((y_ascend &&
        (part[2] >= whole[2]) && (part[2] <= whole[3]) &&
        (part[3] >= whole[2]) && (part[3] <= whole[3])) ||
        (!y_ascend &&
        (part[2] <= whole[2]) && (part[2] >= whole[3]) &&
        (part[3] <= whole[2]) && (part[3] >= whole[3]))) &&
        ((z_ascend &&
        (part[4] >= whole[4]) && (part[4] <= whole[5]) &&
        (part[5] >= whole[4]) && (part[5] <= whole[5])) ||
        (!z_ascend &&
        (part[4] <= whole[4]) && (part[4] >= whole[5]) &&
        (part[5] <= whole[4]) && (part[5] >= whole[5]))))
        return 1;
    return 0;
}

/** check that two Cartesian regions have the same orientation ie they are
 * either both specified in ascending or descending order.
 */
template <typename num_t>
int same_orientation(const num_t *whole, const num_t *part)
{
    if ((((whole[0] <= whole[1]) && (part[0] <= part[1])) ||
        ((whole[0] >= whole[1]) && (part[0] >= part[1]))) &&
        (((whole[2] <= whole[3]) && (part[2] <= part[3])) ||
        ((whole[2] >= whole[3]) && (part[2] >= part[3]))) &&
        (((whole[4] <= whole[5]) && (part[4] <= part[5])) ||
        ((whole[4] >= whole[5]) && (part[4] >= part[5]))))
        return 1;
    return 0;
}

/** where array dimensions specified by nx_max, ny_max, and nz_max are 1, and
 * the extent would be out of bounds, set the extent to [0, 0].  If verbose is
 * set, a warning is reported when the extent was clamped in one or more
 * directions. The return is non zero if any direction was clamped and 0
 * otherwise.
 */
int clamp_dimensions_of_one(unsigned long nx_max, unsigned long ny_max,
    unsigned long nz_max, unsigned long *extent, bool verbose);

/** Return 0 if the passed extent does not exceed array dimensions specified in
 * nx_max, ny_max, and nz_max.  If verbose is set, an error is reported via
 * TECA_ERROR when the extent would be out of bounds.
 */
int validate_extent(unsigned long nx_max, unsigned long ny_max,
    unsigned long nz_max, unsigned long *extent, bool verbose);

};
#endif
