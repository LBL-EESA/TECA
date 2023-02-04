#ifndef teca_geometry_h
#define teca_geometry_h

#include "teca_config.h"
#include "teca_binary_stream.h"

#if defined(TECA_HAS_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <hamr_cuda_malloc_async_allocator.h>
#else
#define __host__
#define __device__
#endif

#include <memory>
#include <limits>

/// @file

/// Codes dealing with computational geometry
namespace teca_geometry
{

/// tests if a point is Left|On|Right of an infinite line.
template<typename n_t>
TECA_EXPORT
__host__ __device__
bool left(n_t e0x, n_t e0y, n_t e1x, n_t e1y, n_t px, n_t py)
{
    // >0 for p left of the line through e0 and e1
    // =0 for p on the line
    // <0 for p right of the line
    return
    ((e1x - e0x)*(py - e0y) - (px -  e0x)*(e1y - e0y)) >= n_t();
}

/** Winding number test for a point in a polygon. The winding number is 0 when
 * the point is outside. The polygon is defined a series of x, y points in
 * counter clockwise order. Defining in clock wise order changes the sign of
 * the winding number. the first and last point of the polygon are required to
 * be the same.
 *
 * @param [in] px the x coordinate of the point.
 * @param [in] py the y coordinate of the point.
 * @param [in] vx the x coordinates of the polygon.
 * @param [in] vy the y coordinates of the polygon.
 * @param [in] nppts the number of points in the polygon.
 *
 */
template<typename n_t>
TECA_EXPORT
__host__ __device__
bool point_in_poly(n_t px, n_t py,
    const n_t *vx, const n_t *vy, unsigned long nppts)
{
    int wn = 0;
    // loop through all edges of the polygon
    unsigned long npptsm1 = nppts - 1;
    for (unsigned long i = 0; i < npptsm1; ++i)
    {
        // edge from vx[i], vy[i] to  vx[i+1], vy[i+1]
        if (vy[i] <= py)
        {
            // if upward crossing and px, py left of edge then
            // have a valid up intersect
            if ((vy[i+1] > py) &&
              (left(vx[i], vy[i], vx[i+1], vy[i+1], px, py))) ++wn;
        }
        else
        {
             // if downward crossing and px, py right of edge then
             // have a valid down intersect
            if ((vy[i+1] <= py) &&
              (!left(vx[i], vy[i], vx[i+1], vy[i+1], px, py))) --wn;
        }
    }
    return wn;
}

/** @breif  The vertices of a 2D polygon in clockwise order. The first
 * and last point are required to be the same.
 */
template <typename coord_t>
struct TECA_EXPORT polygon
{
    using element_type = coord_t;

    polygon() : vx(nullptr), vy(nullptr), n_verts(0) {}

    /// deep copy the vertices of the polygon.
    template <typename in_coord_t>
    void copy(const in_coord_t *in_vx,
        const in_coord_t *in_vy, unsigned long in_n_verts);

    /// transforms coordinates to be in [0 360 -90 90]
    void normalize_coordinates();

    /// returns true if a point is inside the polygon
    bool inside(coord_t px, coord_t py) const;

    /** compute an axis aligned bounding box around the polygon
     * with the layout [x0 x1 y0 y1].
     */
    void get_bounds(coord_t *bounds) const;

    /// serialize to the given stream
    void to_stream(teca_binary_stream &bs) const;

    /// deseriealize from the given stream
    int from_stream(teca_binary_stream &bs);

    std::shared_ptr<coord_t> vx;
    std::shared_ptr<coord_t> vy;
    unsigned long n_verts;
};


// --------------------------------------------------------------------------
template <typename coord_t>
template <typename in_coord_t>
void polygon<coord_t>::copy(const in_coord_t *in_vx,
    const in_coord_t *in_vy, unsigned long in_n_verts)
{
    unsigned long nbytes = in_n_verts*sizeof(coord_t);

    vx = std::shared_ptr<coord_t>((coord_t*)malloc(nbytes), free);
    memcpy(vx.get(), in_vx, nbytes);

    vy = std::shared_ptr<coord_t>((coord_t*)malloc(nbytes), free);
    memcpy(vy.get(), in_vy, nbytes);

    n_verts = in_n_verts;
}

// --------------------------------------------------------------------------
template <typename coord_t>
void polygon<coord_t>::normalize_coordinates()
{
    coord_t *pvx = vx.get();
    for (unsigned long i = 0; i < n_verts; ++i)
        pvx[i] = pvx[i] < coord_t(0) ? pvx[i] + coord_t(360) : pvx[i];
}

// --------------------------------------------------------------------------
template <typename coord_t>
bool polygon<coord_t>::inside(coord_t px, coord_t py) const
{
    return teca_geometry::point_in_poly(px, py, vx.get(), vy.get(), n_verts);
}

// --------------------------------------------------------------------------
template <typename coord_t>
void polygon<coord_t>::get_bounds(coord_t *bounds) const
{
    const coord_t *pvx = vx.get();
    const coord_t *pvy = vy.get();

    bounds[0] = std::numeric_limits<coord_t>::max();
    for (unsigned long i = 0; i < n_verts; ++i)
        bounds[0] = pvx[i] < bounds[0] ? pvx[i] : bounds[0];

    bounds[1] = std::numeric_limits<coord_t>::lowest();
    for (unsigned long i = 0; i < n_verts; ++i)
        bounds[1] = pvx[i] > bounds[1] ? pvx[i] : bounds[1];

    bounds[2] = std::numeric_limits<coord_t>::max();
    for (unsigned long i = 0; i < n_verts; ++i)
        bounds[2] = pvy[i] < bounds[2] ? pvy[i] : bounds[2];

    bounds[3] = std::numeric_limits<coord_t>::lowest();
    for (unsigned long i = 0; i < n_verts; ++i)
        bounds[3] = pvy[i] > bounds[3] ? pvy[i] : bounds[3];
}

// --------------------------------------------------------------------------
template <typename coord_t>
void polygon<coord_t>::to_stream(teca_binary_stream &bs) const
{
    bs.pack(n_verts);
    bs.pack(vx.get(), n_verts);
    bs.pack(vy.get(), n_verts);
}

// --------------------------------------------------------------------------
template <typename coord_t>
int polygon<coord_t>::from_stream(teca_binary_stream &bs)
{
    bs.unpack(n_verts);

    unsigned long nbytes = n_verts*sizeof(coord_t);
    vx = std::shared_ptr<coord_t>((coord_t*)malloc(nbytes), free);
    vy = std::shared_ptr<coord_t>((coord_t*)malloc(nbytes), free);

    bs.unpack(vx.get(), n_verts);
    bs.unpack(vy.get(), n_verts);

    return 0;
}

};
#endif
