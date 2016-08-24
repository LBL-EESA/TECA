#ifndef teca_geometry_h
#define teca_geometry_h

namespace teca_geometry
{
// tests if a point is Left|On|Right of an infinite line.
template<typename n_t>
bool left(n_t e0x, n_t e0y, n_t e1x, n_t e1y, n_t px, n_t py)
{
    // >0 for p left of the line through e0 and e1
    // =0 for p on the line
    // <0 for p right of the line
    return
    ((e1x - e0x)*(py - e0y) - (px -  e0x)*(e1y - e0y)) >= n_t();
}

// winding number test for a point in a polygon
// winding number is 0 when the point is outside.
template<typename n_t>
bool point_in_poly(n_t px, n_t py,
    n_t *vx, n_t *vy, unsigned long nppts)
{
    int wn = 0;
    // loop through all edges of the polygon
    unsigned long npptsm1 = nppts - 1;
    for (unsigned long i = 0; i < npptsm1; ++i)
    {   // edge from vx[i], vy[i] to  vx[i+1], vy[i+1]
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

};

#endif
