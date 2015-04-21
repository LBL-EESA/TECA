#include "teca_ar_detect.h"

#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_table.h"

#include <iostream>
#include <sstream>

using std::ostream;
using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// a description of the atmospheric river
struct atmospheric_river
{
    atmospheric_river() :
        pe(false), length(0.0), width(0.0),
        end_top_lat(0.0), end_top_lon(0.0),
        end_bot_lat(0.0), end_bot_lon(0.0)
    {}

    bool pe;
    double length;
    double width;
    double end_top_lat;
    double end_top_lon;
    double end_bot_lat;
    double end_bot_lon;
};

ostream &operator<<(ostream &os, const atmospheric_river &ar)
{
    os << " type=" << (ar.pe ? "PE" : "AR")
        << " length=" << ar.length
        << " width=" << ar.width
        << " bounds=" << ar.end_bot_lon << ", " << ar.end_bot_lat << ", "
        << ar.end_top_lon << ", " << ar.end_top_lat;
    return os;
}


unsigned sauf(const unsigned nrow, const unsigned ncol, unsigned int *image);

bool ar_detect(
    p_teca_variant_array lat,
    p_teca_variant_array lon,
    const vector<unsigned> &labeled_data,
    unsigned long num_labels,
    double river_start_lat_low,
    double river_start_lon_low,
    double river_end_lat_low,
    double river_end_lon_low,
    double river_end_lat_high,
    double river_end_lon_high,
    double percent_in_mesh,
    double river_width,
    double river_length,
    atmospheric_river &ar);

// set locations in the output where the input array
// has values within the low high range.
template <typename T>
void threshold(
    T *input, unsigned int *output,
    size_t n_vals, T low, T high)
{
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;
}

// binary search that will locate index bounding the value
// above or below such that data[i] <= val or val <= data[i+1]
// depending on the value of lower. return 0 if the value is
// found.
template <typename T>
int bounding_index(T *data, size_t l, size_t r, T val, bool lower, unsigned long &id)
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
        return bounding_index(data, l, m_0, val, lower, id);
    }
    else
    {
        // split the range to the right
        return bounding_index(data, m_1, r, val, lower, id);
    }

    // not found
    return -1;
}














// --------------------------------------------------------------------------
teca_ar_detect::teca_ar_detect() :
    water_vapor_variable("prw"),
    low_water_vapor_threshold(20),
    high_water_vapor_threshold(75),
    search_lat_low(19.0),
    search_lon_low(180.0),
    search_lat_high(56.0),
    search_lon_high(250.0),
    river_start_lat_low(19.0),
    river_start_lon_low(180.0),
    river_end_lat_low(29.0),
    river_end_lon_low(233.0),
    river_end_lat_high(56.0),
    river_end_lon_high(238.0),
    percent_in_mesh(5.0),
    river_width(1250.0),
    river_length(2000.0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_ar_detect::~teca_ar_detect()
{}

// --------------------------------------------------------------------------
teca_metadata teca_ar_detect::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "teca_ar_detect::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata output_md(input_md[0]);

    // TODO -- output is a table (and dataset?)

    return output_md;
}

// --------------------------------------------------------------------------
int teca_ar_detect::get_active_extent(
    p_teca_variant_array lat,
    p_teca_variant_array lon,
    std::vector<unsigned long> &extent) const
{
    extent.resize(6, 0l);

    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        lat.get(),


        TT *a = dynamic_cast<TT*>(lat.get());
        if (bounding_index(a->get(), 0, a->size() - 1,
                static_cast<NT>(this->search_lat_low),
                true, extent[2])
            || bounding_index(a->get(), 0, a->size() - 1,
                static_cast<NT>(this->search_lat_high),
                false, extent[3]))
        {
            TECA_ERROR(
                << "invalid lat cutoff range "
                << this->search_lat_low << ", " << this->search_lat_high)
            return -1;
        }

        a = dynamic_cast<TT*>(lon.get());
        if (bounding_index(a->get(), 0, a->size() - 1,
                static_cast<NT>(this->search_lon_low),
                true, extent[0])
            || bounding_index(a->get(), 0, a->size() - 1,
                static_cast<NT>(this->search_lon_low),
                false, extent[1]))
        {
            TECA_ERROR(
                << "invalid lon_cutoff range "
                << this->search_lon_low << ", " << this->search_lon_low)
            return -1;
        }
        )

    extent[4] = extent[5] = 0l;

    return 0;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_ar_detect::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "teca_ar_detect::get_upstream_request" << endl;
#endif
    (void)port;

    vector<teca_metadata> up_reqs;

    teca_metadata md = input_md[0];

    // locate the extents of the user supplied region of
    // interest
    teca_metadata coords;
    if (md.get("coordinates", coords))
    {
        TECA_ERROR("metadata is missing \"coordinates\"")
        return up_reqs;
    }

    p_teca_variant_array lat;
    p_teca_variant_array lon;
    if (!(lat = coords.get("x")) || !(lon = coords.get("y")))
    {
        TECA_ERROR("metadata missing lat lon coordinates")
        return up_reqs;
    }

    vector<unsigned long> extent(6, 0l);
    if (this->get_active_extent(lat, lon, extent))
    {
        TECA_ERROR("failed to determine the active extent")
        return up_reqs;
    }

    // build the request
    vector<string> arrays;
    request.get("arrays", arrays);
    arrays.push_back(this->water_vapor_variable);

    teca_metadata up_req(request);
    up_req.insert("arrays", arrays);
    up_req.insert("extent", extent);

    up_reqs.push_back(up_req);
    return up_reqs;
}

// --------------------------------------------------------------------------
p_teca_dataset teca_ar_detect::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id() << "teca_ar_detect::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input dataset
    p_teca_cartesian_mesh mesh
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(input_data[0]);
    if (!mesh)
    {
        TECA_ERROR("invalid input. teca_ar_detect requires a teca_cartesian_mesh")
        return nullptr;
    }

    // get coordinate arrays
    p_teca_variant_array lat = mesh->get_y_coordinates();
    p_teca_variant_array lon = mesh->get_x_coordinates();

    if (!lon || !lat)
    {
        TECA_ERROR("invalid mesh. missing lat lon coordinates")
        return nullptr;
    }

    // get time step
    unsigned long time_step;
    mesh->get_time_step(time_step);

    // get date
    // TODO -- discuss with Suren et al
    // date is encoded in the sample data as follows
    p_teca_variant_array date
        = mesh->get_information_arrays()->get("date");

    if (!date)
    {
        TECA_ERROR("Dataset missing date variable")
        return nullptr;
    }

    long YYYY = 0;
    long MM = 0;
    long DD = 0;
    TEMPLATE_DISPATCH(
        teca_variant_array_impl,
        date.get(),

        NT d = dynamic_cast<TT*>(date.get())->get()[0];
        YYYY = static_cast<long>(d)/10000l;
        MM = (static_cast<long>(d) - YYYY*10000l)/100l;
        DD = static_cast<long>(d) - YYYY*10000l - MM*100l;
        )

    // get water vapor data
    vector<unsigned long> extent;
    mesh->get_extent(extent);

    unsigned long num_rows = extent[3] - extent[2] + 1;
    unsigned long num_cols = extent[1] - extent[0] + 1;
    unsigned long num_rc = num_rows*num_cols;

    p_teca_variant_array water_vapor
        = mesh->get_point_arrays()->get(this->water_vapor_variable);

    if (!water_vapor)
    {
        TECA_ERROR(
            << "Dataset missing water vapor variable \""
            << this->water_vapor_variable << "\"")
        return nullptr;
    }

    p_teca_table event = teca_table::New();
    event->declare_columns(
        "year", long(), "month", long(), "day", long(),
        "time_step", long(), "length", double(),
        "width", double(), "end_top_lat", double(),
        "end_top_lon", double(), "end_bot_lat", double(),
        "end_bot_lon", double(), "type", std::string());

    TEMPLATE_DISPATCH(
        teca_variant_array_impl,
        water_vapor.get(),

        NT *p_wv = dynamic_cast<TT*>(water_vapor.get())->get();

        // threshold
        vector<unsigned int> labels(num_rc, 0);

        threshold(p_wv, &labels[0], num_rc,
            static_cast<NT>(this->low_water_vapor_threshold),
            static_cast<NT>(this->high_water_vapor_threshold));

        // label
        int num_labels = sauf(num_rows, num_cols, &labels[0]);

        // detect ar
        atmospheric_river ar;
        if (num_labels
            && ar_detect(lat, lon, labels, num_labels,
            this->river_start_lat_low, this->river_start_lon_low,
            this->river_end_lat_low, this->river_end_lon_low,
            this->river_end_lat_high, this->river_end_lon_high,
            this->percent_in_mesh, this->river_width,
            this->river_length, ar))
        {
            event << YYYY << MM << DD << time_step
                << ar.length << ar.width
                << ar.end_top_lat << ar.end_top_lon
                << ar.end_bot_lat << ar.end_bot_lon
                << std::string(ar.pe ? "PE" : "AR");
        }
        )

    return event;
}





// Code borrowed from John Wu's sauf.cpp
// Find the minimal value starting @arg ind.
inline unsigned afind(const std::vector<unsigned>& equiv,
              const unsigned ind)
{
    unsigned ret = ind;
    while (equiv[ret] < ret)
    {
        ret = equiv[ret];
    }
    return ret;
}

// Set the values starting with @arg ind.
inline void aset(std::vector<unsigned>& equiv,
         const unsigned ind, const unsigned val)
{
    unsigned i = ind;
    while (equiv[i] < i)
    {
        unsigned j = equiv[i];
        equiv[i] = val;
        i = j;
    }
    equiv[i] = val;
}

/*
* Purpose:        Scan with Array-based Union-Find
* Return vals:    Number of connected components
* Description:    SAUF -- Scan with Array-based Union-Find.
* This is an implementation that follows the decision try to minimize
* number of neighbors visited and uses the array-based union-find
* algorithms to minimize work on the union-find data structure.  It works
* with each pixel/cell of the 2D binary image individually.
* The 2D binary image is passed to sauf as a unsigned*.  On input, the
* zero value is treated as the background, and non-zero is treated as
* object.  On successful completion of this function, the non-zero values
* in array image is replaced by its label.
* The return value is the number of components found.
*/
unsigned sauf(const unsigned nrow, const unsigned ncol, unsigned *image)
{
    const unsigned ncells = ncol * nrow;
    const unsigned ncp1 = ncol + 1;
    const unsigned ncm1 = ncol - 1;
    std::vector<unsigned int> equiv;    // equivalence array
    unsigned nextLabel = 1;

    equiv.reserve(ncol);
    equiv.push_back(0);

    // the first cell of the first line
    if (*image != 0)
    {
    *image = nextLabel;
    equiv.push_back(nextLabel);
    ++ nextLabel;
    }
    // first row of cells
    for (unsigned i = 1; i < ncol; ++ i)
    {
    if (image[i] != 0)
    {
        if (image[i-1] != 0)
        {
        image[i] = image[i-1];
        }
        else
        {
        equiv.push_back(nextLabel);
        image[i] = nextLabel;
        ++ nextLabel;
        }
    }
    }

    // scan the rest of lines, check neighbor b first
    for (unsigned j = ncol; j < ncells; j += ncol)
    {
    unsigned nc, nd, k, l;

    // the first point of the line has two neighbors, and the two
    // neighbors must have at most one label (recorded as nc)
    if (image[j] != 0)
    {
        if (image[j-ncm1] != 0)
        nc = image[j-ncm1];
        else if (image[j-ncol] != 0)
        nc = image[j-ncol];
        else
        nc = nextLabel;
        if (nc != nextLabel) { // use existing label
        nc = equiv[nc];
        image[j] = nc;
        }
        else { // need a new label
        equiv.push_back(nc);
        image[j] = nc;
        ++ nextLabel;
        }
    }

    // the rest of the line
    for (unsigned i = j+1; i < j+ncol; ++ i)
    {
        if (image[i] != 0) {
        if (image[i-ncol] != 0) {
            nc = image[i-ncol];
            l = afind(equiv, nc);
            aset(equiv, nc, l);
            image[i] = l;
        }
        else if (i-ncm1<j && image[i-ncm1] != 0) {
            nc = image[i-ncm1];

            if (image[i-1] != 0)
            nd = image[i-1];
            else if (image[i-ncp1] != 0)
            nd = image[i-ncp1];
            else
            nd = nextLabel;
            if (nd < nextLabel) {
            k = afind(equiv, nc);
            l = afind(equiv, nd);
            if (l <= k) {
                aset(equiv, nc, l);
                aset(equiv, nd, l);
            }
            else {
                l = k;
                aset(equiv, nc, k);
                aset(equiv, nd, k);
            }
            image[i] = l;
            }
            else {
            l = afind(equiv, nc);
            aset(equiv, nc, l);
            image[i] = l;
            }
        }
        else if (image[i-1] != 0) {
            nc = image[i-1];
            l = afind(equiv, nc);
            aset(equiv, nc, l);
            image[i] = l;
        }
        else if (image[i-ncp1] != 0) {
            nc = image[i-ncp1];
            l = afind(equiv, nc);
            aset(equiv, nc, l);
            image[i] = l;
        }
        else { // need a new label
            equiv.push_back(nextLabel);
            image[i] = nextLabel;
            ++ nextLabel;
        }
        }
    }
    } // for (unsigned j ...

    // phase II: re-number the labels to be consecutive
    nextLabel = 0;
    const unsigned nequiv = equiv.size();
    for (unsigned i = 0; i < nequiv;  ++ i) {
    if (equiv[i] < i) { // chase one more time
#if defined(_DEBUG) || defined(DEBUG)
        std::cout << i << " final " << equiv[i] << " ==> "
              << equiv[equiv[i]] << std::endl;
#endif
        equiv[i] = equiv[equiv[i]];
    }
    else { // change to the next smallest unused label
#if defined(_DEBUG) || defined(DEBUG)
        std::cout << i << " final " << equiv[i] << " ==> "
              << nextLabel << std::endl;
#endif
        equiv[i] = nextLabel;
        ++ nextLabel;
    }
    }

    if (nextLabel < nequiv) {// relabel all cells to their minimal labels
    for (unsigned i = 0; i < ncells; ++ i)
        image[i] = equiv[image[i]];
    }

#if defined(_DEBUG) || defined(DEBUG)
    std::cout << "sauf(" << nrow << ", " << ncol << ") assigned "
          << nextLabel-1 << " label" << (nextLabel>2?"s":"")
          << ", used " << nequiv << " provisional labels"
          << std::endl;
#endif
    return nextLabel-1;
}

// do any of the detected points meet the river start
// criteria. retrun true if so.
template<typename T>
bool river_start_criteria_lat(
    const vector<int> &boundary_r,
    const T *p_lat,
    T river_start_lat)
{
    unsigned long n = boundary_r.size();
    for (unsigned long q = 0; q < n; ++q)
    {
        if (p_lat[boundary_r[q]] >= river_start_lat)
            return true;
    }
    return false;
}

// do any of the detected points meet the river start
// criteria. retrun true if so.
template<typename T>
bool river_start_criteria_lon(
    const vector<int> &boundary_c,
    const T *p_lon,
    T river_start_lon)
{
    unsigned long n = boundary_c.size();
    for (unsigned long q = 0; q < n; ++q)
    {
        if (p_lon[boundary_c[q]] >= river_start_lon)
            return true;
    }
    return false;
}

// helper return true if the start criteria is
// met, and classifies the ar as PE if it starts
// in the bottom boundary.
template<typename T>
bool river_start_criteria(
    const vector<int> &boundary_r,
    const vector<int> &boundary_c,
    const T *p_lat,
    const T *p_lon,
    T start_lat,
    T start_lon,
    atmospheric_river &ar)
{
    return
         ((ar.pe = river_start_criteria_lat(boundary_r, p_lat, start_lat))
         || river_start_criteria_lon(boundary_c, p_lon, start_lon));
}

// do any of the detected points meet the river end
// criteria? (ie. does it hit the west coasts?) if so
// store a bounding box covering the river and return
// true.
template<typename T>
bool river_end_criteria(
    const vector<int> &boundary_r,
    const vector<int> &boundary_c,
    T *p_lat,
    T *p_lon,
    T river_end_lat_low,
    T river_end_lon_low,
    T river_end_lat_high,
    T river_end_lon_high,
    atmospheric_river &ar)
{
    bool end_criteria = false;

    vector<int> end_col_idx;

    unsigned int count = boundary_r.size();
    for (unsigned int i = 0; i < count; ++i)
    {
        // approximate land mask boundaries for the western coast of the US,
        T lon_val = p_lon[boundary_c[i]];
        if ((lon_val >= river_end_lon_low) && (lon_val <= river_end_lon_high))
            end_col_idx.push_back(i);
    }

    // look for rows that fall between lat boundaries
    T top_lat = T();
    T top_lon = T();
    T bot_lat = T();
    T bot_lon = T();

    bool top_touch = false;
    unsigned int end_col_count = end_col_idx.size();
    for (unsigned int i = 0; i < end_col_count; ++i)
    {
        // approximate land mask boundaries for the western coast of the US,
        T lat_val = p_lat[boundary_r[end_col_idx[i]]];
        if ((lat_val >= river_end_lat_low) && (lat_val <= river_end_lat_high))
        {
            T lon_val = p_lon[boundary_c[end_col_idx[i]]];
            end_criteria = true;
            if (!top_touch)
            {
                top_touch = true;
                top_lat = lat_val;
                top_lon = lon_val;
            }
            bot_lat = lat_val;
            bot_lon = lon_val;
        }
    }

    ar.end_top_lat = top_lat;
    ar.end_top_lon = top_lon;
    ar.end_bot_lat = bot_lat;
    ar.end_bot_lon = bot_lon;

    return end_criteria;
}

/*
* Calculate geodesic distance between two lat, long pairs
* CODE borrowed from: from http://www.geodatasource.com/developers/c
* from http://osiris.tuwien.ac.at/~wgarn/gis-gps/latlong.html
* from http://www.codeproject.com/KB/cpp/Distancecplusplus.aspx
*/
template<typename T>
T geodesic_distance(T lat1, T lon1, T lat2, T lon2)
{
    T R = T(6371.0);
    T deg_to_rad = T(M_PI/180.0);

    T dlat1 = lat1*deg_to_rad;
    T dlon1 = lon1*deg_to_rad;
    T dlat2 = lat2*deg_to_rad;
    T dlon2 = lon2*deg_to_rad;

    T dLon = dlon1 - dlon2;
    T dLat = dlat1 - dlat2;

    T sin_dLat_half_sq = sin(dLat/T(2.0));
    sin_dLat_half_sq *= sin_dLat_half_sq;

    T sin_dLon_half_sq = sin(dLon/T(2.0));
    sin_dLon_half_sq *= sin_dLon_half_sq;

    T aHarv = sin_dLat_half_sq
        + cos(dlat1)*cos(dlat2)*sin_dLon_half_sq;

    T cHarv = T(2.0)*atan2(sqrt(aHarv), sqrt(T(1.0) - aHarv));

    T distance = R*cHarv;

    return distance;
}

/*
* This function calculates the average geodesic width
* As each pixel represents certain area, the total area
* is the product of the number of pixels and the area of
* one pixel. The average width is: the total area divided
* by the medial axis length
* We are calculating the average width, since we are not
* determining where exactly to cut off the tropical region
* to calculate the real width of an atmospheric river
*/
template <typename T>
T avg_width(
    const vector<int> &boundary_r,
    const vector<int> &boundary_c,
    T ar_len,
    const T *p_lat,
    const T *p_lon)
{
    // length of cell in lat direction
    T lat_val[2] = {p_lat[boundary_r[0]], p_lat[boundary_r[0] + 1]};
    T lon_val[2] = {p_lon[boundary_c[0]], p_lon[boundary_c[0]]};
    T dim1 = geodesic_distance(lat_val[0], lon_val[0], lat_val[1], lon_val[1]);

    // length of cell in lon direction
    lat_val[1] = lat_val[0];
    lon_val[1] = p_lon[boundary_c[0] + 1];
    T dim2 = geodesic_distance(lat_val[0], lon_val[0], lat_val[1], lon_val[1]);

    // area
    T pixel_area = dim1*dim2;
    T total_area = pixel_area*boundary_r.size();

    // avg width
    T avg_width = total_area/ar_len;
    return avg_width;
}

/*
* Find the middle point between two pairs of lat and lon values
* http://stackoverflow.com/questions/4164830/geographic-midpoint-between-two-coordinates
*/
template<typename T>
void geodesic_midpoint(T lat1, T lon1, T lat2, T lon2, T &mid_lat, T &mid_lon)
{
    T deg_to_rad = T(M_PI/180.0);
    T dLon = (lon2 - lon1) * deg_to_rad;
    T dLat1 = lat1 * deg_to_rad;
    T dLat2 = lat2 * deg_to_rad;
    T dLon1 = lon1 * deg_to_rad;

    T Bx = cos(dLat2) * cos(dLon);
    T By = cos(dLat2) * sin(dLon);

    mid_lat = atan2(sin(dLat1)+sin(dLat2),
        sqrt((cos(dLat1)+Bx)*(cos(dLat1)+Bx)+By*By));

    mid_lon = dLon1 + atan2(By, (cos(dLat1)+Bx));

    T rad_to_deg = T(180.0/M_PI);
    mid_lat = mid_lat * rad_to_deg;
    mid_lon = mid_lon * rad_to_deg;
}

/*
* Find the length along the medial axis of a connected component
* Medial length is the sum of the distances between the medial
* points in the connected component
*/
template<typename T>
T medial_length(
    const vector<int> &boundary_r,
    const vector<int> &boundary_c,
    const T *p_lat,
    const T *p_lon)
{
    vector<int> jb_r1;
    vector<int> jb_c1;
    vector<int> jb_c2;

    int row_track = -1;

    unsigned int count = boundary_r.size();
    for (unsigned int i = 0; i < count; ++i)
    {
        if (row_track != boundary_r[i])
        {
            jb_r1.push_back(boundary_r[i]);
            jb_c1.push_back(boundary_c[i]);
            row_track = boundary_r[i];
        }
        jb_c2.push_back(boundary_c[i]);
    }

    T total_dist = T();

    unsigned int b_count = jb_r1.size() - 1;
    for (unsigned int i = 0; i < b_count; ++i)
    {
        T lat_val[2];
        T lon_val[2];

        lat_val[0] = p_lat[jb_r1[i]];
        lat_val[1] = p_lat[jb_r1[i]];

        lon_val[0] = p_lon[jb_c1[i]];
        lon_val[1] = p_lon[jb_c2[i]];

        T mid_lat1;
        T mid_lon1;
        geodesic_midpoint(
            lat_val[0], lon_val[0], lat_val[1], lon_val[1],
            mid_lat1, mid_lon1);

        lat_val[0] = p_lat[jb_r1[i+1]];
        lat_val[1] = p_lat[jb_r1[i+1]];

        lon_val[0] = p_lon[jb_c1[i+1]];
        lon_val[1] = p_lon[jb_c2[i+1]];

        T mid_lat2;
        T mid_lon2;
        geodesic_midpoint(
            lat_val[0], lon_val[0], lat_val[1], lon_val[1],
            mid_lat2, mid_lon2);

        total_dist
            += geodesic_distance(mid_lat1, mid_lon1, mid_lat2, mid_lon2);
    }

    return total_dist;
}


// helper return true if the geometric conditions
// on an ar are satisfied. also stores the length
// and width of the river.
template <typename T>
bool river_geometric_criteria(
    const vector<int> &boundary_r,
    const vector<int> &boundary_c,
    const T *p_lat,
    const T *p_lon,
    double river_length,
    double river_width,
    atmospheric_river &ar)
{
    ar.length = medial_length(boundary_r, boundary_c, p_lat, p_lon);

    ar.width = avg_width(boundary_r, boundary_c,
        static_cast<T>(ar.length), p_lat, p_lon);

    return (ar.length >= river_length) && (ar.width <= river_width);
}


/*
* The main function that checks whether an AR event exists in
* given sub-plane of data. This currently applies only to the Western
* coast of the USA. return true if an ar is found.
*/
bool ar_detect(
    p_teca_variant_array lat,
    p_teca_variant_array lon,
    const vector<unsigned> &labels,
    unsigned long num_labels,
    double river_start_lat,
    double river_start_lon,
    double river_end_lat_low,
    double river_end_lon_low,
    double river_end_lat_high,
    double river_end_lon_high,
    double percent_in_mesh,
    double river_width,
    double river_length,
    atmospheric_river &ar)
{
    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        lat.get(),

        NT *p_lat = dynamic_cast<TT*>(lat.get())->get();
        NT *p_lon = dynamic_cast<TT*>(lon.get())->get();

        NT start_lat = static_cast<NT>(river_start_lat);
        NT start_lon = static_cast<NT>(river_start_lon);
        NT end_lat_low = static_cast<NT>(river_end_lat_low);
        NT end_lon_low = static_cast<NT>(river_end_lon_low);
        NT end_lat_high = static_cast<NT>(river_end_lat_high);
        NT end_lon_high = static_cast<NT>(river_end_lon_high);

        // for all discrete connected component labels
        // verify if there exists an AR
        vector<int> boundary_r;
        vector<int> boundary_c;

        unsigned long num_rows = lat->size();
        unsigned long num_cols = lon->size();

        // # in PE is % of points in regious mesh
        unsigned long num_rc = num_rows*num_cols;
        unsigned long thr_count = num_rc*percent_in_mesh/100;

        for (unsigned int i = 1; i <= num_labels; ++i)
        {
            // get all the points of this connected component
            for (unsigned long r = 0, q = 0; r < num_rows; ++r)
            {
                for (unsigned long c = 0; c < num_cols; ++c, ++q)
                {
                    if (labels[q] == i)
                    {
                        boundary_r.push_back(r);
                        boundary_c.push_back(c);
                    }
                }
            }

            // check for ar criteria
            unsigned long count = boundary_r.size();
            if ((count > thr_count)
                && river_start_criteria(
                    boundary_r, boundary_c, p_lat, p_lon,
                    start_lat, start_lon, ar)
                && river_end_criteria(
                    boundary_r, boundary_c, p_lat, p_lon,
                    end_lat_low, end_lon_low,
                    end_lat_high, end_lon_high,
                    ar)
                && river_geometric_criteria(
                    boundary_r, boundary_c, p_lat, p_lon,
                    river_length, river_width, ar))
            {
                return true;
            }
        }
        )
    return false;
}
