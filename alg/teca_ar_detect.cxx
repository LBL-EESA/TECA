#include "teca_ar_detect.h"

#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_table.h"
#include "teca_calendar.h"
#include "teca_coordinate_util.h"

#include <iostream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;

#define TECA_DEBUG 1
#if TECA_DEBUG > 0
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_programmable_algorithm.h"
int write_mesh(
    const const_p_teca_cartesian_mesh &mesh,
    const const_p_teca_variant_array &vapor,
    const const_p_teca_variant_array &thres,
    const const_p_teca_variant_array &ccomp,
    const const_p_teca_variant_array &lsmask,
    const std::string &base_name);
#endif

/*************** MYCODE *****************/

//Straight skeleton
#include<boost/shared_ptr.hpp>

//CGAL lib.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <boost/iterator/zip_iterator.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;

//Straight skeleton
#include<CGAL/Polygon_2.h>
#include<CGAL/create_straight_skeleton_2.h>

typedef CGAL::Polygon_2<K>           Polygon_2 ;
typedef CGAL::Straight_skeleton_2<K> Ss ;
typedef boost::shared_ptr<Ss> SsPtr ;

typedef CGAL::Delaunay_triangulation_2<K>  Delaunay_triangulation_2;
typedef Delaunay_triangulation_2::Edge_iterator Edge_iterator;
typedef Delaunay_triangulation_2::Face_handle FH;
typedef Delaunay_triangulation_2::Finite_faces_iterator FFI;
typedef Delaunay_triangulation_2::Edge DE;
typedef Delaunay_triangulation_2::Face DF;

typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned, K>    Vb;
typedef CGAL::Triangulation_data_structure_2<Vb>                    Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>                      Delaunay;
typedef Delaunay::Point                                             Point;
typedef Delaunay::Vertex_handle VH;
typedef Delaunay::Edge DEDGE;
typedef Delaunay::Face_handle FHN;

//Straight skeleton
template<class K>
void print_point ( CGAL::Point_2<K> const& p )
{
    std::cout << "(" << p.x() << "," << p.y() << ")" ;
}

void save_circumcenters_to_vtk_file(std::vector<Point_2> circumcenters_coordinates, unsigned long time_step);

template<class K>
void print_straight_skeleton( CGAL::Straight_skeleton_2<K> const& ss, unsigned long time_step )
{
    typedef CGAL::Straight_skeleton_2<K> Ss ;

    typedef typename Ss::Vertex_const_handle     Vertex_const_handle ;
    typedef typename Ss::Halfedge_const_handle   Halfedge_const_handle ;
    typedef typename Ss::Halfedge_const_iterator Halfedge_const_iterator ;

    Halfedge_const_handle null_halfedge ;
    Vertex_const_handle   null_vertex ;

    std::cout << "Straight skeleton with " << ss.size_of_vertices()
    << " vertices, " << ss.size_of_halfedges()
    << " halfedges and " << ss.size_of_faces()
    << " faces" << std::endl ;

    std::vector<Point_2> circumcenters_coordinates;
    std::vector<Point_2> circumcenters_coordinates1;

    for ( Halfedge_const_iterator i = ss.halfedges_begin(); i != ss.halfedges_end(); ++i )
    {
        print_point(i->opposite()->vertex()->point()) ;
        std::cout << "->" ;
        print_point(i->vertex()->point());
        std::cout << " " << ( i->is_bisector() ? "bisector" : "contour" ) << std::endl;

        circumcenters_coordinates.push_back(i->opposite()->vertex()->point());
        circumcenters_coordinates.push_back(i->vertex()->point());
    }

    save_circumcenters_to_vtk_file(circumcenters_coordinates, 1000);
    save_circumcenters_to_vtk_file(circumcenters_coordinates1, 10000);

}

void straight_skeleton_ar(std::vector<Point_2>& selected_data_points, unsigned long time_step)
{
    Polygon_2 poly ;
    /*
    for(int i=0; i<selected_data_points.size(); i++)
    {
        poly.push_back(selected_data_points[i]);
    }
    */
    for (std::vector<Point_2>::iterator it = selected_data_points.begin(), end = selected_data_points.end(); it != end; ++it)
    {
        poly.push_back((*it));
    }

    /*
    poly.push_back( Point(-1,-1) ) ;
    poly.push_back( Point(0,-12) ) ;
    poly.push_back( Point(1,-1) ) ;
    poly.push_back( Point(12,0) ) ;
    poly.push_back( Point(1,1) ) ;
    poly.push_back( Point(0,12) ) ;
    poly.push_back( Point(-1,1) ) ;
    poly.push_back( Point(-12,0) ) ;
    */

    // You can pass the polygon via an iterator pair
    SsPtr iss = CGAL::create_interior_straight_skeleton_2(poly.vertices_begin(), poly.vertices_end());
    // Or you can pass the polygon directly, as below.

    // To create an exterior straight skeleton you need to specify a maximum offset.
    //double lMaxOffset = 5 ;
    //SsPtr oss = CGAL::create_exterior_straight_skeleton_2(lMaxOffset, poly);

    print_straight_skeleton(*iss, time_step);
    //print_straight_skeleton(*oss);
}

//Structure that stores data points with labels.
struct Coordinate
{
    unsigned int x, y;
    unsigned int label;

    Coordinate(unsigned int param_x, unsigned int param_y, unsigned int label_xy) : x(param_x), y(param_y), label(label_xy) {}
};

//Find neigbours based on ghost zones.
template <typename T, typename T1>
    void find_neighbours(unsigned int *lext, unsigned int *oext,
                         unsigned int *igrid, unsigned int *ogrid,
                         unsigned int nx, unsigned int nxx,
                         const T *p_land_sea_mask, std::vector<Point_2>& selected_data_points,
                         std::vector<unsigned long>& labels_of_selected_dp,
                         const T1 *p_lon, const T1 *p_lat);

template <typename T, typename T1>
    void find_segmentation(const T *input, unsigned int *p_con_comp_skel,
                           T low, unsigned long num_rc, unsigned long num_cols,
                           const T1 *p_land_sea_mask);

//Functions for skeletonization algorithm.
template <typename T>
    void compute_skeleton_of_ar(const_p_teca_variant_array land_sea_mask, unsigned int *p_con_comp,
                                unsigned long num_rc, const T *input, T low, T high, unsigned long num_rows,
                                unsigned long num_cols, const_p_teca_variant_array lat,
                                const_p_teca_variant_array lon, unsigned long time_step,
                                T search_lat_low,
                                T search_lon_low,
                                T search_lat_high,
                                T search_lon_high);

std::pair<FHN,FHN> FacesN (DEDGE const& e);

template <typename T>
bool check_constraints(const Point_2& p0, const Point_2& p1,
                       T search_lat_low,
                       T search_lon_low,
                       T search_lat_high,
                       T search_lon_high);

void compute_voronoi_diagram(std::vector<Point_2> selected_data_points,
                             std::vector<unsigned long> labels_of_selected_dp, unsigned long time_step);

float compute_width(std::vector<float>& vec);

float compute_distance(Point_2 const& p0, Point_2 const& p1);

void print_array(unsigned int *p_con_comp_skel, unsigned long num_rc,
                 unsigned long num_rows, unsigned long num_cols);

//Save to VTK file format.
void save_circumcenters_to_vtk_file(std::vector<Point_2> circumcenters_coordinates, unsigned long time_step);

//Class that represents vertices in a graph.
class Node
{
    public:
        int index;
        Point pixel;
        float value;
        int parent = -1, component = -1; // default value for a non-existing node

        //Additional members.
        std::map<float, int> four_neighbors;

        Node (int i, float v, Point p) { index = i, value = v, pixel = p; }

        Node (int i, float v, Point p, std::map<float, int> m) { index = i, value = v, pixel = p, four_neighbors = m; }
};

//Class that represents edges in a graph.
class Edge
{
    public:
        float weight;
        // Indices in a given vector
        int endpoint0, endpoint1;

        Edge (float w, int e0X, int e0Y) { weight = w; endpoint0 = e0X; endpoint1 = e0Y; }
};

template <typename T, typename T1>
    void build_graph(std::vector<Node>& nodes, std::vector<Edge>const& edges,
                     unsigned long num_rows, unsigned long num_cols,
                     const T *input, const T1 *p_lon, const T1 *p_lat);

bool sort_edges(Edge const& e0, Edge const& e1);

class Component
{
    public:
        int index;
        unsigned long nb_nodes;
        int max_value;
        int min_value = -1;
        int sum_nodes;
        std::vector<int> inodes;
        std::vector<int> icomponents;

        bool region_A;
        bool region_B;

        bool is_tracked;

        int max_value_AB = -1;

        std::map<float, int> neighbors;

        Component(Node node)
        {
            index = node.component;
            nb_nodes = 1;
            max_value = node.value;
            sum_nodes = sum_nodes + node.value;
            region_A = false;
            region_B = false;
            is_tracked = false;

            neighbors = node.four_neighbors;
        }

        Component(Component &c0, Component &c1)
        {
            region_A = false;
            region_B = false;

            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            max_value = std::max(c0.max_value, c1.max_value);

            int i0 = c0.index;
            int i1 = c1.index;

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); //preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );
        }

        Component(Component &c0, Component &c1, int i)
        {
            region_A = false;
            region_B = false;

            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            max_value = std::max(c0.max_value, c1.max_value);

            index = i;

            int i0 = c0.index;
            int i1 = c1.index;

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); //preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );
        }

        //Merge two old components into a new component.
        Component(int i, float value, Component &c0, Component &c1)
        {
            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            sum_nodes = c0.sum_nodes + c1.sum_nodes;

            max_value = std::max(c0.max_value, c1.max_value);

            int i0 = c0.index;
            int i1 = c1.index;

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); //preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );

            //Checking merging two components for two regions constraint.
            if(c0.region_A || c1.region_A)
            {
                region_A = true;
            }

            if(c0.region_B || c1.region_B)
            {
                region_B = true;
            }

            //Check if both components (c0 and c1) touch both boxes.
            bool c0_both = c0.region_A && c0.region_B;
            bool c1_both = c1.region_A && c1.region_B;

            //If c0 and c1 touches both boxes.
            if(c0_both && c1_both)
            {
                if(c0.max_value_AB > c1.max_value_AB)
                {
                    index = c0.index;
                    max_value_AB = c0.max_value_AB;
                }
                else
                {
                    index = c1.index;
                    max_value_AB = c1.max_value_AB;
                }
            }
            //If c0 and c1 does not touch both boxes.
            else if(!c0_both && !c1_both)
            {
                if(c0.region_A && !c0.region_B && !c1.region_A && c1.region_B)
                {
                    max_value_AB = value; //current value
                    index = i;
                }
                else if(!c0.region_A && c0.region_B && c1.region_A && !c1.region_B)
                {
                    max_value_AB = value; //current value
                    index = i;
                }
                else
                {
                    index = i;
                }
            }
            //If c0 does not touch both boxes, but c1 touches both.
            else if (!c0_both && c1_both)
            {
                index = c1.index;
                max_value_AB = c1.max_value_AB;
            }
            //If c0 touches both boxes, but c0 does not touch both.
            else if (c0_both && !c1_both)
            {
                index = c0.index;
                max_value_AB = c0.max_value_AB;
            }



            /*
            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            sum_nodes = c0.sum_nodes + c1.sum_nodes;

            //Checking merging two components for two regions constraint.
            if(c0.region_A || c1.region_A)
            //if(c0.region_A && c1.region_A)
            {
                region_A = true;
            }

            if(c0.region_B || c1.region_B)
            //if(c0.region_B && c1.region_B)
            {
                region_B = true;
            }

            //If first and second component touches box A.
            if((c0.region_A && !(c0.region_B)) && (c1.region_A && !(c1.region_B)))
            {
                max_value = std::max(c0.max_value, c1.max_value); //deepest3 = max(deepest1, deepest2)

                if(c0.max_value_AB > c1.max_value_AB)
                {
                    index = c0.index;
                    max_value_AB = c0.max_value_AB; //deepestAB3
                }
                else
                {
                    index = c1.index;
                    max_value_AB = c1.max_value_AB; //deepestAB3
                }
            }
            //If first and second component touches box B.
            else if((!(c0.region_A) && c0.region_B) && (!(c1.region_A) && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value); //deepest3 = max(deepest1, deepest2)

                if(c0.max_value_AB > c1.max_value_AB)
                {
                    index = c0.index;
                    max_value_AB = c0.max_value_AB; //deepestAB3
                }
                else
                {
                    index = c1.index;
                    max_value_AB = c1.max_value_AB; //deepestAB3
                }
            }
            //If second component touches only box B and first component does not touch anything.
            else if(!(c0.region_A && c0.region_B) && (!(c1.region_A) && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c1.max_value_AB;

                index = c1.index;
            }
            //If first component touches only box B and second component does not touch anything.
            else if((!(c0.region_A) && c0.region_B) && !(c1.region_A && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c0.max_value_AB;

                index = c0.index;
            }
            //If second component touches only box A and first component does not touch anything.
            else if(!(c0.region_A && c0.region_B) && (c1.region_A && !(c1.region_B)))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c1.max_value_AB;

                index = c1.index;
            }
            //If first component touches only box A and second component does not touch anything.
            else if((c0.region_A && !(c0.region_B)) && !(c1.region_A && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c0.max_value_AB;

                index = c0.index;
            }
            //If second component touches both boxes and first one touches only box B.
            else if((!(c0.region_A) && c0.region_B) && (c1.region_A && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c1.max_value_AB;

                index = c1.index;
            }
            //If first component touches both boxes and second one touches only box B.
            else if((c0.region_A && c0.region_B) && (!(c1.region_A) && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c0.max_value_AB;

                index = c0.index;
            }
            //If second component touches both boxes and first one touches only box A.
            else if((c0.region_A && !(c0.region_B)) && (c1.region_A && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c1.max_value_AB;

                index = c1.index;
            }
            //If first component touches both boxes and second one touches only box A.
            else if((c0.region_A && c0.region_B) && (c1.region_A && !(c1.region_B)))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c0.max_value_AB;

                index = c0.index;
            }
            //If first component touches both boxes.
            else if((c0.region_A && c0.region_B) && !(c1.region_A && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value); //deepest3 = max(deepest1, deepest2)

                max_value_AB = c0.max_value_AB; //deepestAB3

                index = c0.index;
            }
            //If second component touches both boxes.
            else if(!(c0.region_A && c0.region_B) && (c1.region_A && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                max_value_AB = c1.max_value_AB;

                index = c1.index;
            }
            //If two components touch both boxes.
            else if((c0.region_A && c0.region_B) && (c1.region_A && c1.region_B))
            {
                max_value = std::max(c0.max_value, c1.max_value);

                //argmaxdeppestAB
                max_value_AB = std::max(c0.max_value_AB, c1.max_value_AB);

                if(max_value_AB == c0.max_value_AB)
                {
                    index = c0.index;
                }
                else
                {
                    index = c1.index;
                }
            }
            //If first component touches only box A and second component tochues only box B (they approach each other from two opposite sides).
            else if((c0.region_A && !(c0.region_B)) && (!(c1.region_A) && c1.region_B))
            {
                max_value_AB = value; //deepestAB3 = threshold value

                max_value = std::max(c0.max_value, c1.max_value);

                if(c0.max_value_AB > c1.max_value_AB)
                {
                    index = c0.index;
                }
                else
                {
                    index = c1.index;
                }
            }
            //If first component touches only box B and second component tochues only box A (they approach each other from two opposite sides).
            else if((!(c0.region_A) && c0.region_B) && (c1.region_A && !(c1.region_B)))
            {
                max_value_AB = value; //deepestAB3 = threshold value

                max_value = std::max(c0.max_value, c1.max_value);

                if(c0.max_value_AB > c1.max_value_AB)
                {
                    index = c0.index;
                }
                else
                {
                    index = c1.index;
                }
            }
            //If both components do not touch both boxes.
            else if(!(c0.region_A && c0.region_B) && !(c1.region_A && c1.region_B))
            {
                //index = i;

                max_value = std::max(c0.max_value, c1.max_value);

                //if(c0.max_value_AB > c1.max_value_AB)
                if(c0.max_value > c1.max_value)
                {
                    index = c0.index;
                }
                else
                {
                    index = c1.index;
                }
            }

            int i0 = c0.index;
            int i1 = c1.index;

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); // preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );
            */

            /*
            index = i;
            max_value = std::max(c0.max_value, c1.max_value);
            nb_nodes = c0.nb_nodes + c1.nb_nodes;
            sum_nodes = c0.sum_nodes + c1.sum_nodes;

            c0.min_value = value;
            c1.min_value = value;

            int i0 = c0.index;
            int i1 = c1.index;

            bool is_younger = (c0.max_value - c0.min_value > c1.max_value - c1.min_value);

            //If i0 is younger than i1.
            if(is_younger)
            {
                std::swap(i0, i1);
            }

            icomponents.push_back(i0);
            icomponents.push_back(i1);

            inodes.reserve( c0.inodes.size() + c1.inodes.size() ); // preallocate memory
            inodes.insert( inodes.end(), c0.inodes.begin(), c0.inodes.end() );
            inodes.insert( inodes.end(), c1.inodes.begin(), c1.inodes.end() );
            */
        }
};

void add_node(int inode, Node const& node, Component &comp);

int find_root(std::vector<Node>& nodes, int n0);

void finding_components(std::vector<Node> nodes, std::vector<Edge> edges, std::vector<Component>& components, unsigned long time_step, unsigned int *regionA, unsigned int *regionB);

//Functions for finding threshold parameter.
template <typename T>
    void union_find_alg(const T *input, unsigned long num_rc,
                        unsigned long num_rows, unsigned long num_cols, const_p_teca_variant_array lat,
                        const_p_teca_variant_array lon, unsigned long time_step, const_p_teca_variant_array land_sea_mask);

void save_triang_to_vtk_file(std::vector<Point_2> circumcenters_coordinates, unsigned long time_step);


/*************** MYCODE *****************/

// a description of the atmospheric river
struct atmospheric_river
{
    atmospheric_river() :
        pe(false), length(0.0),
        min_width(0.0), max_width(0.0),
        end_top_lat(0.0), end_top_lon(0.0),
        end_bot_lat(0.0), end_bot_lon(0.0)
    {}

    bool pe;
    double length;
    double min_width;
    double max_width;
    double end_top_lat;
    double end_top_lon;
    double end_bot_lat;
    double end_bot_lon;
};

std::ostream &operator<<(std::ostream &os, const atmospheric_river &ar)
{
    os << " type=" << (ar.pe ? "PE" : "AR")
        << " length=" << ar.length
        << " width=" << ar.min_width << ", " << ar.max_width
        << " bounds=" << ar.end_bot_lon << ", " << ar.end_bot_lat << ", "
        << ar.end_top_lon << ", " << ar.end_top_lat;
    return os;
}

unsigned sauf(const unsigned nrow, const unsigned ncol, unsigned int *image);

bool ar_detect(
    const_p_teca_variant_array lat,
    const_p_teca_variant_array lon,
    const_p_teca_variant_array land_sea_mask,
    p_teca_unsigned_int_array con_comp,
    unsigned long n_comp,
    double river_start_lat,
    double river_start_lon,
    double river_end_lat_low,
    double river_end_lon_low,
    double river_end_lat_high,
    double river_end_lon_high,
    double percent_in_mesh,
    double river_width,
    double river_length,
    double land_threshold_low,
    double land_threshold_high,
    atmospheric_river &ar);

// set locations in the output where the input array
// has values within the low high range.
template <typename T>
void threshold(
    const T *input, unsigned int *output,
    size_t n_vals, T low, T high)
{
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;
}

// --------------------------------------------------------------------------
teca_ar_detect::teca_ar_detect() :
    water_vapor_variable("prw"),
    land_sea_mask_variable(""),
    low_water_vapor_threshold(20),
    high_water_vapor_threshold(75),
    search_lat_low(19.0),
    search_lon_low(180.0),
    search_lat_high(56.0),
    search_lon_high(250.0),
    river_start_lat_low(18.0),
    river_start_lon_low(180.0),
    river_end_lat_low(29.0),
    river_end_lon_low(233.0),
    river_end_lat_high(56.0),
    river_end_lon_high(238.0),
    percent_in_mesh(5.0),
    river_width(1250.0),
    river_length(2000.0),
    land_threshold_low(1.0),
    land_threshold_high(std::numeric_limits<double>::max())
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_ar_detect::~teca_ar_detect()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_ar_detect::get_properties_description(
    const std::string &prefix, options_description &opts)
{
    options_description ard_opts("Options for "
        + (prefix.empty()?"teca_ar_detect":prefix));

    ard_opts.add_options()
        TECA_POPTS_GET(std::string, prefix, water_vapor_variable,
            "name of variable containing water vapor values")
        TECA_POPTS_GET(double, prefix, low_water_vapor_threshold,
            "low water vapor threshold")
        TECA_POPTS_GET(double, prefix, high_water_vapor_threshold,
            "high water vapor threshold")
        TECA_POPTS_GET(double, prefix, search_lat_low,
            "search space low latitude")
        TECA_POPTS_GET(double, prefix, search_lon_low,
            "search space low longitude")
        TECA_POPTS_GET(double, prefix, search_lat_high,
            "search space high latitude")
        TECA_POPTS_GET(double, prefix, search_lon_high,
            "search space high longitude")
        TECA_POPTS_GET(double, prefix, river_start_lat_low,
            "latitude used to classify as AR or PE")
        TECA_POPTS_GET(double, prefix, river_start_lon_low,
            "longitude used to classify as AR or PE")
        TECA_POPTS_GET(double, prefix, river_end_lat_low,
            "CA coastal region low latitude")
        TECA_POPTS_GET(double, prefix, river_end_lon_low,
            "CA coastal region low longitude")
        TECA_POPTS_GET(double, prefix, river_end_lat_high,
            "CA coastal region high latitude")
        TECA_POPTS_GET(double, prefix, river_end_lon_high,
            "CA coastal region high longitude")
        TECA_POPTS_GET(double, prefix, percent_in_mesh,
            "size of river in relation to search space area")
        TECA_POPTS_GET(double, prefix, river_width,
            "minimum river width")
        TECA_POPTS_GET(double, prefix, river_length,
            "minimum river length")
        TECA_POPTS_GET(std::string, prefix, land_sea_mask_variable,
            "name of variable containing land-sea mask values")
        TECA_POPTS_GET(double, prefix, land_threshold_low,
            "low land value")
        TECA_POPTS_GET(double, prefix, land_threshold_high,
            "high land value")
        ;

    opts.add(ard_opts);
}

// --------------------------------------------------------------------------
void teca_ar_detect::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, water_vapor_variable)
    TECA_POPTS_SET(opts, double, prefix, low_water_vapor_threshold)
    TECA_POPTS_SET(opts, double, prefix, high_water_vapor_threshold)
    TECA_POPTS_SET(opts, double, prefix, search_lat_low)
    TECA_POPTS_SET(opts, double, prefix, search_lon_low)
    TECA_POPTS_SET(opts, double, prefix, search_lat_high)
    TECA_POPTS_SET(opts, double, prefix, search_lon_high)
    TECA_POPTS_SET(opts, double, prefix, river_start_lat_low)
    TECA_POPTS_SET(opts, double, prefix, river_start_lon_low)
    TECA_POPTS_SET(opts, double, prefix, river_end_lat_low)
    TECA_POPTS_SET(opts, double, prefix, river_end_lon_low)
    TECA_POPTS_SET(opts, double, prefix, river_end_lat_high)
    TECA_POPTS_SET(opts, double, prefix, river_end_lon_high)
    TECA_POPTS_SET(opts, double, prefix, percent_in_mesh)
    TECA_POPTS_SET(opts, double, prefix, river_width)
    TECA_POPTS_SET(opts, double, prefix, river_length)
    TECA_POPTS_SET(opts, std::string, prefix, land_sea_mask_variable)
    TECA_POPTS_SET(opts, double, prefix, land_threshold_low)
    TECA_POPTS_SET(opts, double, prefix, land_threshold_high)
}

#endif

// --------------------------------------------------------------------------
teca_metadata teca_ar_detect::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id()
        << "teca_ar_detect::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata output_md(input_md[0]);
    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_ar_detect::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id()
        << "teca_ar_detect::get_upstream_request" << endl;
#endif
    (void)port;

    std::vector<teca_metadata> up_reqs;

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
    if (!(lat = coords.get("y")) || !(lon = coords.get("x")))
    {
        TECA_ERROR("metadata missing lat lon coordinates")
        return up_reqs;
    }

    std::vector<double> bounds = {this->search_lon_low,
        this->search_lon_high, this->search_lat_low,
        this->search_lat_high, 0.0, 0.0};

    // build the request
    std::vector<std::string> arrays;
    request.get("arrays", arrays);
    arrays.push_back(this->water_vapor_variable);
    if (!this->land_sea_mask_variable.empty())
        arrays.push_back(this->land_sea_mask_variable);

    teca_metadata up_req(request);
    up_req.insert("arrays", arrays);
    up_req.insert("bounds", bounds);

    up_reqs.push_back(up_req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_ar_detect::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#if TECA_DEBUG > 1
    cerr << teca_parallel_id() << "teca_ar_detect::execute";
    this->to_stream(cerr);
    cerr << endl;
#endif
    (void)port;
    (void)request;

    // get the input dataset
    const_p_teca_cartesian_mesh mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);
    if (!mesh)
    {
        TECA_ERROR("invalid input. teca_cartesian_mesh is required")
        return nullptr;
    }

    // get coordinate arrays
    const_p_teca_variant_array lat = mesh->get_y_coordinates();
    const_p_teca_variant_array lon = mesh->get_x_coordinates();

    if (!lon || !lat)
    {
        TECA_ERROR("invalid mesh. missing lat lon coordinates")
        return nullptr;
    }

    // get land sea mask
    const_p_teca_variant_array land_sea_mask;
    if (this->land_sea_mask_variable.empty() ||
        !(land_sea_mask = mesh->get_point_arrays()->get(this->land_sea_mask_variable)))
    {
        // input doesn't have it, generate a stand in such
        // that land fall criteria will evaluate true
        size_t n = lat->size()*lon->size();
        p_teca_double_array lsm = teca_double_array::New(n, this->land_threshold_low);
        land_sea_mask = lsm;
    }

    // get the mesh extents
    std::vector<unsigned long> extent;
    mesh->get_extent(extent);

    unsigned long num_rows = extent[3] - extent[2] + 1;
    unsigned long num_cols = extent[1] - extent[0] + 1;
    unsigned long num_rc = num_rows*num_cols;

    // get water vapor data
    const_p_teca_variant_array water_vapor
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
        "time", double(), "time_step", long(),
        "length", double(), "min width", double(),
        "max width", double(), "end_top_lat", double(),
        "end_top_lon", double(), "end_bot_lat", double(),
        "end_bot_lon", double(), "type", std::string());

    // get calendar
    std::string calendar;
    mesh->get_calendar(calendar);
    event->set_calendar(calendar);

    // get units
    std::string units;
    mesh->get_time_units(units);
    event->set_time_units(units);

    // get time step
    unsigned long time_step;
    mesh->get_time_step(time_step);

    // get offset of the current timestep
    double time = 0.0;
    mesh->get_time(time);

    TEMPLATE_DISPATCH(
        const teca_variant_array_impl,
        water_vapor.get(),

        const NT *p_wv = dynamic_cast<TT*>(water_vapor.get())->get();

        // threshold
        p_teca_unsigned_int_array con_comp
            = teca_unsigned_int_array::New(num_rc, 0);

        unsigned int *p_con_comp = con_comp->get();

        threshold(p_wv, p_con_comp, num_rc,
            static_cast<NT>(this->low_water_vapor_threshold),
            static_cast<NT>(this->high_water_vapor_threshold));

#if TECA_DEBUG > 0
        p_teca_variant_array thresh = con_comp->new_copy();
#endif

        // label
        int num_comp = sauf(num_rows, num_cols, p_con_comp);

        /*************** MYCODE *****************/

#if TECA_DEBUG > 0

        union_find_alg(p_wv, num_rc, num_rows, num_cols, lat, lon, time_step, land_sea_mask); //It works properly!

        /*compute_skeleton_of_ar(land_sea_mask, p_con_comp,
                               num_rc, p_wv,
                               static_cast<NT>(this->low_water_vapor_threshold),
                               static_cast<NT>(this->high_water_vapor_threshold),
                               num_rows, num_cols,
                               lat, lon, time_step,
                               static_cast<NT>(this->search_lat_low),
                               static_cast<NT>(this->search_lon_low),
                               static_cast<NT>(this->search_lat_high),
                               static_cast<NT>(this->search_lon_high));
         */

#endif
        /*************** MYCODE *****************/

#if TECA_DEBUG > 0
        write_mesh(mesh, water_vapor, thresh, con_comp,
            land_sea_mask, "ar_mesh_%t%.%e%");
#endif

        // detect ar
        atmospheric_river ar;
        if (num_comp &&
            ar_detect(lat, lon, land_sea_mask, con_comp, num_comp,
                this->river_start_lat_low, this->river_start_lon_low,
                this->river_end_lat_low, this->river_end_lon_low,
                this->river_end_lat_high, this->river_end_lon_high,
                this->percent_in_mesh, this->river_width,
                this->river_length, this->land_threshold_low,
                this->land_threshold_high, ar))
        {
#if TECA_DEBUG > 0
            cerr << teca_parallel_id() << " event detected " << time_step << endl;
#endif
            event << time << time_step
                << ar.length << ar.min_width << ar.max_width
                << ar.end_top_lat << ar.end_top_lon
                << ar.end_bot_lat << ar.end_bot_lon
                << std::string(ar.pe ? "PE" : "AR");
        }
        )

    return event;
}

// --------------------------------------------------------------------------
void teca_ar_detect::to_stream(std::ostream &os) const
{
    os << " water_vapor_variable=" << this->water_vapor_variable
        << " land_sea_mask_variable=" << this->land_sea_mask_variable
        << " low_water_vapor_threshold=" << this->low_water_vapor_threshold
        << " high_water_vapor_threshold=" << this->high_water_vapor_threshold
        << " river_start_lon_low=" << this->river_start_lon_low
        << " river_start_lat_low=" << this->river_start_lat_low
        << " river_end_lon_low=" << this->river_end_lon_low
        << " river_end_lat_low=" << this->river_end_lat_low
        << " river_end_lon_high=" << this->river_end_lon_high
        << " river_end_lat_high=" << this->river_end_lat_high
        << " percent_in_mesh=" << this->percent_in_mesh
        << " river_width=" << this->river_width
        << " river_length=" << this->river_length
        << " land_threshodl_low=" << this->land_threshold_low
        << " land_threshodl_high=" << this->land_threshold_high;
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
        std::cerr << i << " final " << equiv[i] << " ==> "
              << equiv[equiv[i]] << std::endl;
#endif
        equiv[i] = equiv[equiv[i]];
    }
    else { // change to the next smallest unused label
#if defined(_DEBUG) || defined(DEBUG)
        std::cerr << i << " final " << equiv[i] << " ==> "
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
    std::cerr << "sauf(" << nrow << ", " << ncol << ") assigned "
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
    const std::vector<int> &con_comp_r,
    const T *p_lat,
    T river_start_lat)
{
    unsigned long n = con_comp_r.size();
    for (unsigned long q = 0; q < n; ++q)
    {
        if (p_lat[con_comp_r[q]] >= river_start_lat)
            return true;
    }
    return false;
}

// do any of the detected points meet the river start
// criteria. retrun true if so.
template<typename T>
bool river_start_criteria_lon(
    const std::vector<int> &con_comp_c,
    const T *p_lon,
    T river_start_lon)
{
    unsigned long n = con_comp_c.size();
    for (unsigned long q = 0; q < n; ++q)
    {
        if (p_lon[con_comp_c[q]] >= river_start_lon)
            return true;
    }
    return false;
}

// helper return true if the start criteria is
// met, and classifies the ar as PE if it starts
// in the bottom boundary.
template<typename T>
bool river_start_criteria(
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    const T *p_lat,
    const T *p_lon,
    T start_lat,
    T start_lon,
    atmospheric_river &ar)
{
    return
         ((ar.pe = river_start_criteria_lat(con_comp_r, p_lat, start_lat))
         || river_start_criteria_lon(con_comp_c, p_lon, start_lon));
}

// do any of the detected points meet the river end
// criteria? (ie. does it hit the west coasts?) if so
// store a bounding box covering the river and return
// true.
template<typename T>
bool river_end_criteria(
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    const T *p_lat,
    const T *p_lon,
    T river_end_lat_low,
    T river_end_lon_low,
    T river_end_lat_high,
    T river_end_lon_high,
    atmospheric_river &ar)
{
    bool end_criteria = false;

    std::vector<int> end_col_idx;

    unsigned int count = con_comp_r.size();
    for (unsigned int i = 0; i < count; ++i)
    {
        // approximate land mask boundaries for the western coast of the US,
        T lon_val = p_lon[con_comp_c[i]];
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
        T lat_val = p_lat[con_comp_r[end_col_idx[i]]];
        if ((lat_val >= river_end_lat_low) && (lat_val <= river_end_lat_high))
        {
            T lon_val = p_lon[con_comp_c[end_col_idx[i]]];
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

    T R = T(6371.0);
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
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    T ar_len,
    const T *p_lat,
    const T *p_lon)
{
/*
    // TODO -- need bounds checking when doing things like
    // p_lat[con_comp_r[0] + 1]. also because it's potentially
    // a stretched cartesian mesh need to compute area of
    // individual cells

    // length of cell in lat direction
    T lat_val[2] = {p_lat[con_comp_r[0]], p_lat[con_comp_r[0] + 1]};
    T lon_val[2] = {p_lon[con_comp_c[0]], p_lon[con_comp_c[0]]};
    T dlat = geodesic_distance(lat_val[0], lon_val[0], lat_val[1], lon_val[1]);

    // length of cell in lon direction
    lat_val[1] = lat_val[0];
    lon_val[1] = p_lon[con_comp_c[0] + 1];
    T dlon = geodesic_distance(lat_val[0], lon_val[0], lat_val[1], lon_val[1]);
*/
    (void)con_comp_c;
    // compute area of the first cell in the input mesh
    // length of cell in lat direction
    T lat_val[2] = {p_lat[0], p_lat[1]};
    T lon_val[2] = {p_lon[0], p_lon[0]};
    T dlat = geodesic_distance(lat_val[0], lon_val[0], lat_val[1], lon_val[1]);

    // length of cell in lon direction
    lat_val[1] = lat_val[0];
    lon_val[1] = p_lon[1];
    T dlon = geodesic_distance(lat_val[0], lon_val[0], lat_val[1], lon_val[1]);

    // area
    T pixel_area = dlat*dlon;
    T total_area = pixel_area*con_comp_r.size();

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
    mid_lat *= rad_to_deg;
    mid_lon *= rad_to_deg;
}

/*
* Find the length along the medial axis of a connected component
* Medial length is the sum of the distances between the medial
* points in the connected component
*/
template<typename T>
T medial_length(
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    const T *p_lat,
    const T *p_lon)
{
    std::vector<int> jb_r1;
    std::vector<int> jb_c1;
    std::vector<int> jb_c2;

    long row_track = -1;

    unsigned long count = con_comp_r.size();
    for (unsigned long i = 0; i < count; ++i)
    {
        if (row_track != con_comp_r[i])
        {
            jb_r1.push_back(con_comp_r[i]);
            jb_c1.push_back(con_comp_c[i]);

            jb_c2.push_back(con_comp_c[i]);

            row_track = con_comp_r[i];
        }
        else
        {
            jb_c2.back() = con_comp_c[i];
        }
    }

    T total_dist = T();

    long b_count = jb_r1.size() - 1;
    for (long i = 0; i < b_count; ++i)
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

/*
// Suren's function
// helper return true if the geometric conditions
// on an ar are satisfied. also stores the length
// and width of the river.
template <typename T>
bool river_geometric_criteria(
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    const T *p_lat,
    const T *p_lon,
    double river_length,
    double river_width,
    atmospheric_river &ar)
{
    ar.length = medial_length(con_comp_r, con_comp_c, p_lat, p_lon);

    ar.width = avg_width(con_comp_r, con_comp_c,
        static_cast<T>(ar.length), p_lat, p_lon);

    return (ar.length >= river_length) && (ar.width <= river_width);
}
*/

// Junmin's function for height of a triangle
template<typename T>
T triangle_height(T base, T s1, T s2)
{
    // area from Heron's fomula
    T p = (base + s1 + s2)/T(2);
    T area = p*(p - base)*(p - s1)*(p - s2);
    // detect impossible triangle
    if (area < T())
        return std::min(s1, s2);
    // height from A = 1/2 b h
    return T(2)*sqrt(area)/base;
}

// TDataProcessor::check_geodesic_width_top_down
// Junmin's function for detecting river based on
// it's geometric properties
template <typename T>
bool river_geometric_criteria(
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    const T *p_lat,
    const T *p_lon,
    double river_length,
    double river_width,
    atmospheric_river &ar)
{
    std::vector<int> distinct_rows;
    std::vector<int> leftmost_col;
    std::vector<int> rightmost_col;

    int row_track = -1;
    size_t count = con_comp_r.size();
    for (size_t i = 0; i < count; ++i)
    {
        if (row_track != con_comp_r[i])
        {
            row_track = con_comp_r[i];

            distinct_rows.push_back(con_comp_r[i]);
            leftmost_col.push_back(con_comp_c[i]);
            rightmost_col.push_back(con_comp_c[i]);
        }
        else
        {
            rightmost_col.back() = con_comp_c[i];
        }
    }

    // river metrics
    T length_from_top = T();
    T min_width = std::numeric_limits<T>::max();
    T max_width = std::numeric_limits<T>::lowest();

    for (long i = distinct_rows.size() - 2; i >= 0; --i)
    {
        // for each row w respect to row above it. triangulate
        // a quadrilateral composed of left and right most points
        // in this and the above rows. ccw ordering from lower
        // left corner is A,B,D,C.

        // low left-right distance
        T AB = geodesic_distance(
            p_lat[distinct_rows[i]], p_lon[leftmost_col[i]],
            p_lat[distinct_rows[i]], p_lon[rightmost_col[i]]);

        // left side bottom-top distance
        T AC = geodesic_distance(
            p_lat[distinct_rows[i]], p_lon[leftmost_col[i]],
            p_lat[distinct_rows[i+1]], p_lon[leftmost_col[i]]);

        // distance from top left to bottom right, across
        T BC = geodesic_distance(
            p_lat[distinct_rows[i]], p_lon[rightmost_col[i]],
            p_lat[distinct_rows[i+1]], p_lon[leftmost_col[i+1]]);

        // high left-right distance
        T CD = geodesic_distance(
            p_lat[distinct_rows[i+1]], p_lon[leftmost_col[i+1]],
            p_lat[distinct_rows[i+1]], p_lon[rightmost_col[i+1]]);

        // right side bottom-top distance
        T BD = geodesic_distance(
            p_lat[distinct_rows[i]], p_lon[rightmost_col[i]],
            p_lat[distinct_rows[i+1]], p_lon[rightmost_col[i+1]]);

        T height_from_b = triangle_height(AC, AB, BC);
        T height_from_c = triangle_height(BD, BC, CD);

        T curr_min = std::min(height_from_b, height_from_c);

        // test width criteria
        if (curr_min > river_width)
        {
            // TODO -- first time through the loop length == 0. is it intentional
            // to discard the detection or should length calc take place before this test?
            // note: first time through loop none of the event details have been recoreded
            if (length_from_top <= river_length)
            {
                 // too short to be a river
                return false;
            }
            else
            {
                 // part of a connected region is AR
                ar.min_width = static_cast<double>(min_width);
                ar.max_width = static_cast<double>(max_width);
                ar.length = static_cast<double>(length_from_top);
                return true;
            }
        }

        // update width
        min_width = std::min(min_width, curr_min);
        max_width = std::max(max_width, curr_min);

        // update length
        T mid_bot_lat;
        T mid_bot_lon;
        geodesic_midpoint(
            p_lat[distinct_rows[i]], p_lon[leftmost_col[i]],
            p_lat[distinct_rows[i]], p_lon[rightmost_col[i]],
            mid_bot_lat, mid_bot_lon);

        T mid_top_lat;
        T mid_top_lon;
        geodesic_midpoint(
            p_lat[distinct_rows[i+1]], p_lon[leftmost_col[i+1]],
            p_lat[distinct_rows[i+1]], p_lon[rightmost_col[i+1]],
            mid_top_lat, mid_top_lon);

        length_from_top += geodesic_distance(
            mid_bot_lat, mid_bot_lon, mid_top_lat, mid_top_lon);
    }

    // check the length criteria.
    // TODO: if we are here the widtrh critera was not met
    // so the following detection is based solely on the length?
    if (length_from_top > river_length)
    {
        // AR
        ar.min_width = static_cast<double>(min_width);
        ar.max_width = static_cast<double>(max_width);
        ar.length = static_cast<double>(length_from_top);
        return true;
    }

    return false;
}



// Junmin's function checkRightBoundary
// note: if land sea mask is not available land array
// must all be true.
template<typename T>
bool river_end_criteria(
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    const std::vector<bool> &land,
    const T *p_lat,
    const T *p_lon,
    T river_end_lat_low,
    T river_end_lon_low,
    T river_end_lat_high,
    T river_end_lon_high,
    atmospheric_river &ar)
{
    // locate component points within shoreline
    // box
    bool first_crossing = false;
    bool event_detected = false;

    T top_lat = T();
    T top_lon = T();
    T bot_lat = T();
    T bot_lon = T();

    std::vector<int> right_bound_col_idx;
    size_t count = con_comp_c.size();
    for (size_t i = 0; i < count; ++i)
    {
        T lat = p_lat[con_comp_r[i]];
        T lon = p_lon[con_comp_c[i]];

        if ((lat >= river_end_lat_low) && (lat <= river_end_lat_high)
            && (lon >= river_end_lon_low) && (lon <= river_end_lon_high))
        {
            if (!event_detected)
                event_detected = land[i];

            if (!first_crossing)
            {
                first_crossing = true;
                top_lat = lat;
                top_lon = lon;
            }
            bot_lat = lat;
            bot_lon = lon;
        }
    }

    ar.end_top_lat = top_lat;
    ar.end_top_lon = top_lon;
    ar.end_bot_lat = bot_lat;
    ar.end_bot_lon = bot_lon;

    return event_detected;
}

// Junmin's function
template<typename T>
void classify_event(
    const std::vector<int> &con_comp_r,
    const std::vector<int> &con_comp_c,
    const T *p_lat,
    const T *p_lon,
    T start_lat,
    T start_lon,
    atmospheric_river &ar)
{
    // classification determined by first detected point in event
    // is closer to left or to bottom
    T lat = p_lat[con_comp_r[0]];
    T lon = p_lon[con_comp_c[0]];

    ar.pe = false;
    if ((lat - start_lat) < (lon - start_lon))
        ar.pe = true; // PE
}

/*
* The main function that checks whether an AR event exists in
* given sub-plane of data. This currently applies only to the Western
* coast of the USA. return true if an ar is found.
*/
bool ar_detect(
    const_p_teca_variant_array lat,
    const_p_teca_variant_array lon,
    const_p_teca_variant_array land_sea_mask,
    p_teca_unsigned_int_array con_comp,
    unsigned long n_comp,
    double river_start_lat,
    double river_start_lon,
    double river_end_lat_low,
    double river_end_lon_low,
    double river_end_lat_high,
    double river_end_lon_high,
    double percent_in_mesh,
    double river_width,
    double river_length,
    double land_threshold_low,
    double land_threshold_high,
    atmospheric_river &ar)
{
    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        lat.get(),
        1,

        NESTED_TEMPLATE_DISPATCH(
            const teca_variant_array_impl,
            land_sea_mask.get(),
            2,

            const NT1 *p_lat = dynamic_cast<TT1*>(lat.get())->get();
            const NT1 *p_lon = dynamic_cast<TT1*>(lon.get())->get();

            const NT2 *p_land_sea_mask
                = dynamic_cast<TT2*>(land_sea_mask.get())->get();

            NT1 start_lat = static_cast<NT1>(river_start_lat);
            NT1 start_lon = static_cast<NT1>(river_start_lon);
            NT1 end_lat_low = static_cast<NT1>(river_end_lat_low);
            NT1 end_lon_low = static_cast<NT1>(river_end_lon_low);
            NT1 end_lat_high = static_cast<NT1>(river_end_lat_high);
            NT1 end_lon_high = static_cast<NT1>(river_end_lon_high);

            unsigned long num_rows = lat->size();
            unsigned long num_cols = lon->size();

            // # in PE is % of points in regious mesh
            unsigned long num_rc = num_rows*num_cols;
            unsigned long thr_count = num_rc*percent_in_mesh/100.0;

            unsigned int *p_labels = con_comp->get();
            for (unsigned int i = 1; i <= n_comp; ++i)
            {
                // for all discrete connected component labels
                // verify if there exists an AR
                std::vector<int> con_comp_r;
                std::vector<int> con_comp_c;
                std::vector<bool> land;

                for (unsigned long r = 0, q = 0; r < num_rows; ++r)
                {
                    for (unsigned long c = 0; c < num_cols; ++c, ++q)
                    {
                        if (p_labels[q] == i)
                        {
                            // gather points of this connected component
                            con_comp_r.push_back(r);
                            con_comp_c.push_back(c);

                            // identify them as land or not
                            land.push_back(
                                (p_land_sea_mask[q] >= land_threshold_low)
                                && (p_land_sea_mask[q] < land_threshold_high));

                        }
                    }
                }

                // check for ar criteria
                unsigned long count = con_comp_r.size();
                if ((count > thr_count)
                    && river_end_criteria(
                        con_comp_r, con_comp_c, land,
                        p_lat, p_lon,
                        end_lat_low, end_lon_low,
                        end_lat_high, end_lon_high,
                        ar)
                    && river_geometric_criteria(
                        con_comp_r, con_comp_c, p_lat, p_lon,
                        river_length, river_width, ar))
                {
                    // determine if PE or AR
                    classify_event(
                        con_comp_r, con_comp_c, p_lat, p_lon,
                        start_lat, start_lon, ar);
                    return true;
                }
            }


            )
        )

    return false;
}

#if TECA_DEBUG > 0
// helper to dump a dataset for debugging
int write_mesh(
    const const_p_teca_cartesian_mesh &mesh,
    const const_p_teca_variant_array &vapor,
    const const_p_teca_variant_array &thresh,
    const const_p_teca_variant_array &ccomp,
    const const_p_teca_variant_array &lsmask,
    const std::string &file_name)
{
    p_teca_cartesian_mesh m = teca_cartesian_mesh::New();
    m->copy_metadata(mesh);

    p_teca_array_collection pac = m->get_point_arrays();
    pac->append("vapor", std::const_pointer_cast<teca_variant_array>(vapor));
    pac->append("thresh", std::const_pointer_cast<teca_variant_array>(thresh));
    pac->append("ccomp", std::const_pointer_cast<teca_variant_array>(ccomp));
    pac->append("lsmask", std::const_pointer_cast<teca_variant_array>(lsmask));

    p_teca_programmable_algorithm s = teca_programmable_algorithm::New();
    s->set_number_of_input_connections(0);
    s->set_number_of_output_ports(1);
    s->set_execute_callback(
        [m] (unsigned int, const std::vector<const_p_teca_dataset> &,
        const teca_metadata &) -> const_p_teca_dataset { return m; }
        );

    p_teca_vtk_cartesian_mesh_writer w
        = teca_vtk_cartesian_mesh_writer::New();

    w->set_file_name(file_name);
    w->set_input_connection(s->get_output_port());
    w->update();

    return 0;
}

#endif


/********************************* MY CODE *****************************/

//Add new node to the componenent.
void add_node(int inode, Node const& node, Component &comp)
{
    comp.nb_nodes++;
    comp.sum_nodes = comp.sum_nodes + node.value;
    comp.inodes.push_back(inode);
    //comp.icomponents.push_back(node.component);

    //std::cerr << node.component << endl;
}

bool sort_volumes(Component const& c0, Component const& c1)
{
    if(c0.nb_nodes > c1.nb_nodes)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool sort_volumes_values(Component const& c0, Component const& c1)
{
    if(c0.sum_nodes > c1.sum_nodes)
    {
        return true;
    }
    else
    {
        return false;
    }
}

class Ratio
{
public:
    float r_ratio;
    int r_threshold_value;

    Ratio(float ratio, int threshold_value)
    {
        r_ratio = ratio;
        r_threshold_value = threshold_value;
    }
};

class Volume
{
public:
    int v_nb_nodes;
    int v_threshold_value;
    int v_id;
    std::vector<int> v_icomponents;

    Volume(int nb_nodes, int threshold_value)
    {
        v_nb_nodes = nb_nodes;
        v_threshold_value = threshold_value;
    }

    Volume(int nb_nodes, int threshold_value, int id)
    {
        v_nb_nodes = nb_nodes;
        v_threshold_value = threshold_value;
        v_id = id;
    }

    Volume(int nb_nodes, int threshold_value, int id, std::vector<int> icomponents)
    {
        v_nb_nodes = nb_nodes;
        v_threshold_value = threshold_value;
        v_id = id;
        v_icomponents = icomponents;
    }
};

bool sort_increasing_order(Ratio const& c0, Ratio const& c1)
{
    if(c0.r_threshold_value < c1.r_threshold_value)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void save_ratio_to_file(std::vector<Ratio> ratios, unsigned long time_step)
{
    //sort in decreasing order
    std::sort( ratios.begin(), ratios.end(), sort_increasing_order );

    std::ostringstream oss;
    oss << "ar_ratios_" << std::setw(6) << std::setfill('0') << time_step << ".txt";

    std::string file_name = oss.str();

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    for (std::vector<Ratio>::iterator it = ratios.begin(), end = ratios.end(); it != end; ++it)
    {
        ofs << (*it).r_threshold_value << "," << (*it).r_ratio << endl;
        //ofs << (*it).r_ratio << "," << (*it).r_max_value << endl;
    }

    ofs.close();
}

void save_volume_to_file(std::vector<Volume> volumes, unsigned long time_step)
{
    std::ostringstream oss;
    oss << "ar_volumes_" << std::setw(6) << std::setfill('0') << time_step << ".txt";

    std::string file_name = oss.str();

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    for (std::vector<Volume>::iterator it = volumes.begin(), end = volumes.end(); it != end; ++it)
    {
        //ofs << (*it).v_threshold_value << "," << (*it).v_nb_nodes << "," << (*it).v_id << endl;
        ofs << (*it).v_threshold_value << "," << (*it).v_nb_nodes << "," << (*it).v_id;

        //std::cerr << (*it).v_icomponents[0] << "," << (*it).v_icomponents[1] << endl;

        if((*it).v_icomponents.size() > 0)
        {
            ofs << "," << (*it).v_icomponents[0] << "," << (*it).v_icomponents[1] << endl;
        }
        else
        {
            ofs << "," << -1 << "," << -1 << endl;
        }
    }

    ofs.close();
}

//The volume plots for the top 4-5 connected components in one figure
void track_several_components(std::vector<Component> components, std::vector<Volume>& volumes, int val)
{
    /*
    int n = 4;
    int vol = 0;

    if(components.size() > 0)
    {
        //Tracking volume
        std::vector<Component> components_tmp1;
        components_tmp1 = components;

        std::sort( components_tmp1.begin(), components_tmp1.end(), sort_volumes );

        if(volumes.size() >= n)
        //if(volumes.size() > n)
        {
            for(int i = 0; i < n; i++)
            {
                vol = components_tmp1[i].nb_nodes;

                //Volume vtmp = volumes.back() - i;
                Volume vtmp = volumes.at((volumes.size() - 1) - i);
                //std::cerr << (volumes.size() - 1) - i << endl;

                if(vtmp.v_threshold_value != val)
                {
                    volumes.push_back( Volume(vol, val) );
                }
                else
                {
                    //If ratio is greater than exisiting ratio for given threshold.
                    if(vol > vtmp.v_nb_nodes)
                    {
                        volumes.pop_back();
                        volumes.push_back( Volume(vol, val) );
                    }
                }
            }
        }
        else
            volumes.push_back( Volume(vol, val) );
    }
    */

    //int n = 4;
    int vol = 0;
    int ind = 0;

    if(components.size() > 0)
    {
        //Tracking volume
        std::vector<Component> components_tmp1;
        components_tmp1 = components;

        //std::sort( components_tmp1.begin(), components_tmp1.end(), sort_volumes );

        for (std::vector<Component>::iterator it = components_tmp1.begin(), end = components_tmp1.end(); it != end; ++it)
        {
            vol = (*it).nb_nodes;
            ind = (*it).index;

            volumes.push_back( Volume(vol, val, ind) );
        }
    }
}

//Tracking size of component (1st largest) and ratio.
void track_size_of_components(std::vector<Component> components, std::vector<Volume>& volumes, std::vector<Ratio>& ratios, int val)
{
    if(components.size() > 0)
    {
        //Tracking volume
        std::vector<Component> components_tmp1;
        components_tmp1 = components;

        std::sort( components_tmp1.begin(), components_tmp1.end(), sort_volumes );

        //we want to track all of components not only the biggest one !!!
        int vol = components_tmp1[0].nb_nodes;

        if(volumes.size() > 0)
        {
            Volume vtmp = volumes.back();

            if(vtmp.v_threshold_value != val)
            {
                volumes.push_back( Volume(vol, val) );
            }
            else
            {
                //If ratio is greater than exisiting ratio for given threshold.
                if(vol > vtmp.v_nb_nodes)
                {
                    volumes.pop_back();
                    volumes.push_back( Volume(vol, val) );
                }
            }
        }
        else
            //Add ratio.
            volumes.push_back( Volume(vol, val) );

        //Add volume.
        //volumes.push_back( Volume(vol, val) );

        //Tracking ratio
        if(components.size() > 1)
        {
            std::vector<Component> components_tmp0;
            components_tmp0 = components;

            std::sort( components_tmp0.begin(), components_tmp0.end(), sort_volumes );

            //(1st volume - 2nd volume) / 1st volume
            float ratio = (float)(components_tmp0[0].nb_nodes - components_tmp0[1].nb_nodes)/components_tmp0[0].nb_nodes;

            if(ratios.size() > 0)
            {
                Ratio rtmp = ratios.back();

                if(rtmp.r_threshold_value != val)
                {
                    ratios.push_back( Ratio(ratio, val) );
                }
                else
                {
                    //If ratio is greater than exisiting ratio for given threshold.
                    if(ratio > rtmp.r_ratio)
                    {
                        ratios.pop_back();
                        ratios.push_back( Ratio(ratio, val) );
                    }
                }
            }
            else
                //Add ratio.
                ratios.push_back( Ratio(ratio, val) );
        }
    }
}

//Save to file points that belong to components that touch box boxes.
void save_components_touching_both_boxes(std::vector<Point_2>& circumcenters_coordinates, unsigned long time_step, int val, int ind, int nb_n)
{
    /*
    std::ofstream dataf;;
    dataf.open("data.txt", std::ofstream::out | std::ofstream::app);
    dataf << val << " " << time_step << " " << ind  << " " << nb_n << std::endl;
    dataf.close();
    return;
    */
    std::ostringstream oss;
    oss << "ar_comp_"
       << val << "_" << time_step << "_" << ind  << "_" << nb_n << ".vtk";

    std::string file_name = oss.str();

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    unsigned int size_value = 0;

    //If number of points is even.
    if(circumcenters_coordinates.size() % 2 == 0)
    {
        size_value = circumcenters_coordinates.size();
    }
    else //If it is odd.
    {
        size_value = circumcenters_coordinates.size() - 1;
    }

    ofs << "# vtk DataFile Version 4.0" << endl;
    ofs << "vtk output" << endl;
    ofs << "ASCII" << endl;
    ofs << "DATASET UNSTRUCTURED_GRID" << endl;
    ofs << "POINTS " << size_value << " " << "float" << endl;

    ofs << std::setprecision(2) << std::fixed;

    //Set cooridnates of points.
    for(unsigned long p = 0; p < size_value; p+=2)
    {
        ofs << (float)circumcenters_coordinates[p].x() << " " << (float)circumcenters_coordinates[p].y() << " " << 0 << " "
        << (float)circumcenters_coordinates[p + 1].x() << " " << (float)circumcenters_coordinates[p + 1].y() << " " << 0 << " " << endl;
    }

    //Set type of cells.
    int nb_points = 2;

    ofs << "CELLS " << size_value/2 << " " << size_value/2 * (nb_points + 1) << endl;

    for(unsigned long c = 0; c < size_value; c+=2)
    {
        ofs << 2 << " " << c << " " << c + 1 << endl;
    }

    ofs << endl;

    ofs << "CELL_TYPES" << " " << size_value/2 << endl;

    //Set cell type as a number.
    const int cell_type = 1;

    for(unsigned long ct = 0; ct < size_value/2; ct++)
    {
        ofs << cell_type << endl;
    }

    ofs.close();
}

//Track components that touch both boxes.
void track_components_into_regions(std::vector<Node>& nodes, std::vector<Component>& components)
{
    std::vector<Component> components_tmp = components;

    std::sort( components_tmp.begin(), components_tmp.end(), sort_volumes );

    //Track how the size of the component is changing.
    int counter_files = components_tmp.size();
    for(unsigned int i = 0; i < components_tmp.size(); i++)
    {
        //std::cerr << components_tmp[i].region_A << " " << components_tmp[i].region_B << endl;

        if(components_tmp[i].region_A == true && components_tmp[i].region_B == true)
        //if(components_tmp[i].region_A == true || components_tmp[i].region_B == true)
        {
            if(components_tmp[i].nb_nodes > 0)
            {
                std::vector<Point_2> selected_points;

                for(unsigned int j = 0; j < components_tmp[i].inodes.size(); ++j)
                {
                    Point_2 p = Point_2(nodes[components_tmp[i].inodes[j]].pixel.x(), nodes[components_tmp[i].inodes[j]].pixel.y());

                    selected_points.push_back(p);
                }

                std::cerr << "### Index of component: " << components_tmp[i].index << " ### Max value: " << components_tmp[i].max_value << " ### Size: " << components_tmp[i].nb_nodes << " ### Inodes size: " << components_tmp[i].inodes.size() << " ### Id of file: " << counter_files << endl;

                //Print indices of components.
                for(unsigned int a = 0; a < components_tmp[i].icomponents.size(); a++)
                {
                    std::cerr << "Index: " << components_tmp[i].icomponents[a] << " | ";
                }
                    std::cerr << endl;

                save_components_touching_both_boxes(selected_points, counter_files, components_tmp[i].max_value, components_tmp[i].index, components_tmp[i].nb_nodes);
                counter_files--;
            }
        }
    }
}

//If one of the components touches one of the boxes.
void check_if_touches_box_merging(Component& comp, Component& c0_tmp, Component& c1_tmp, int val)
{
    if(c0_tmp.region_A == true || c1_tmp.region_A == true)
    {
        comp.region_A = true;
    }

    if(c0_tmp.region_B == true || c1_tmp.region_B == true)
    {
        comp.region_B = true;
    }
}

//Check when to start tracking components.
void tracking_size_of_component(int val, Component& component, std::vector<Volume>& volumes, int index_comp, std::vector<Node>& nodes, unsigned time_step)
{
    bool check_flag = false;

    if(check_flag == true)
    {
        //Save if component touches both regions.
        if(component.region_A && component.region_B)
        {
            /*
            for(unsigned int x = 0; x < component.icomponents.size(); x++)
            {
                std::cerr << component.icomponents[x] << " ";
            }

            std::cerr << endl;
            */

            volumes.push_back( Volume(component.nb_nodes, val, component.index, component.icomponents) );
        }
    }
    else
    {
        //Save all possible components for vizualization.
        volumes.push_back( Volume(component.nb_nodes, val, component.index, component.icomponents) );
    }
}

//For adding new node as a new component.
void check_if_touches_box_v2(int n, unsigned int *regionA, unsigned int *regionB, Component& c_tmp, int val, int i)
{
    if(regionA[n] == 1)
    {
        c_tmp.region_A = true;
    }

    if(regionB[n] == 1)
    {
        c_tmp.region_B = true;
    }
}

//If component touches one of the boxes.
void check_if_touches_box(int n0, unsigned int *regionA, unsigned int *regionB, Component& c_tmp, int val, int i)
{
    if(regionA[n0] == 1)
    {
        if(!c_tmp.region_A && c_tmp.region_B)
        {
            c_tmp.max_value_AB = val;
            c_tmp.index = i;
        }

        c_tmp.region_A = true;
    }

    if(regionB[n0] == 1)
    {
        if(c_tmp.region_A && !c_tmp.region_B)
        {
            c_tmp.max_value_AB = val;
            c_tmp.index = i;
        }

        c_tmp.region_B = true;
    }
}

bool sort_nodes(Node const& n0, Node const& n1)
{
    if(n0.value > n1.value)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//Finding root of component.
int find_root(std::vector<Node>& nodes, int n0)
{
    //std::cerr << "find_root " << nodes[n0].parent << n0 << endl;

    //If node is new.
    if(nodes[n0].parent == -1)
    {
        return -1;
    }

    //If n0 is the root.
    if(nodes[n0].parent != n0)
    {
        return find_root(nodes, nodes[n0].parent);
    }
    else
    {
        return nodes[n0].parent;
    }
}

template <typename T, typename T1>
    void finding_neighbours_of_nodes(std::vector<Node>& nodes1, int num_rows, int num_cols, const T *input, const T1 *p_lon, const T1 *p_lat)
{
    for(int i=0, y=0; y<num_rows; y++)
    {
        for(int x=0; x<num_cols; x++, i++)
        {
            std::map<float, int> neigbours;

            if(x > 0) { neigbours.insert(std::pair<float, int>(input[i - 1], i - 1)); } // left

            if(x < num_cols-1) { neigbours.insert(std::pair<float, int>(input[i + 1], i + 1)); } // right

            if(y > 0) { neigbours.insert(std::pair<float, int>(input[i - num_cols], i - num_cols)); } // above

            if(y < num_rows-1) { neigbours.insert(std::pair<float, int>(input[i + num_cols], i + num_cols)); } // below

            nodes1.push_back( Node(i ,input[i], Point(p_lon[x] , p_lat[y]), neigbours) );
        }
    }
}

void check_to_set_birth_value(int n0, unsigned int *regionA, unsigned int *regionB, Component& c_tmp, float val)
{
    if(regionA[n0] == 1)
    {
        if(!c_tmp.region_A && c_tmp.region_B)
        {
            c_tmp.max_value_AB = val;
        }

        c_tmp.region_A = true;
    }

    if(regionB[n0] == 1)
    {
        if(c_tmp.region_A && !c_tmp.region_B)
        {
            c_tmp.max_value_AB = val;
        }

        c_tmp.region_B = true;
    }
}

template <typename T, typename T1>
void save_components(unsigned long time_step, int val, int ind, int nb_n, int num_cols, int num_rows, const T *input, const T1 *p_lon, const T1 *p_lat, Component& component)
{
    std::ostringstream oss;
    oss << "ar_seg_" << val << "_" << time_step << "_" << ind  << "_" << nb_n << ".vtk";
    
    std::string file_name = oss.str();
    
    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);
    
    ofs << "# vtk DataFile Version 4.0" << endl;
    ofs << "vtk output" << endl;
    ofs << "ASCII" << endl;
    ofs << "DATASET RECTILINEAR_GRID" << endl;
    
    ofs << "DIMENSIONS " << num_cols << " " << num_rows << " " << 1 << endl;
    
    ofs << "X_COORDINATES " << num_cols << " " << "double" << endl;
    
    for(int i = 0; i < num_cols; i++)
    {
       ofs << p_lon[i] << " ";
    }
    
    ofs << "\n" << endl;
    
    ofs << "Y_COORDINATES " << num_rows << " " << "double" << endl;
    
    for(int i = 0; i < num_rows; i++)
    {
        ofs << p_lat[i] << " ";
    }
    
    ofs << "\n" << endl;
    
    ofs << "Z_COORDINATES " << 1 << " " << "double" << endl;
    ofs << 0 << endl;
    
    ofs << "POINT_DATA " << num_cols*num_rows << endl;
    ofs << "FIELD" << " " << "FieldData" << " " << 2 << endl;
    
    ofs << "vapor " << 1 << " " << num_rows*num_cols << " " << "float" << endl;
    
    for(int i = 0; i < num_rows*num_cols; i++)
    {
        ofs << input[i] << " ";
    }
    
    ofs << "\n" << endl;
    
    ofs << "thresh " << 1 << " " << num_cols*num_rows << " unsigned_int" << endl;

    //Segmentation based on nodes of component.
    unsigned int segment_tab[num_rows*num_cols] = {0};
    
    for(int j = 0; j < (int)component.inodes.size(); j++)
    {
        segment_tab[component.inodes[j]] = 1;
    }
    
    for(int i = 0; i < num_cols*num_rows; i++)
    {
        ofs << segment_tab[i] << " ";
    }
    
    ofs.close();
}

template <typename T, typename T1>
void tracking_volume_of_component(unsigned long time_step, int val,
    Component& component, std::vector<Volume>& volumes, bool tracking_flag,
    std::vector<Node>& nodes, int num_cols, int num_rows, const T *input, const T1 *p_lon, const T1 *p_lat)
{
    if(tracking_flag)
    {
        //Save if component touches both regions.
        if(component.region_A && component.region_B)
        {
            volumes.push_back( Volume(component.nb_nodes, val, component.index, component.icomponents) );

            //Save points to file.
            std::vector<Point_2> selected_points;

            for(unsigned int j = 0; j < component.inodes.size(); ++j)
            {
                Point_2 p = Point_2(nodes[component.inodes[j]].pixel.x(), nodes[component.inodes[j]].pixel.y());

                selected_points.push_back(p);
            }

            //save_components_touching_both_boxes(selected_points, time_step, val, component.index, component.nb_nodes);
            save_components(time_step, val, component.index, component.nb_nodes, num_cols, num_rows, input, p_lon, p_lat, component);
        }
    }
    else
    {
        volumes.push_back( Volume(component.nb_nodes, val, component.index, component.icomponents) );
    }
}

void check_two_region_constraint(int n0, unsigned int *regionA, unsigned int *regionB, Component& c_tmp)
{
    if(regionA[n0] == 1)
    {
        c_tmp.region_A = true;
    }

    if(regionB[n0] == 1)
    {
        c_tmp.region_B = true;
    }
}

void check_when_merging(Component& new_comp, Component& c0, Component& c1, float value, int i)
{
    //Check if both components (c0 and c1) touch both boxes.
    bool c0_both = c0.region_A && c0.region_B;
    bool c1_both = c1.region_A && c1.region_B;

    //If c0 and c1 touches both boxes.
    if(c0_both && c1_both)
    {
        if(c0.max_value_AB > c1.max_value_AB)
        {
            new_comp.index = c0.index;
            new_comp.max_value_AB = c0.max_value_AB;
        }
        else
        {
            new_comp.index = c1.index;
            new_comp.max_value_AB = c1.max_value_AB;
        }

        new_comp.region_A = true;
        new_comp.region_B = true;
    }
    //If c0 and c1 does not touch both boxes.
    else if(!c0_both && !c1_both)
    {
        if(c0.region_A && !c0.region_B && !c1.region_A && c1.region_B)
        {
            new_comp.max_value_AB = value; //current value
            new_comp.index = i;

            new_comp.region_A = true;
            new_comp.region_B = true;
        }
        else if(!c0.region_A && c0.region_B && c1.region_A && !c1.region_B)
        {
            new_comp.max_value_AB = value; //current value
            new_comp.index = i;

            new_comp.region_A = true;
            new_comp.region_B = true;
        }
        else
        {
            new_comp.index = i;

            //If touches one of the boxes.
            if(c0.region_A || c1.region_A)
            {
                new_comp.region_A = true;
            }

            if(c0.region_B || c1.region_B)
            {
                new_comp.region_B = true;
            }
        }
    }
    //If c0 does not touch both boxes, but c1 touches both.
    else if (!c0_both && c1_both)
    {
        new_comp.index = c1.index;
        new_comp.max_value_AB = c1.max_value_AB;

        new_comp.region_A = true;
        new_comp.region_B = true;
    }
    //If c0 touches both boxes, but c0 does not touch both.
    else if (c0_both && !c1_both)
    {
        new_comp.index = c0.index;
        new_comp.max_value_AB = c0.max_value_AB;

        new_comp.region_A = true;
        new_comp.region_B = true;
    }
    else
    {
        std::cerr << "DEBUG" << endl;
    }
}

template <typename T, typename T1>
    void finding_components_v2(std::vector<Component>& components, unsigned long time_step, unsigned int *regionA, unsigned int *regionB, int num_cols, int num_rows, const T *input, const T1 *p_lon, const T1 *p_lat)
{
    std::vector<Volume> volumes;

    std::vector<Node> nodes; //Unsorted list of nodes.
    std::vector<Node> sorted_nodes; //Sorted list of nodes.

    //Creating nodes and finding theirs four neighbours.
    finding_neighbours_of_nodes(nodes, num_rows, num_cols, input, p_lon, p_lat);

    //Make a copy of the vector.
    sorted_nodes = nodes;

    //Sort nodes in the decreasing order.
    std::sort( sorted_nodes.begin(), sorted_nodes.end(), sort_nodes );

    int n_nodes = sorted_nodes.size();
    std::cerr << "Nodes size: " << sorted_nodes.size() << endl;

    int n0, n1, r0, r1, c0, c1;
    float val, val_n1;

    bool tracking_flag = true;

    for(int n = 0; n < n_nodes; n++)
    {
        //Get real index of node.
        n0 = sorted_nodes[n].index;
        //It is a root of itself.
        nodes[n0].parent = n0;
        //Store index of a new component.
        nodes[n0].component = (int)components.size();

        //Create new component.
        Component comp(nodes[n0]);
        comp.inodes.push_back(n0);

        if(tracking_flag)
            check_two_region_constraint(n0, regionA, regionB, comp);

        components.push_back(comp);

        //For each neighbour of node n.
        for (std::map<float, int>::iterator it = sorted_nodes[n].four_neighbors.begin(); it!=sorted_nodes[n].four_neighbors.end(); ++it)
        {
            //Get index of neighbour.
            n1 = it->second;

            //Get value.
            val_n1 = it->first;

            //Weight of edge between n0 and its neighbour.
            val = std::min(sorted_nodes[n].value, val_n1);

            //Skip lower nodes.
            if(val_n1 < sorted_nodes[n].value || (val_n1 == sorted_nodes[n].value && n1 < n0))
            {
                std::cerr << "Skip node" << endl;
                continue;
            }
            else
            {
                //Merge two components.
                std::cerr << "Merge" << endl;

                //Find roots of both nodes.
                r0 = find_root(nodes, n0);
                r1 = find_root(nodes, n1);

                //Find indices of components for which the nodes belong to.
                c0 = nodes[r0].component;
                c1 = nodes[r1].component;

                //If nodes are in the same component.
                if(c0 == c1)
                    continue;

                //Merge two components.
                Component comp_merge(components[c0], components[c1]);

                check_when_merging(comp_merge, components[c0], components[c1], val, (int)components.size());

                nodes[r0].parent = r1;
                nodes[r1].component = comp_merge.index;

                comp_merge.inodes.push_back(r1); //???

                tracking_volume_of_component(time_step, val, comp_merge, volumes, tracking_flag, nodes, num_cols, num_rows, input, p_lon, p_lat);

                if((int)components.size() != comp_merge.index)
                    components[comp_merge.index] = comp_merge;
                else
                    components.push_back(comp_merge);
            }
        }
    }

    save_volume_to_file(volumes, time_step);
}

//Finding connected components based on the Union-Find data structure.
//We track the growth of components with the varying threshold value.
void finding_components(std::vector<Node> nodes, std::vector<Edge> edges, std::vector<Component>& components, unsigned long time_step, unsigned int *regionA, unsigned int *regionB)
{
    std::vector<Ratio> ratios;
    std::vector<Volume> volumes;

    int val, n0, n1;
    int r0, r1;
    int num_nodes;
    int c0, c1;
    int nb_edges = edges.size();

    for(int e = 0; e < nb_edges; e++)
    {
        val = edges[e].weight;
        n0 = edges[e].endpoint0;
        n1 = edges[e].endpoint1;

        //Swap if n0 is smaller than n1.
        if(nodes[n0].value > nodes[n1].value)
        {
            std::swap(n0, n1);
        }

        r1 = find_root(nodes, n1);

        //If it equals -1, both nodes are new (there is no higher node).
        if(r1 == -1)
        {
            num_nodes++;

            //Now n1 is a root of itself.
            r1 = n1;
            nodes[n1].parent = n1;
            nodes[n1].component = (int)components.size();

            //Create new component.
            Component comp(nodes[n1]); //Max value at birth.
            comp.inodes.push_back(n1);

            //If component touches one of the boxes.
            check_if_touches_box_v2(n1, regionA, regionB, comp, val, int(components.size()));

            components.push_back(comp);
        }

        r0 = find_root(nodes, n0);

        c1 = nodes[r1].component;

        //If second node is also new. It has lower value and it will joint the component of the highest node.
        if(r0 == -1)
        {
            num_nodes++;

            //The higher node is now the parent.
            nodes[n0].parent = r1;

            //Join lower node wit higher node.
            add_node(n0, nodes[n0], components[c1]);

            //If component touches one of the boxes.
            check_if_touches_box(n0, regionA, regionB, components[c1], val, int(components.size()));

            //Checking if we can start tracking components.
            tracking_size_of_component(val, components[c1], volumes, components[c1].index, nodes, time_step);

            //Add other node to set - go to the next edge.
            continue;
        }

        c0 = nodes[r0].component;

        //If edge is inside one component.
        if(c0 == c1)
        {
            continue;
        }
        else
        {
            //Join two components
            Component comp(int(components.size()), val, components[c0], components[c1]);
            nodes[r0].parent = r1;
            nodes[r1].component = comp.index;

            //Checking if we can start tracking components.
            tracking_size_of_component(val, comp, volumes, comp.index, nodes, time_step);

            components.push_back(comp);
        }
    }

    std::cerr << "\nNB of components " << components.size() << endl;

    //Save volume to text file.
    save_volume_to_file(volumes, time_step);
}

//Sort values in decreasing order based on the weight values.
bool sort_edges(Edge const& e0, Edge const& e1)
{
    if(e0.weight > e1.weight)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//Restrict to predefined regions (e.g., lower boundary of the box and land).
template <typename T>
    void generate_regions_in_box(unsigned long height_box, const T *p_land_sea_mask, unsigned long num_rc, unsigned long num_rows, unsigned long num_cols, unsigned int *outputA, unsigned int *outputB)
{
    unsigned int output_region_A_tmp[num_rc] = {0};

    //Create two sets.
    for(unsigned long i = 0; i < num_rc; ++i)
    {
        //Assigning ones at the same place where is the land-sea mask.
        if(p_land_sea_mask[i] == 1)
            output_region_A_tmp[i] = 1;

        //Assigning ones to the last rows at the bottom of the box.
        if(i < height_box*num_cols)
            outputB[i] = 1;
    }

    //Compute labels for data points using SAUF.
    unsigned int num_comp = sauf(num_rows, num_cols, output_region_A_tmp);
    std::cerr << "Nb components: " << num_comp << endl;

    //Set of size of connected components.
    unsigned int label_counter[num_comp] = {0};

    //Calculate size of each component.
    for(unsigned long j = 1; j <= num_comp; j++)
    {
        int counter = 0;

        for(unsigned long i = 0; i < num_rc; ++i)
        {
            if(output_region_A_tmp[i] == j)
            {
                counter++;
            }
        }

        label_counter[j-1] = counter;
    }

    //Take those components that are greater than some arbitrary size value.
    for(unsigned long i = 1; i <= num_comp; ++i)
    {
        //Arbitrary value for skipping components.
        unsigned int skipping_size = 40;

        if(label_counter[i-1] < skipping_size)
        {
            continue;
        }
        else
        {
            for(unsigned long j = 0; j < num_rc; ++j)
            {
                if(output_region_A_tmp[j] == i)
                {
                    outputA[j] = 1;
                }
            }
        }
    }

    print_array(outputA, num_rc, num_rows, num_cols);
    print_array(outputB, num_rc, num_rows, num_cols);

}

//Build graph on IWV 2d grid.
template <typename T>
    void union_find_alg(const T *input, unsigned long num_rc, unsigned long num_rows,
                    unsigned long num_cols, const_p_teca_variant_array lat,
                    const_p_teca_variant_array lon, unsigned long time_step, const_p_teca_variant_array land_sea_mask)
{
    //If want to generate specific regions.
    bool reg_flag = true;

    NESTED_TEMPLATE_DISPATCH_FP(
            const teca_variant_array_impl,
            lat.get(),
            1,

            NESTED_TEMPLATE_DISPATCH(
            const teca_variant_array_impl,
            land_sea_mask.get(),
            2,

            const NT1 *p_lat = dynamic_cast<TT1*>(lat.get())->get();
            const NT1 *p_lon = dynamic_cast<TT1*>(lon.get())->get();

            const NT2 *p_land_sea_mask = dynamic_cast<TT2*>(land_sea_mask.get())->get();

            unsigned int output_region_ls_A_box[num_rc] = {0};
            unsigned int output_region_b_B_box[num_rc] = {0};

            if(reg_flag)
            {
                //Size of the bottom box;
                unsigned long height_box = 1;

                //Generate specific regions of interest.
                generate_regions_in_box(height_box, p_land_sea_mask, num_rc, num_rows, num_cols, output_region_ls_A_box, output_region_b_B_box);
            }

            std::vector<Edge> edges;
            std::vector<Node> nodes;

            //Build graph.
            build_graph(nodes, edges, num_rows, num_cols, input, p_lon, p_lat);

            //Sort edges in decreasing order.
            std::sort( edges.begin(), edges.end(), sort_edges );

            std::cerr << "Nb of nodes based on 2D grid: " << nodes.size() << endl;
            std::cerr << "Nb of edges based on 2D grid: " << edges.size() << endl;

            std::vector<Component> components;
            std::vector<Node> output_nodes;
            output_nodes = nodes;

            //Find connected components based on union-find data structure.
            //finding_components(output_nodes, edges, components, time_step, output_region_ls_A_box, output_region_b_B_box);

            finding_components_v2(components, time_step, output_region_ls_A_box, output_region_b_B_box, num_cols, num_rows, input, p_lon, p_lat);
    ))

}

//Finding 4-connected space based on 2D grid of IWV data.
template <typename T, typename T1>
    void build_graph(std::vector<Node>& nodes, std::vector<Edge>& edges,
                 unsigned long num_rows, unsigned long num_cols, const T *input,
                 const T1 *p_lon, const T1 *p_lat)
{
    //Build a set of nodes.
    for (unsigned long j = 0; j < num_rows; j++)
    {
        for (unsigned long i = 0; i < num_cols; i++)
        {
            unsigned long q = j * num_cols + i;

            //std::cerr << q << endl;

            nodes.push_back( Node(q ,input[q], Point(p_lon[i] , p_lat[j])) ); //Adding value for a pixel (IWV value) and real position in a grid (lon and lat).
        }
    }

    //Build a set of edges (4-connectivity) based on "ghost zones".
    int ng = 1;
    unsigned long nx = (num_cols - 1) + 1;

    //std::cerr << num_cols << " " << num_rows << endl;

    for(unsigned long j = ng, jj = 0; j <= (num_rows - 1) - ng; ++j, ++jj)
    {
        for(unsigned long i = ng, ii = 0; i <= (num_cols - 1) - ng; ++i, ++ii)
        {
            unsigned long q = j * nx + i;

            //std::cerr << q << endl;

            edges.push_back( Edge(std::min(input[q], input[q - 1]), q, q - 1) ); //left
            edges.push_back( Edge(std::min(input[q], input[q + 1]), q, q + 1) ); //right
            edges.push_back( Edge(std::min(input[q], input[q - nx]), q, q - nx) ); //top
            edges.push_back( Edge(std::min(input[q], input[q + nx]), q, q + nx) ); //down
        }
    }
}

//Rule for sorting points based on their coordinates.
bool xComparator(const Point_2& a, const Point_2& b)
{
    if(a.y() < b.y())
        return true;
    if(a.y() > b.y())
        return false;
    return a.x()<b.x();
}

//Print array for debugging purposes.
void print_array(unsigned int *p_con_comp_skel, unsigned long num_rc, unsigned long num_rows, unsigned long num_cols)
{
    for(unsigned long i = 0; i < num_rc; i+=num_cols)
    {
        for(unsigned long j = i; j < i + num_cols; j++)
        {
            std::cerr << p_con_comp_skel[j] << " ";
        }
        std::cerr << endl;
    }
    std::cerr << endl;
}

//Finding neighbours and boundary points.
template <typename T, typename T1>
    void find_neighbours(unsigned long *lext, unsigned int *igrid,
                         unsigned long *oext, unsigned int *ogrid, unsigned int nx,
                         unsigned int nxx, const T *p_land_sea_mask, std::vector<Point_2>& selected_data_points,
                         std::vector<unsigned long>& labels_of_selected_dp, const T1 *p_lon, const T1 *p_lat)
    {
        for(unsigned int j = lext[2], jj = oext[2]; j <= lext[3]; ++j, ++jj)
        {
            for(unsigned int i = lext[0], ii = oext[0]; i <= lext[1]; ++i, ++ii)
            {
                int q = j * nx + i;
                int qq = jj * nxx + ii;

                if(igrid[q] > 0 || p_land_sea_mask[q] > 0)
                {
                    continue;
                }
                else if(igrid[q + 1] > 0)
                {
                    ogrid[qq] = igrid[q + 1];
                }
                else if(igrid[q - 1] > 0)
                {
                    ogrid[qq] = igrid[q - 1];
                }
                else if(igrid[q + nx] > 0)
                {
                    ogrid[qq] = igrid[q + nx];
                }
                else if(igrid[q - nx] > 0)
                {
                    ogrid[qq] = igrid[q - nx];
                }
            }
        }

        //Find boundary points.
        for (unsigned long j = lext[2], jj = oext[2]; j <= lext[3]; ++j, ++jj)
        {
            for (unsigned long i = lext[0], ii = oext[0]; i <= lext[1]; ++i, ++ii)
            {
                unsigned long qq = jj * nxx + ii;

                //if(qq == lext[1])
                //{
                    //ogrid[qq] = 9;
                    //std::cerr << qq << endl;
                //}

                if (ogrid[qq] > 0)
                {
                    Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                    selected_data_points.push_back(p1);
                    labels_of_selected_dp.push_back(ogrid[qq]);
                }
            }
        }
    }

//Finding segmentation based on IWV values and land-sea mask.
template <typename T, typename T1>
    void find_segmentation(const T *input, unsigned int *p_con_comp_skel, T low, unsigned long num_rc, unsigned long num_cols, const T1 *p_land_sea_mask)
{
    //Find points needed for labeling.
    for(unsigned long i = 0; i < num_rc; i+=num_cols)
    {
        for(unsigned long j = i; j < num_cols+i; j++)
        {
            if(input[j] >= low || p_land_sea_mask[j] > 0)
                p_con_comp_skel[j] = 0;
            else
                p_con_comp_skel[j] = 1;
        }
    }
}

void save_boundary_points(std::vector<Point_2>& selected_data_points, unsigned long time_step)
{
    std::ostringstream oss;
    oss << "ar_boundary_points_" << std::setw(6) << std::setfill('0') << time_step << ".txt";

    std::string file_name = oss.str();

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    for (std::vector<Point_2>::iterator it = selected_data_points.begin(), end = selected_data_points.end(); it != end; ++it)
    {
        ofs << (*it).x() << " , " << (*it).y() << endl;
    }

    ofs.close();
}

//Find boundary points.
template <typename T, typename T1>
void find_boundary_points(std::vector<Point_2>& selected_data_points,
                          std::vector<Coordinate>& values_selected_points,
                          const T *input,
                          unsigned int *output,
                          unsigned long n_vals,
                          unsigned long num_cols,
                          unsigned long num_rows,
                          const T1 *p_lon, const T1 *p_lat, T low, T high, unsigned long time_step)
{
    for (unsigned long i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;

    int ng = 1;

    unsigned long n_size_r = (num_rows - 1);
    unsigned long n_size_c = (num_cols - 1);

    for (unsigned long j = ng; j <= (n_size_r - ng); ++j)
    {
        for (unsigned long i = ng; i <= (n_size_c - ng); ++i)
        {
            unsigned long q = j * num_cols + i;

            if(output[q] == 1)
            {
                //At the bottom of the boundary.
                if(output[q - num_cols] == 1 && q >= num_cols && q <= (2*num_cols))
                {
                    Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                    selected_data_points.push_back(p1);
                }

                /*
                //Right site of the boundary.
                if(output[q - num_cols] == 1 && output[q + 1] == 1 && i == n_size_c - ng)
                {
                    Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                    selected_data_points.push_back(p1);
                }

                //Left site of the boundary.
                if(output[q - num_cols] == 1 && output[q - 1] == 1 && i == ng)
                {
                    Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                    selected_data_points.push_back(p1);
                }
                */
                continue;
            }
            else if(output[q + 1] == 1)
            {
                Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                selected_data_points.push_back(p1);
            }
            else if(output[q - 1] == 1)
            {
                Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                selected_data_points.push_back(p1);
            }
            else if(output[q + num_cols] == 1)
            {
                Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                selected_data_points.push_back(p1);
            }
            else if(output[q - num_cols] == 1)
            {
                Point_2 p1 = Point_2(p_lon[i] , p_lat[j]);

                selected_data_points.push_back(p1);
            }
        }
    }

    //Save real coordinates of pixels.
    for (unsigned long j = ng; j <= n_size_r - ng; ++j)
    {
        for (unsigned long i = ng; i <= n_size_c - ng; ++i)
        {
            int q = j * num_cols + i;

            unsigned int param0 = p_lon[i];//round(p_lon[i]);
            unsigned int param1 = p_lat[j];//round(p_lat[j]);

            Coordinate coo = Coordinate(param0, param1, output[q]);
            values_selected_points.push_back(coo);
        }
    }

    save_boundary_points(selected_data_points, time_step);
}

//Checking if skeleton points are outside of the segmentation.
bool check_points(const Point_2& p0, const Point_2& p1, std::vector<Point_2> selected_data_points, std::vector<Coordinate>& values_selected_points)
{
    unsigned int point0x = trunc(p0.x());
    unsigned int point0y = trunc(p0.y());

    unsigned int point1x = trunc(p1.x());
    unsigned int point1y = trunc(p1.y());

    bool flag0 = true;
    bool flag1 = true;

    for (std::vector<Coordinate>::iterator it = values_selected_points.begin(), end = values_selected_points.end(); it != end; ++it)
    {
        unsigned int tmp_pointX = (*it).x;
        unsigned int tmp_pointY = (*it).y;
        unsigned int lab = (*it).label;

        //std::cerr << (*it).x << " , " << (*it).y << " , " << lab << endl;

        if(lab == 0)
        {
            //std::cerr << (*it).x << " , " << (*it).y << " YES" << endl;

            if(point0x == tmp_pointX && point0y == tmp_pointY)
            {
                flag0 = false;
            }

            if(point1x == tmp_pointX && point1y == tmp_pointY)
            {
                flag1 = false;
            }
        }
    }

    return (flag0 && flag1) ? 1 : 0;
}

//Checking if one skeleton point is outside of the segmentation.
bool check_pointsv2(const Point_2& p0, std::vector<Coordinate>& values_selected_points)
{
    unsigned int point0x = round(p0.x());
    unsigned int point0y = round(p0.y());

    bool flag = true;

    //std::cerr << "##Circumcenter" << p0.x() << " , " << p0.y() << endl;

    int counter = 0;

    for (std::vector<Coordinate>::iterator it = values_selected_points.begin(), end = values_selected_points.end(); it != end; ++it)
    {
        unsigned int tmp_pointX = round((*it).x);
        unsigned int tmp_pointY = round((*it).y);
        unsigned int lab = (*it).label;

        if(point0x == tmp_pointX && point0y == tmp_pointY)
        {
            counter++;
            if(lab == 0)
            {
                flag = false;
                break;
            }
            else
            {
                continue;
            }
        }
        else
        {
            continue;
        }
    }

    //if(counter == 0) //IT WORKS WITH THIS CONDITION BUT NOT SO GOOD !!! SOME POINTS ARE REMOVED FROM SEGMENTATION.
      //  flag = false;

    return flag;
}

//Checking if skeleton points are outside of the patch.
template <typename T>
    bool check_constraints(const Point_2& p0, const Point_2& p1,
                       T search_lat_low, T search_lon_low,
                       T search_lat_high, T search_lon_high)
{
    if(p0.y() > search_lat_high - 1 || p0.y() < search_lat_low - 1 || p0.x() > search_lon_high - 1 || p0.x() < search_lon_low - 1)
    {
        return false;
    }
    else if(p1.y() > search_lat_high - 1 || p1.y() < search_lat_low - 1 || p1.x() > search_lon_high - 1 || p1.x() < search_lon_low - 1)
    {
        return false;
    }
    else
        return true;
}

template <typename T>
    void compute_triangulation_voronoi_diagram(std::vector<Point_2> selected_data_points,
                                               std::vector<Coordinate> values_selected_points,
                                               unsigned long time_step,
                                               T search_lat_low, T search_lon_low,
                                               T search_lat_high, T search_lon_high)
{
    //Create list of indices for boundary points.
    std::vector<unsigned long> list_of_indices;
    for(unsigned long int j = 0; j < selected_data_points.size(); j++)
        list_of_indices.push_back(j);

    //Merge boundary points with created indices.
    Delaunay dt2;
    dt2.insert(boost::make_zip_iterator(boost::make_tuple(selected_data_points.begin(),
                                                          list_of_indices.begin() )),
                                                          boost::make_zip_iterator(boost::make_tuple( selected_data_points.end(),
                                                          list_of_indices.end() )));

    //Coordinates of skeleton points.
    std::vector<Point_2> circumcenters_coordinates;
    std::vector<Point_2> triangules_coordinates;

    Delaunay::Finite_edges_iterator eit;

    for (eit = dt2.finite_edges_begin(); eit != dt2.finite_edges_end(); ++eit)
    {
        //Skip edge if it is infinite.
        CGAL::Object o = dt2.dual(eit);
        if (CGAL::object_cast<K::Ray_2>(&o)) continue;

        //Take the edge.
        DEDGE e = *eit;

        //Find neigbouring triangles.
        std::pair<FHN,FHN> faces = FacesN(e);

        //First face.
        FHN face0 = faces.first;
        Point_2 point0 = dt2.dual(face0);

        //Second face.
        FHN face1 = faces.second;
        Point_2 point1 = dt2.dual(face1);

        //Check if points are inside the patch.
        if(check_constraints(point0, point1, search_lat_low, search_lon_low, search_lat_high, search_lon_high))
        {
            //Check if points are inside the segmentation.
            /*
            if(check_points(point0, point1, selected_data_points, values_selected_points))
            {
               circumcenters_coordinates.push_back(point0);
               circumcenters_coordinates.push_back(point1);
            }
            */

            if(check_pointsv2(point0, values_selected_points) && check_pointsv2(point1, values_selected_points))
            {
                circumcenters_coordinates.push_back(point0);
                circumcenters_coordinates.push_back(point1);
            }
        }
    }

    if(circumcenters_coordinates.size() > 0)
        save_circumcenters_to_vtk_file(circumcenters_coordinates, time_step);

    //For ploting triangulation.
    for(Delaunay::Finite_faces_iterator fi = dt2.finite_faces_begin(); fi != dt2.finite_faces_end(); fi++)
    {
        Point_2 point0 = Point_2(fi->vertex(0)->point().hx(), fi->vertex(0)->point().hy());
        Point_2 point1 = Point_2(fi->vertex(1)->point().hx(), fi->vertex(1)->point().hy());
        Point_2 point2 = Point_2(fi->vertex(2)->point().hx(), fi->vertex(2)->point().hy());

        //Save vertices of triangle.
        triangules_coordinates.push_back(point0);
        triangules_coordinates.push_back(point1);
        triangules_coordinates.push_back(point2);
    }

    //Save Delaunay triangulation to VTK file.
    if(triangules_coordinates.size() > 0)
        save_triang_to_vtk_file(triangules_coordinates, time_step);
}

//Function that computes global skeleton based on Delanuay triangulation.
template <typename T>
    void compute_global_skeleton(const T *input, unsigned long num_rc, unsigned long num_rows, unsigned long num_cols, const_p_teca_variant_array lat, const_p_teca_variant_array lon, unsigned long time_step,
                              T search_lat_low,
                              T search_lon_low,
                              T search_lat_high,
                              T search_lon_high,
                              T low, T high)
{
    TEMPLATE_DISPATCH_FP(
         const teca_variant_array_impl,
         lat.get(),

         const NT *p_lat = dynamic_cast<TT*>(lat.get())->get();
         const NT *p_lon = dynamic_cast<TT*>(lon.get())->get();

         p_teca_unsigned_int_array con_comp_skel = teca_unsigned_int_array::New(num_rc, 0);
         unsigned int *p_con_comp_skel = con_comp_skel->get();

         std::vector<Point_2> selected_data_points;

         //Values neeed for coordinates approximation of points.
         std::vector<Coordinate> values_selected_points;

         //Finding boundary potins for triangulation alg.
         find_boundary_points(selected_data_points, values_selected_points, input, p_con_comp_skel, num_rc, num_cols, num_rows, p_lon, p_lat, low, high, time_step);

         //print_array(p_con_comp_skel, num_rc, num_rows, num_cols);

         //straight_skeleton_ar(selected_data_points, time_step);

         //Computing triangulation and finding Voronoi points.
         compute_triangulation_voronoi_diagram(selected_data_points, values_selected_points, time_step, search_lat_low, search_lon_low, search_lat_high, search_lon_high);
    )
}

//Function that computes local skeleton based on Delanuay triangulation.
template <typename T>
    void compute_local_skeleton(const_p_teca_variant_array land_sea_mask,
                            unsigned long num_rc,
                            const T *input, T low,
                            unsigned long num_rows,
                            unsigned long num_cols, const_p_teca_variant_array lat, const_p_teca_variant_array lon, unsigned long time_step,
                            T search_lat_low,
                            T search_lon_low,
                            T search_lat_high,
                            T search_lon_high)
{
     NESTED_TEMPLATE_DISPATCH_FP(
         const teca_variant_array_impl,
         lat.get(),
         1,

         NESTED_TEMPLATE_DISPATCH(
         const teca_variant_array_impl,
         land_sea_mask.get(),
         2,

         const NT1 *p_lat = dynamic_cast<TT1*>(lat.get())->get();
         const NT1 *p_lon = dynamic_cast<TT1*>(lon.get())->get();

         const NT2 *p_land_sea_mask = dynamic_cast<TT2*>(land_sea_mask.get())->get();

         //Vectors for boundary points and labels for Delanuay triang.
         std::vector<Point_2> selected_data_points;
         std::vector<unsigned long> labels_of_selected_dp;

         //Array with labels for SAUF algorithm.
         p_teca_unsigned_int_array con_comp_skel = teca_unsigned_int_array::New(num_rc, 0);
         unsigned int *p_con_comp_skel = con_comp_skel->get();

         //Segmentation of data based on water vapour and land-sea mask.
         find_segmentation(input, p_con_comp_skel, low, num_rc, num_cols, p_land_sea_mask);

         //std::cerr << "\n### Segmentation" << endl;
         print_array(p_con_comp_skel, num_rc, num_rows, num_cols);

         //Compute labels for data points using SAUF.
         int num_comp = sauf(num_rows, num_cols, p_con_comp_skel);
         std::cerr << "\n### Nb of connected components = " << num_comp << endl;

         //Print content of array.
         print_array(p_con_comp_skel, num_rc, num_rows, num_cols);

         //Size of ghost zone.
         int ng = 1;

         //Initialize arrays for function that uses ghost zone.
         unsigned long iext[4] = {0};
         iext[0] = 0;
         iext[1] = num_cols - 1;
         iext[2] = 0;
         iext[3] = num_rows - 1;

         unsigned long nx = iext[1] - iext[0] + 1;
         unsigned long ny = iext[3] - iext[2] + 1;

         unsigned long lext[4] = {0};
         lext[0] = iext[0] + ng;
         lext[1] = iext[1] - ng;
         lext[2] = iext[2] + ng;
         lext[3] = iext[3] - ng;

         unsigned long nyy = lext[3] - lext[2] + 1;
         unsigned long nxx = lext[1] - lext[0] + 1;

         unsigned long oext[4] = {0};
         oext[0] = 0;
         oext[1] = nx - 2*ng - 1;
         oext[2] = 0;
         oext[3] = ny - 2*ng - 1;

         p_teca_unsigned_int_array p_ogrid_array = teca_unsigned_int_array::New(nxx * nyy, 0);
         unsigned int *p_ogrid = p_ogrid_array->get();

         //Find boundary points of segmentation.
         find_neighbours(lext, p_con_comp_skel, oext, p_ogrid, nx, nxx, p_land_sea_mask, selected_data_points, labels_of_selected_dp, p_lon, p_lat);

         print_array(p_ogrid, nxx*nyy, nyy, nxx);

         //Save some boundary points for other method.
         //save_boundary_points(selected_data_points, time_step);

         //Compute Voronoi points based on CGAL library.
         compute_voronoi_diagram(selected_data_points, labels_of_selected_dp, time_step);
     ))
}

//Computing skeleton based on Delanuay triangulation.
template <typename T>
    void compute_skeleton_of_ar(const_p_teca_variant_array land_sea_mask,
                            unsigned int *p_con_comp,
                            unsigned long num_rc,
                            const T *input,
                            T low, T high,
                            unsigned long num_rows,
                            unsigned long num_cols, const_p_teca_variant_array lat, const_p_teca_variant_array lon, unsigned long time_step,
                            T search_lat_low,
                            T search_lon_low,
                            T search_lat_high,
                            T search_lon_high)
{
    //If flag is true - calculate global skeleton.
    //Otherwise - calculate local skeleton.
    bool type_flag = false;

    if(type_flag)
    {
        compute_global_skeleton(input, num_rc, num_rows, num_cols, lat, lon, time_step, search_lat_low, search_lon_low, search_lat_high, search_lon_high, low, high);
    }
    else
    {
        compute_local_skeleton(land_sea_mask, num_rc, input, low, num_rows, num_cols, lat, lon, time_step, search_lat_low, search_lon_low, search_lat_high, search_lon_high);
    }
}

//Finding a pair of neighbouring triangles (faces).
std::pair<FHN,FHN> FacesN (DEDGE const& e)
{
    //If we have an edge e, then e.first and e.first.neighbor( e.second ) are the two incident facets.
    FHN f1 = e.first;
    FHN f2 = e.first->neighbor( e.second );
    return std::make_pair (f1, f2);
}

//Saving values of width to text file.
void save_width_to_file(std::vector<float> ar_widths, unsigned long time_step)
{
    std::ostringstream oss;
    oss << "ar_width_" << std::setw(6) << std::setfill('0') << time_step << ".txt";

    std::string file_name = oss.str();

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    //save one value per line
    /*
     23.2
     34.3
     24.2
     ...
     23.2
     */
    for (std::vector<float>::iterator it = ar_widths.begin(), end = ar_widths.end(); it != end; ++it)
    {
        ofs << (*it) << endl;
    }

    ofs.close();
}

//Computing distance between two points.
float compute_distance(Point_2 const& p0, Point_2 const& p1)
{
    float dx;
    float dy;
    float dist;

    dx = p0.x() - p1.x();
    dy = p1.y() - p1.y();

    dist = sqrt(dx*dx - dy*dy);

    return dist;
}

//Computing median width.
float compute_width(std::vector<float>& vec)
{
    std::sort(vec.begin(), vec.end());

    if(vec.size() % 2 == 0)
        return (vec[vec.size()/2 - 1] + vec[vec.size()/2]) / 2;
    else
        return vec[vec.size()/2];
}

//Computing Voronoi diagram based on Delanuay triangulation.
void compute_voronoi_diagram(std::vector<Point_2> selected_data_points, std::vector<unsigned long> labels_of_selected_dp, unsigned long time_step)
{
    //Create list of indices for boundary points.
    std::vector<unsigned long> list_of_indices;
    for(unsigned long int j = 0; j < labels_of_selected_dp.size(); j++)
        list_of_indices.push_back(j);

    //Merge boundary points with created indices.
    Delaunay dt2;
    dt2.insert(boost::make_zip_iterator(boost::make_tuple(selected_data_points.begin(),
                                                          list_of_indices.begin() )),
                                                          boost::make_zip_iterator(boost::make_tuple( selected_data_points.end(),
                                                          list_of_indices.end() )));

    CGAL_assertion(dt2.number_of_vertices() == selected_data_points.size());

    //Coordinates of skeleton points.
    std::vector<Point_2> circumcenters_coordinates;
    std::vector<Point_2> triangules_coordinates;

    //Vector of calculated widths.
    std::vector<float> ar_widths;

    Delaunay::Finite_edges_iterator eit;

    for (eit = dt2.finite_edges_begin(); eit != dt2.finite_edges_end(); ++eit)
    {
        //Skip edge if it is infinite.
        CGAL::Object o = dt2.dual(eit);
        if (CGAL::object_cast<K::Ray_2>(&o)) continue;

        //Take the edge.
        DEDGE e = *eit;

        //Find indices of vertices of selected edge.
        int i1= e.first->vertex( (e.second+1)%3 )->info();
        int i2= e.first->vertex( (e.second+2)%3 )->info();

        //Check if labels are different.
        if(labels_of_selected_dp[i1] != labels_of_selected_dp[i2])
        {
            //Find neigbouring triangles.
            std::pair<FHN,FHN> faces = FacesN(e);

            //First face.
            FHN face0 = faces.first;
            Point_2 point0 = dt2.dual(face0);

            //Second face.
            FHN face1 = faces.second;
            Point_2 point1 = dt2.dual(face1);

            float ar_width = compute_distance(point0, point1);
            ar_widths.push_back(ar_width);

            circumcenters_coordinates.push_back(point0);
            circumcenters_coordinates.push_back(point1);
        }
    }

    //Save to file.
    if(!circumcenters_coordinates.empty())
        save_circumcenters_to_vtk_file(circumcenters_coordinates, time_step);

    float ar_length = 0.0f;
    if (circumcenters_coordinates.size())
    {
        std::sort(circumcenters_coordinates.begin(), circumcenters_coordinates.end(), xComparator);
        Point_2 p_0 = circumcenters_coordinates.front();
        Point_2 p_1 = circumcenters_coordinates.back();

        ar_length = compute_distance(p_0, p_1);

        std::cerr << "\nAR length: " << ar_length << endl;
    }

    float ar_width_median = 0.0f;

    if(!ar_widths.empty())
    {
        //save width
        save_width_to_file(ar_widths, time_step);

        ar_width_median = compute_width(ar_widths);
        std::cerr << "\nAR width: " << ar_width_median << endl;
    }

    //For ploting triangulation.
    for(Delaunay::Finite_faces_iterator fi = dt2.finite_faces_begin(); fi != dt2.finite_faces_end(); fi++)
    {
        Point_2 point0 = Point_2(fi->vertex(0)->point().hx(), fi->vertex(0)->point().hy());
        Point_2 point1 = Point_2(fi->vertex(1)->point().hx(), fi->vertex(1)->point().hy());
        Point_2 point2 = Point_2(fi->vertex(2)->point().hx(), fi->vertex(2)->point().hy());

        //Save vertices of triangle.
        triangules_coordinates.push_back(point0);
        triangules_coordinates.push_back(point1);
        triangules_coordinates.push_back(point2);
    }

    if(!triangules_coordinates.empty())
        save_triang_to_vtk_file(triangules_coordinates, time_step);
}

//Save Delaunay triangulation to file.
void save_triang_to_vtk_file(std::vector<Point_2> circumcenters_coordinates, unsigned long time_step)
{
    std::ostringstream oss1;
    oss1 << "ar_deltrang_" << std::setw(6) << std::setfill('0') << time_step << ".vtk";

    std::string file_name1 = oss1.str();

    std::ofstream ofs;
    ofs.open (file_name1, std::ofstream::out | std::ofstream::trunc);

    ofs << "# vtk DataFile Version 4.0" << endl;
    ofs << "vtk output" << endl;
    ofs << "ASCII" << endl;
    ofs << "DATASET UNSTRUCTURED_GRID" << endl;
    ofs << "POINTS " << circumcenters_coordinates.size() << " " << "float" << endl;

    ofs << std::setprecision(2) << std::fixed;

    for(unsigned long p = 0; p < circumcenters_coordinates.size(); p+=3)
    {
        ofs << (float)circumcenters_coordinates[p].x() << " " << (float)circumcenters_coordinates[p].y() << " " << 0 << " "
        << (float)circumcenters_coordinates[p + 1].x() << " " << (float)circumcenters_coordinates[p + 1].y() << " " << 0 << " "
        << (float)circumcenters_coordinates[p + 2].x() << " " << (float)circumcenters_coordinates[p + 2].y() << " " << 0 << " " << endl;
    }

    int nb_points = 3;

    ofs << "CELLS " << circumcenters_coordinates.size()/nb_points << " " << circumcenters_coordinates.size()/nb_points * (nb_points + 1) << endl;

    for(unsigned long c = 0; c < circumcenters_coordinates.size(); c+=3)
    {
        ofs << nb_points << " " << c << " " << c + 1 << " " << c + 2 << endl;
    }

    ofs << endl;

    ofs << "CELL_TYPES" << " " << circumcenters_coordinates.size()/nb_points << endl;

    //Set cell type as a number.
    const int cell_type = 5;

    for(unsigned long ct = 0; ct < circumcenters_coordinates.size()/nb_points; ct++)
    {
        ofs << cell_type << endl;
    }

    ofs.close();
}

//Save skeleton points to VTK file.
void save_circumcenters_to_vtk_file(std::vector<Point_2> circumcenters_coordinates, unsigned long time_step)
{
    std::ostringstream oss;
    oss << "ar_skel_" << std::setw(6) << std::setfill('0') << time_step << ".vtk";

    std::string file_name = oss.str();

    //std::sort(circumcenters_coordinates.begin(), circumcenters_coordinates.end(), xComparator);

    std::ofstream ofs;
    ofs.open (file_name, std::ofstream::out | std::ofstream::trunc);

    ofs << "# vtk DataFile Version 4.0" << endl;
    ofs << "vtk output" << endl;
    ofs << "ASCII" << endl;
    ofs << "DATASET UNSTRUCTURED_GRID" << endl;
    ofs << "POINTS " << circumcenters_coordinates.size() << " " << "float" << endl;

    ofs << std::setprecision(2) << std::fixed;

    for(unsigned long p = 0; p < circumcenters_coordinates.size(); p+=2)
    {
        ofs << (float)circumcenters_coordinates[p].x() << " " << (float)circumcenters_coordinates[p].y() << " " << 0 << " "
        << (float)circumcenters_coordinates[p + 1].x() << " " << (float)circumcenters_coordinates[p + 1].y() << " " << 0 << " " << endl;
    }

    int nb_points = 2;

    ofs << "CELLS " << circumcenters_coordinates.size()/2 << " " << circumcenters_coordinates.size()/2 * (nb_points + 1) << endl;

    for(unsigned long c = 0; c < circumcenters_coordinates.size(); c+=2)
    {
        ofs << 2 << " " << c << " " << c + 1 << endl;
    }

    ofs << endl;

    ofs << "CELL_TYPES" << " " << circumcenters_coordinates.size()/2 << endl;

    //Set cell type as a number.
    const int cell_type = 3;

    for(unsigned long ct = 0; ct < circumcenters_coordinates.size()/2; ct++)
    {
        ofs << cell_type << endl;
    }

    ofs.close();
}


