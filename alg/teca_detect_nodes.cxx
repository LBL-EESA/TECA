#include "teca_detect_nodes.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"

#include <Variable.h>
#include <ThresholdOp.h>
#include <ClosedContourOp.h>
#include <NodeOutputOp.h>
#include <DataArray1D.h>
#include <SimpleGridUtilities.h>
#include <kdtree.h>

#include <iostream>
#include <string>
#include <set>
#include <queue>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

#define TECA_DEBUG 2

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::cos;

// PIMPL idiom hides internals
// defines the API for reduction operators
class teca_detect_nodes::internals_t
{
public:
    internals_t() {}
    ~internals_t() {}

    template<typename T>
    static
    int string_parsing(std::set<std::string> &req_arrays,
                        VariableRegistry &varreg,
                        std::string &cmd,
                        std::vector<T> &vec);

public:
    VariableRegistry varreg;

    std::vector<ClosedContourOp> vec_closed_contour_op;
    std::vector<ClosedContourOp> vec_no_closed_contour_op;
    std::vector<ThresholdOp> vec_threshold_op;
    std::vector<NodeOutputOp> vec_output_op;

    std::string str_search_by;
    bool f_search_by_minima;

    std::set<std::string> dependent_variables;
};

// --------------------------------------------------------------------------
/**
 * Determine if the given field has a closed contour about this point.
 */
template <typename real>
bool has_closed_contour(
    const SimpleGrid &grid,
    const DataArray1D<real> &dataState,
    const int ix0,
    double dDeltaAmt,
    double dDeltaDist,
    double dMinMaxDist)
{
    // Verify arguments
    if (dDeltaAmt == 0.0)
    {
       _EXCEPTIONT("Closed contour amount must be non-zero");
    }
    if (dDeltaDist <= 0.0)
    {
       _EXCEPTIONT("Closed contour distance must be positive");
    }

    // Find min/max near point
    int ixOrigin;

    if (dMinMaxDist == 0.0)
    {
       ixOrigin = ix0;
    }
    // Find a local minimum / maximum
    else
    {
       real dValue;
       float dR;

       FindLocalMinMax<real>(
                              grid,
                              (dDeltaAmt > 0.0),
                              dataState,
                              ix0,
                              dMinMaxDist,
                              ixOrigin,
                              dValue,
                              dR);
    }

    // Set of visited nodes
    std::set<int> setNodesVisited;

    // Set of nodes to visit
    std::queue<int> queueToVisit;
    queueToVisit.push(ixOrigin);

    // Reference value
    real dRefValue = dataState[ixOrigin];

    const double dLat0 = grid.m_dLat[ixOrigin];
    const double dLon0 = grid.m_dLon[ixOrigin];

    Announce(2, "Checking (%lu) : (%1.5f %1.5f)",
       ixOrigin, dLat0, dLon0);

    // Build up nodes
    while (queueToVisit.size() != 0)
    {
       int ix = queueToVisit.front();
       queueToVisit.pop();

       if (setNodesVisited.find(ix) != setNodesVisited.end())
       {
          continue;
       }

       setNodesVisited.insert(ix);

       // Great circle distance to this element
       double dLatThis = grid.m_dLat[ix];
       double dLonThis = grid.m_dLon[ix];

       double dR =
          sin(dLat0) * sin(dLatThis)
          + cos(dLat0) * cos(dLatThis) * cos(dLonThis - dLon0);

       if (dR >= 1.0)
       {
          dR = 0.0;
       }
       else if (dR <= -1.0)
       {
          dR = 180.0;
       }
       else
       {
          dR = 180.0 / M_PI * acos(dR);
       }
       if (dR != dR)
       {
          _EXCEPTIONT("NaN value detected");
       }

       Announce(2, "-- (%lu) : (%1.5f %1.5f) : dx %1.5f",
          ix, dLatThis, dLonThis, dR);

       // Check great circle distance
       if (dR > dDeltaDist)
       {
          Announce(2, "Failed criteria; returning");
          AnnounceEndBlock(2, NULL);
          return false;
       }
       // Verify sufficient increase in value
       if (dDeltaAmt > 0.0)
       {
          if (dataState[ix] - dRefValue >= dDeltaAmt)
          {
             continue;
          }
       }
       // Verify sufficient decrease in value
       else
       {
          if (dRefValue - dataState[ix] >= -dDeltaAmt)
          {
             continue;
          }
       }

       // Add all neighbors of this point
       for (int n = 0; n < grid.m_vecConnectivity[ix].size(); ++n)
       {
          queueToVisit.push(grid.m_vecConnectivity[ix][n]);
       }
    }

    // Report success with criteria
    Announce(2, "Passed criteria; returning");
    AnnounceEndBlock(2, NULL);
    return true;
}

// --------------------------------------------------------------------------
/**
 * Determine if the given field satisfies the threshold.
 */
template <typename real>
bool satisfies_threshold(
    const SimpleGrid &grid,
    const DataArray1D<real> &dataState,
    const int ix0,
    const ThresholdOp::Operation op,
    const double dTargetValue,
    const double dMaxDist)
{
    // Verify that dMaxDist is less than 180.0
    if (dMaxDist > 180.0)
    {
    	_EXCEPTIONT("MaxDist must be less than 180.0");
    }

    // Queue of nodes that remain to be visited
    std::queue<int> queueNodes;
    queueNodes.push(ix0);

    // Set of nodes that have already been visited
    std::set<int> setNodesVisited;

    // Latitude and longitude at the origin
    double dLat0 = grid.m_dLat[ix0];
    double dLon0 = grid.m_dLon[ix0];

    // Loop through all latlon elements
    while (queueNodes.size() != 0)
    {
       int ix = queueNodes.front();
       queueNodes.pop();

       if (setNodesVisited.find(ix) != setNodesVisited.end())
       {
          continue;
       }

       setNodesVisited.insert(ix);

       // Great circle distance to this element
       double dLatThis = grid.m_dLat[ix];
       double dLonThis = grid.m_dLon[ix];

       double dR =
          sin(dLat0) * sin(dLatThis)
          + cos(dLat0) * cos(dLatThis) * cos(dLonThis - dLon0);

       if (dR >= 1.0)
       {
          dR = 0.0;
       }
       else if (dR <= -1.0)
       {
          dR = 180.0;
       }
       else
       {
          dR = 180.0 / M_PI * acos(dR);
       }
       if (dR != dR)
       {
          _EXCEPTIONT("NaN value detected");
       }
       if ((ix != ix0) && (dR > dMaxDist))
       {
          continue;
       }

       // Value at this location
       double dValue = dataState[ix];

       // Apply operator
       if (op == ThresholdOp::GreaterThan)
       {
          if (dValue > dTargetValue)
          {
             return true;
          }
       }
       else if (op == ThresholdOp::LessThan)
       {
          if (dValue < dTargetValue)
          {
             return true;
          }
       }
       else if (op == ThresholdOp::GreaterThanEqualTo)
       {
          if (dValue >= dTargetValue)
          {
             return true;
          }
       }
       else if (op == ThresholdOp::LessThanEqualTo)
       {
          if (dValue <= dTargetValue)
          {
             return true;
          }
       }
       else if (op == ThresholdOp::EqualTo)
       {
          if (dValue == dTargetValue)
          {
             return true;
          }
       } else if (op == ThresholdOp::NotEqualTo)
       {
          if (dValue != dTargetValue)
          {
             return true;
          }
       }
       else
       {
          _EXCEPTIONT("Invalid operation");
       }

       // Special case: zero distance
       if (dMaxDist == 0.0)
       {
          return false;
       }

       // Add all neighbors of this point
       for (int n = 0; n < grid.m_vecConnectivity[ix].size(); ++n)
       {
          queueNodes.push(grid.m_vecConnectivity[ix][n]);
       }
    }

    return false;
}

// --------------------------------------------------------------------------
int teca_detect_nodes::detect_cyclones_unstructured(
    const_p_teca_cartesian_mesh mesh,
    SimpleGrid &grid,
    std::set<int> &set_candidates)
{
    // get coordinate arrays
    const_p_teca_variant_array y = mesh->get_y_coordinates();
    const_p_teca_variant_array x = mesh->get_x_coordinates();

    if (!y || !x)
    {
       TECA_FATAL_ERROR("mesh coordinates are missing.")
       return -1;
    }

    VARIANT_ARRAY_DISPATCH_FP(y.get(),

       DataArray1D<double> vec_lat(y->size(), false);
       auto [sp_y, p_y] = get_cpu_accessible<CTT>(y);
       vec_lat.AttachToData((void*)p_y);

       for (int j = 0; j < y->size(); ++j)
       {
          vec_lat[j] *= M_PI / 180.0;
       }

       assert_type<CTT>(x);
       DataArray1D<double> vec_lon(x->size(), false);
       auto [sp_x, p_x] = get_cpu_accessible<CTT>(x);
       vec_lon.AttachToData((void*)p_x);

       for (int i = 0; i < x->size(); ++i)
       {
          vec_lon[i] *= M_PI / 180.0;
       }

       // No connectivity file; check for latitude/longitude dimension
       if (this->in_connect == "")
       {
          AnnounceStartBlock("Generating RLL grid data");
          grid.GenerateLatitudeLongitude(
                    vec_lat, vec_lon, this->regional, this->diag_connect, true);
          AnnounceEndBlock("Done");
       }
       // Check for connectivity file
       else
       {
          TECA_ERROR("Loading grid data from connectivity file"
              "is not supported")
          return -1;
       }
    )

    // get search_by array
    const_p_teca_variant_array search_by =
       mesh->get_point_arrays()->get(this->internals->str_search_by);

    if (!search_by)
    {
       TECA_FATAL_ERROR("Dataset missing search_by variable \""
           << this->internals->str_search_by << "\"")
       return -1;
    }

    int n_rejected_merge = 0;
    VARIANT_ARRAY_DISPATCH_FP(search_by.get(),

       DataArray1D<float> data_search(search_by->size(), false);
       auto [sp_search_by, p_search_by] = get_cpu_accessible<CTT>(search_by);
       data_search.AttachToData((void*)p_search_by);

       // Tag all minima
       AnnounceStartBlock("FindAllLocalMinMax");
       if (this->search_by_threshold == "")
       {
          if (this->internals->f_search_by_minima)
             FindAllLocalMinima<float>(grid, data_search, set_candidates);
          else
             FindAllLocalMaxima<float>(grid, data_search, set_candidates);
       }
       else
       {
          FindAllLocalMinMaxWithThreshold<float>(
                                              grid,
                                              data_search,
                                              this->internals->f_search_by_minima,
                                              this->search_by_threshold,
                                              set_candidates);
       }
       AnnounceEndBlock("Done");

       // Eliminate based on merge distance
       AnnounceStartBlock("Eliminate based on merge distance");
       if (this->merge_dist != 0.0)
       {
          std::set<int> set_new_candidates;

          // Calculate chord distance
          double d_sph_dist = 2.0 * sin(0.5 * this->merge_dist / 180.0 * M_PI);

          // Create a new KD Tree containing all nodes
          kdtree * kdMerge = kd_create(3);
          if (kdMerge == NULL)
          {
             _EXCEPTIONT("kd_create(3) failed");
          }

          std::set<int>::const_iterator iter_candidate = set_candidates.begin();
          for (; iter_candidate != set_candidates.end(); ++iter_candidate)
          {
             double d_lat = grid.m_dLat[*iter_candidate];
             double d_lon = grid.m_dLon[*iter_candidate];

             double dx = cos(d_lon) * cos(d_lat);
             double dy = sin(d_lon) * cos(d_lat);
             double dz = sin(d_lat);

             kd_insert3(kdMerge, dx, dy, dz, (void*)(&(*iter_candidate)));
          }

          // Loop through all candidates find set of nearest neighbors
          iter_candidate = set_candidates.begin();
          for (; iter_candidate != set_candidates.end(); ++iter_candidate)
          {
             double d_lat = grid.m_dLat[*iter_candidate];
             double d_lon = grid.m_dLon[*iter_candidate];

             double dx = cos(d_lon) * cos(d_lat);
             double dy = sin(d_lon) * cos(d_lat);
             double dz = sin(d_lat);

             // Find all neighbors within d_sph_dist
             kdres * kdresMerge =
                kd_nearest_range3(kdMerge, dx, dy, dz, d_sph_dist);

             // Number of neighbors
             int n_neighbors = kd_res_size(kdresMerge);
             if (n_neighbors == 0)
             {
                set_new_candidates.insert(*iter_candidate);
             }
             else
             {
                double d_value =
                   static_cast<double>(data_search[*iter_candidate]);

                bool f_extrema = true;
                for (;;)
                {
                   int * ppr = (int *)(kd_res_item_data(kdresMerge));

                   if (this->internals->f_search_by_minima)
                   {
                      if (static_cast<double>(data_search[*ppr]) < d_value)
                      {
                         f_extrema = false;
                         break;
                      }

                   }
                   else
                   {
                      if (static_cast<double>(data_search[*ppr]) > d_value)
                      {
                         f_extrema = false;
                         break;
                      }
                   }

                   int i_has_more = kd_res_next(kdresMerge);
                   if (!i_has_more)
                   {
                      break;
                   }
                }

                if (f_extrema)
                {
                   set_new_candidates.insert(*iter_candidate);
                }
                else
                {
                   n_rejected_merge++;
                }
             }

             kd_res_free(kdresMerge);
          }

          // Destroy the KD Tree
          kd_free(kdMerge);

          // Update set of pressure minima
          set_candidates = set_new_candidates;
       }
       AnnounceEndBlock("Done");
    )

    // Eliminate based on interval
    AnnounceStartBlock("Eliminate based on interval");
    int n_rejected_location = 0;
    if ((this->min_lat != this->max_lat) ||
        (this->min_lon != this->max_lon) ||
	     (this->min_abs_lat != 0.0))
    {
       std::set<int> set_new_candidates;

       std::set<int>::const_iterator iter_candidate = set_candidates.begin();
       for (; iter_candidate != set_candidates.end(); ++iter_candidate)
       {
          double d_lat = grid.m_dLat[*iter_candidate];
          double d_lon = grid.m_dLon[*iter_candidate];

          if (this->min_lat != this->max_lat)
          {
             if (d_lat < this->min_lat)
             {
                n_rejected_location++;
                continue;
             }
             if (d_lat > this->max_lat)
             {
                n_rejected_location++;
                continue;
             }
          }
          if (this->min_lon != this->max_lon)
          {
             if (d_lon < 0.0)
             {
                int i_lon_shift = static_cast<int>(d_lon / (2.0 * M_PI));
                d_lon += static_cast<double>(i_lon_shift + 1) * 2.0 * M_PI;
             }
             if (d_lon >= 2.0 * M_PI)
             {
                int i_lon_shift = static_cast<int>(d_lon / (2.0 * M_PI));
                d_lon -= static_cast<double>(i_lon_shift - 1) * 2.0 * M_PI;
             }
             if (this->min_lon < this->max_lon)
             {
                if (d_lon < this->min_lon)
                {
                   n_rejected_location++;
                   continue;
                }
                if (d_lon > this->max_lon)
                {
                   n_rejected_location++;
                   continue;
                }
             }
             else
             {
                if ((d_lon > this->max_lon) &&
                    (d_lon < this->min_lon))
                {
                   n_rejected_location++;
                   continue;
                }
             }
          }
          if (this->min_abs_lat != 0.0)
          {
             if (fabs(d_lat) < this->min_abs_lat)
             {
                n_rejected_location++;
                continue;
             }
          }
          set_new_candidates.insert(*iter_candidate);
       }
       set_candidates = set_new_candidates;
    }
    AnnounceEndBlock("Done");

    // Eliminate based on thresholds
    AnnounceStartBlock("Eliminate based on thresholds");
    DataArray1D<int> vec_rejected_threshold(
                                       this->internals->vec_threshold_op.size());
    for (int tc = 0; tc < this->internals->vec_threshold_op.size(); ++tc)
    {
       std::set<int> set_new_candidates;

       // Load the search variable data
       Variable &var =
          this->internals->varreg.Get(
                                 this->internals->vec_threshold_op[tc].m_varix);
       const_p_teca_variant_array threshold_var =
          mesh->get_point_arrays()->get(var.GetName());

       if (!threshold_var)
       {
          TECA_FATAL_ERROR("Dataset missing variable \""
              << var.GetName() << "\"")
          return -1;
       }

       VARIANT_ARRAY_DISPATCH_FP(threshold_var.get(),

          DataArray1D<float> data_state(threshold_var->size(), false);
          auto [sp_threshold_var, p_threshold_var] = get_cpu_accessible<CTT>(threshold_var);
          data_state.AttachToData((void*)p_threshold_var);

          // Loop through all pressure minima
          std::set<int>::const_iterator iter_candidate = set_candidates.begin();
          for (; iter_candidate != set_candidates.end(); ++iter_candidate)
          {
              // Determine if the threshold is satisfied
              bool f_satisfies_threshold = satisfies_threshold<float>(
                       grid, data_state, *iter_candidate,
                       this->internals->vec_threshold_op[tc].m_eOp,
                       this->internals->vec_threshold_op[tc].m_dValue,
                       this->internals->vec_threshold_op[tc].m_dDistance);

              // If not rejected, add to new pressure minima array
              if (f_satisfies_threshold)
              {
                 set_new_candidates.insert(*iter_candidate);
              }
              else
              {
                 vec_rejected_threshold[tc]++;
              }
          }

          set_candidates = set_new_candidates;
       )
    }
    AnnounceEndBlock("Done");

    // Eliminate based on closed contours
    AnnounceStartBlock("Eliminate based on closed contours");
    DataArray1D<int> vec_rejected_closed_contour(
                                   this->internals->vec_closed_contour_op.size());
    for (int ccc = 0; ccc < this->internals->vec_closed_contour_op.size(); ++ccc)
    {
       std::set<int> set_new_candidates;

       // Load the search variable data
       Variable &var =
          this->internals->varreg.Get(
                            this->internals->vec_closed_contour_op[ccc].m_varix);
       const_p_teca_variant_array closed_contour_var =
          mesh->get_point_arrays()->get(var.GetName());

       if (!closed_contour_var)
       {
          TECA_FATAL_ERROR("Dataset missing variable \""
              << var.GetName() << "\"")
          return -1;
       }

       VARIANT_ARRAY_DISPATCH_FP(closed_contour_var.get(),

          DataArray1D<float> data_state(closed_contour_var->size(), false);
          auto [sp_closed_contour_var, p_closed_contour_var] = get_cpu_accessible<CTT>(closed_contour_var);
          data_state.AttachToData((void*)p_closed_contour_var);

          // Loop through all pressure minima
          std::set<int>::const_iterator iter_candidate = set_candidates.begin();
          for (; iter_candidate != set_candidates.end(); ++iter_candidate)
          {
             // Determine if a closed contour is present
             bool f_has_closed_contour = has_closed_contour<float>(
                    grid, data_state, *iter_candidate,
                    this->internals->vec_closed_contour_op[ccc].m_dDeltaAmount,
                    this->internals->vec_closed_contour_op[ccc].m_dDistance,
                    this->internals->vec_closed_contour_op[ccc].m_dMinMaxDist);

             // If not rejected, add to new pressure minima array
             if (f_has_closed_contour)
             {
                set_new_candidates.insert(*iter_candidate);
             }
             else
             {
                vec_rejected_closed_contour[ccc]++;
             }
          }

          set_candidates = set_new_candidates;
       )
    }
    AnnounceEndBlock("Done");

    // Eliminate based on no closed contours
    AnnounceStartBlock("Eliminate based on no closed contours");
    DataArray1D<int> vec_rejected_no_closed_contour(
                                 this->internals->vec_no_closed_contour_op.size());
    for (int ccc = 0; ccc < this->internals->vec_no_closed_contour_op.size(); ++ccc)
    {
       std::set<int> set_new_candidates;

       // Load the search variable data
       Variable &var =
          this->internals->varreg.Get(
                          this->internals->vec_no_closed_contour_op[ccc].m_varix);
       const_p_teca_variant_array no_closed_contour_var =
          mesh->get_point_arrays()->get(var.GetName());

       if (!no_closed_contour_var)
       {
          TECA_FATAL_ERROR("Dataset missing variable \""
              << var.GetName() << "\"")
          return -1;
       }

       VARIANT_ARRAY_DISPATCH_FP(no_closed_contour_var.get(),

          DataArray1D<float> data_state(no_closed_contour_var->size(), false);
          auto [sp_no_closed_contour_var, p_no_closed_contour_var] = get_cpu_accessible<CTT>(no_closed_contour_var);
          data_state.AttachToData((void*)p_no_closed_contour_var);

          // Loop through all pressure minima
          std::set<int>::const_iterator iter_candidate = set_candidates.begin();
          for (; iter_candidate != set_candidates.end(); ++iter_candidate)
          {
             // Determine if a closed contour is present
             bool f_has_closed_contour = has_closed_contour<float>(
                   grid,
                   data_state,
                   *iter_candidate,
                   this->internals->vec_no_closed_contour_op[ccc].m_dDeltaAmount,
                   this->internals->vec_no_closed_contour_op[ccc].m_dDistance,
                   this->internals->vec_no_closed_contour_op[ccc].m_dMinMaxDist);

             // If a closed contour is present, reject this candidate
             if (f_has_closed_contour)
             {
                vec_rejected_no_closed_contour[ccc]++;
             }
             else
             {
                set_new_candidates.insert(*iter_candidate);
             }
          }

          set_candidates = set_new_candidates;
       )
    }
    AnnounceEndBlock("Done");

    Announce("Total candidates: %i", set_candidates.size());
    Announce("Rejected (  location): %i", n_rejected_location);
    Announce("Rejected (    merged): %i", n_rejected_merge);

    for (int tc = 0; tc < vec_rejected_threshold.GetRows(); ++tc)
    {
       Announce("Rejected (thresh. %s): %i",
          this->internals->varreg.GetVariableString(
                          this->internals->vec_threshold_op[tc].m_varix).c_str(),
                                                     vec_rejected_threshold[tc]);
    }

    for (int ccc = 0; ccc < vec_rejected_closed_contour.GetRows(); ++ccc)
    {
       Announce("Rejected (contour %s): %i",
          this->internals->varreg.GetVariableString(
                     this->internals->vec_closed_contour_op[ccc].m_varix).c_str(),
                                                vec_rejected_closed_contour[ccc]);
    }

    for (int ccc = 0; ccc < vec_rejected_no_closed_contour.GetRows(); ++ccc)
    {
       Announce("Rejected (nocontour %s): %i",
          this->internals->varreg.GetVariableString(
                   this->internals->vec_no_closed_contour_op[ccc].m_varix).c_str(),
                                              vec_rejected_no_closed_contour[ccc]);
    }

    return 0;
}

// --------------------------------------------------------------------------
teca_detect_nodes::teca_detect_nodes() :
    in_connect(""),
    search_by_min(""),
    search_by_max(""),
    closed_contour_cmd(""),
    no_closed_contour_cmd(""),
    threshold_cmd(""),
    output_cmd(""),
    search_by_threshold(""),
    min_lon(0.0),
    max_lon(10.0),
    min_lat(-20.0),
    max_lat(20.0),
    min_abs_lat(0.0),
    merge_dist(6.0),
    diag_connect(false),
    regional(true),
    out_header(true)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    this->internals = new teca_detect_nodes::internals_t;
}

// --------------------------------------------------------------------------
teca_detect_nodes::~teca_detect_nodes()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_detect_nodes::get_properties_description(
    const std::string &prefix, options_description &opts)
{
    options_description ard_opts("Options for "
        + (prefix.empty()?"teca_tc_candidates":prefix));

    ard_opts.add_options()
        TECA_POPTS_GET(std::string, prefix, in_connect, "")
        TECA_POPTS_GET(std::string, prefix, search_by_min, "")
        TECA_POPTS_GET(std::string, prefix, search_by_max, "")
        TECA_POPTS_GET(std::string, prefix, closed_contour_cmd, "")
        TECA_POPTS_GET(std::string, prefix, no_closed_contour_cmd, "")
        TECA_POPTS_GET(std::string, prefix, threshold_cmd, "")
        TECA_POPTS_GET(std::string, prefix, output_cmd, "")
        TECA_POPTS_GET(std::string, prefix, search_by_threshold, "")
        TECA_POPTS_GET(double, prefix, min_lon, "")
        TECA_POPTS_GET(double, prefix, max_lon, "")
        TECA_POPTS_GET(double, prefix, min_lat, "")
        TECA_POPTS_GET(double, prefix, max_lat, "")
        TECA_POPTS_GET(double, prefix, min_abs_lat, "")
        TECA_POPTS_GET(double, prefix, merge_dist, "")
        TECA_POPTS_GET(bool, prefix, diag_connect, "")
        TECA_POPTS_GET(bool, prefix, regional, "")
        TECA_POPTS_GET(bool, prefix, out_header, "")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    opts.add(ard_opts);
}

// --------------------------------------------------------------------------
void teca_detect_nodes::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, in_connect)
    TECA_POPTS_SET(opts, std::string, prefix, search_by_min)
    TECA_POPTS_SET(opts, std::string, prefix, search_by_max)
    TECA_POPTS_SET(opts, std::string, prefix, closed_contour_cmd)
    TECA_POPTS_SET(opts, std::string, prefix, no_closed_contour_cmd)
    TECA_POPTS_SET(opts, std::string, prefix, threshold_cmd)
    TECA_POPTS_SET(opts, std::string, prefix, output_cmd)
    TECA_POPTS_SET(opts, std::string, prefix, search_by_threshold)
    TECA_POPTS_SET(opts, double, prefix, min_lon)
    TECA_POPTS_SET(opts, double, prefix, max_lon)
    TECA_POPTS_SET(opts, double, prefix, min_lat)
    TECA_POPTS_SET(opts, double, prefix, max_lat)
    TECA_POPTS_SET(opts, double, prefix, min_abs_lat)
    TECA_POPTS_SET(opts, double, prefix, merge_dist)
    TECA_POPTS_SET(opts, bool, prefix, diag_connect)
    TECA_POPTS_SET(opts, bool, prefix, regional)
    TECA_POPTS_SET(opts, bool, prefix, out_header)
}
#endif

// --------------------------------------------------------------------------
template<typename T>
int teca_detect_nodes::internals_t::string_parsing(
   std::set<std::string> &dep_vars,
   VariableRegistry &varreg,
   std::string &cmd,
   std::vector<T> &vec)
{
   int i_last = 0;
   for (int i = 0; i <= cmd.length(); ++i)
   {
      if ((i == cmd.length()) ||
          (cmd[i] == ';') ||
          (cmd[i] == ':'))
      {
         std::string strSubStr = cmd.substr(i_last, i - i_last);

         int i_next_op = (int)(vec.size());
         vec.resize(i_next_op + 1);
         vec[i_next_op].Parse(varreg, strSubStr);

         // Load the search variable data
         Variable &var = varreg.Get(vec[i_next_op].m_varix);
         // Get the data directly from a variable
         if (!var.IsOp())
         {
            dep_vars.insert(var.GetName());
         }
         // Evaluate a data operator to get the contents of this variable
         else
         {
            TECA_ERROR("Data operator is not supported")
            return -1;
         }

         i_last = i + 1;
      }
   }
   return 0;
}

// --------------------------------------------------------------------------
int teca_detect_nodes::initialize()
{
    std::set<std::string> dep_vars;

    // Only one of search by min or search by max should be specified
    if ((this->search_by_min == "") && (this->search_by_max == ""))
    {
    	 this->internals->str_search_by = "PSL";
    }
    if ((this->search_by_min != "") && (this->search_by_max != ""))
    {
    	 TECA_ERROR("Only one of --searchbymin or --searchbymax can"
    		" be specified");
       return -1;
    }

    this->internals->f_search_by_minima = true;
    if (this->search_by_min != "")
    {
       this->internals->str_search_by = this->search_by_min;
    	 this->internals->f_search_by_minima = true;
    }
    if (this->search_by_max != "")
    {
       this->internals->str_search_by = this->search_by_max;
    	 this->internals->f_search_by_minima = false;
    }
    dep_vars.insert(this->internals->str_search_by);

    // Parse the closed contour command string
    if (this->closed_contour_cmd != "")
    {
       if (internals_t::string_parsing(dep_vars, this->internals->varreg,
           this->closed_contour_cmd, this->internals->vec_closed_contour_op))
          return -1;
    }

    // Parse the no closed contour command string
    if (this->no_closed_contour_cmd != "")
    {
       if (internals_t::string_parsing(dep_vars, this->internals->varreg,
           this->no_closed_contour_cmd, this->internals->vec_no_closed_contour_op))
          return -1;
    }

    // Parse the threshold operator command string
    if (this->threshold_cmd != "")
    {
       if (internals_t::string_parsing(dep_vars, this->internals->varreg,
           this->threshold_cmd, this->internals->vec_threshold_op))
          return -1;
    }

    // Parse the output operator command string
    if (this->output_cmd != "")
    {
       if (internals_t::string_parsing(dep_vars, this->internals->varreg,
           this->output_cmd, this->internals->vec_output_op))
          return -1;
    }

    this->internals->dependent_variables = std::move(dep_vars);

    this->min_lon *= M_PI / 180.0;
    this->max_lon *= M_PI / 180.0;
    this->min_lat *= M_PI / 180.0;
    this->max_lat *= M_PI / 180.0;
    this->min_abs_lat *= M_PI / 180.0;

    return 0;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_detect_nodes::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &md_in,
    const teca_metadata &req_in)
{
    if (this->get_verbose() > 1)
    {
       std::cerr << teca_parallel_id()
            << "teca_detect_nodes::get_upstream_request" << std::endl;
    }

    (void)port;
    (void)md_in;
    (void)req_in;

    std::vector<teca_metadata> up_reqs;
    teca_metadata md = md_in[0];

    // locate the extents of the user supplied region of
    // interest
    teca_metadata coords;
    if (md.get("coordinates", coords))
    {
       TECA_FATAL_ERROR("metadata is missing \"coordinates\"")
       return up_reqs;
    }

    p_teca_variant_array in_x, in_y, in_z;
    if (!(in_x = coords.get("x")) ||
        !(in_y = coords.get("y")) ||
        !(in_z = coords.get("z")))
    {
       TECA_FATAL_ERROR("metadata missing coordinate arrays")
       return up_reqs;
    }

    unsigned long extent[6] = {0};
    double req_bounds[6] = {0.0};
    req_bounds[0] = this->min_lon * 180.0 / M_PI;
    req_bounds[1] = this->max_lon * 180.0 / M_PI;
    req_bounds[2] = this->min_lat * 180.0 / M_PI;
    req_bounds[3] = this->max_lat * 180.0 / M_PI;

    if (teca_coordinate_util::bounds_to_extent(req_bounds,
            in_x, in_y, in_z, extent) ||
        teca_coordinate_util::validate_extent(in_x->size(),
            in_y->size(), in_z->size(), extent, true))
    {
       TECA_FATAL_ERROR("failed to determine the active extent")
       return up_reqs;
    }

    if (this->get_verbose() > 1)
    {
       cerr << teca_parallel_id() << "active_bound = "
           << this->min_lon<< ", " << this->max_lon
           << ", " << this->min_lat << ", " << this->max_lat
           << endl;
       cerr << teca_parallel_id() << "active_extent = "
           << extent[0] << ", " << extent[1] << ", " << extent[2] << ", "
           << extent[3] << ", " << extent[4] << ", " << extent[5] << endl;
    }

    // get the requested arrays
    std::set<std::string> req_arrays;
    req_in.get("arrays", req_arrays);

    req_arrays.insert(this->internals->dependent_variables.begin(),
         this->internals->dependent_variables.end());

    teca_metadata req(req_in);
    req.set("arrays", req_arrays);
    req.set("extent", extent);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_detect_nodes::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &req_in)
{
    if (this->get_verbose() > 1)
    {
       std::cerr << teca_parallel_id()
            << "teca_detect_nodes::execute" << std::endl;
    }

    (void)port;
    (void)req_in;

    if (!input_data.size())
    {
       TECA_FATAL_ERROR("empty input")
       return nullptr;
    }

    // get the input dataset
    const_p_teca_cartesian_mesh mesh
       = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);
    if (!mesh)
    {
       TECA_FATAL_ERROR("teca_cartesian_mesh is required")
       return nullptr;
    }

    // get time step
    unsigned long time_step;
    mesh->get_time_step(time_step);

    // get temporal offset of the current timestep
    double time_offset = 0.0;
    mesh->get_time(time_offset);

    // get offset units
    std::string time_units;
    mesh->get_time_units(time_units);

    // get offset calendar
    std::string calendar;
    mesh->get_calendar(calendar);

    std::set<int> set_candidates;

    SimpleGrid grid;

    if (this->detect_cyclones_unstructured(mesh, grid, set_candidates))
    {
       TECA_FATAL_ERROR("TC detector encountered an error")
       return nullptr;
    }

    // Write candidate information
    int i_candidate_ix = 0;

    // Apply output operators
    std::vector< std::vector<std::string> > vec_output_value;
    vec_output_value.resize(set_candidates.size());
    for (int i = 0; i < set_candidates.size(); ++i)
    {
       vec_output_value[i].resize(this->internals->vec_output_op.size());
    }

    for (int outc = 0; outc < this->internals->vec_output_op.size(); ++outc)
    {
       Variable &var =
          this->internals->varreg.Get(
                                  this->internals->vec_output_op[outc].m_varix);
       const_p_teca_variant_array output_var =
          mesh->get_point_arrays()->get(var.GetName());

       if (!output_var)
       {
          TECA_FATAL_ERROR("Dataset missing variable \"" <<
              var.GetName() << "\"")
          return nullptr;
       }

       VARIANT_ARRAY_DISPATCH_FP(output_var.get(),

          DataArray1D<float> data_state(output_var->size(), false);
          auto [sp_output_var, p_output_var] = get_cpu_accessible<CTT>(output_var);
          data_state.AttachToData((void*)p_output_var);

          // Loop through all pressure minima
          i_candidate_ix = 0;
          std::set<int>::const_iterator iter_candidate = set_candidates.begin();
          for (; iter_candidate != set_candidates.end(); ++iter_candidate)
          {
             ApplyNodeOutputOp<float>(
                    this->internals->vec_output_op[outc],
                    grid,
                    data_state,
                    *iter_candidate,
                    vec_output_value[i_candidate_ix][outc]);

             i_candidate_ix++;
          }
       )
    }

    // Output all candidates
    i_candidate_ix = 0;

    double lat_array[set_candidates.size()];
    double lon_array[set_candidates.size()];
    int i_array[set_candidates.size()];
    int j_array[set_candidates.size()];

    std::set<int>::const_iterator iter_candidate = set_candidates.begin();
    for (; iter_candidate != set_candidates.end(); ++iter_candidate)
    {
       if (grid.m_nGridDim.size() == 1)
       {
          i_array[i_candidate_ix] = *iter_candidate;
       }
       else if (grid.m_nGridDim.size() == 2)
       {
          i_array[i_candidate_ix] =
             (*iter_candidate) % static_cast<int>(grid.m_nGridDim[1]);
          j_array[i_candidate_ix] =
             (*iter_candidate) / static_cast<int>(grid.m_nGridDim[1]);
       }
       lat_array[i_candidate_ix] = grid.m_dLat[*iter_candidate] * 180.0 / M_PI;
       lon_array[i_candidate_ix] = grid.m_dLon[*iter_candidate] * 180.0 / M_PI;

       i_candidate_ix++;
    }

    // build the output
    p_teca_table out_table = teca_table::New();
    out_table->set_calendar(calendar);
    out_table->set_time_units(time_units);

    // add time stamp
    out_table->declare_columns("step", long(), "time", double());
    for (unsigned long i = 0; i < set_candidates.size(); ++i)
       out_table << time_step << time_offset;

    // put the arrays into the table
    p_teca_variant_array_impl<double> latitude =
       teca_variant_array_impl<double>::New(set_candidates.size(), lat_array);
    p_teca_variant_array_impl<double> longitude =
       teca_variant_array_impl<double>::New(set_candidates.size(), lon_array);
    p_teca_variant_array_impl<int> i =
       teca_variant_array_impl<int>::New(set_candidates.size(), i_array);

    out_table->append_column("i", i);
    if (grid.m_nGridDim.size() == 2)
    {
       p_teca_variant_array_impl<int> j =
          teca_variant_array_impl<int>::New(set_candidates.size(), j_array);
       out_table->append_column("j", j);
    }
    out_table->append_column("lat", latitude);
    out_table->append_column("lon", longitude);

    for (int outc = 0; outc < this->internals->vec_output_op.size(); ++outc)
    {
       i_candidate_ix = 0;
       std::string output_array[set_candidates.size()];
       std::set<int>::const_iterator iter_candidate = set_candidates.begin();
       for (; iter_candidate != set_candidates.end(); ++iter_candidate)
       {
          output_array[i_candidate_ix] = vec_output_value[i_candidate_ix][outc];
          i_candidate_ix++;
       }

       p_teca_variant_array_impl<std::string> output =
          teca_variant_array_impl<std::string>::New(set_candidates.size(),
                                                                output_array);

       Variable &var = this->internals->varreg.Get(
          this->internals->vec_output_op[outc].m_varix);
       out_table->append_column(var.ToString(this->internals->varreg).c_str(),
                                                                      output);
    }

#if TECA_DEBUG > 1
    out_table->to_stream(cerr);
    cerr << std::endl;
#endif

    return out_table;
}
