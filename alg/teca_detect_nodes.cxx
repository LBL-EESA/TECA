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

public:
    VariableRegistry varreg;

    std::vector<ClosedContourOp> vecClosedContourOp;
    std::vector<ClosedContourOp> vecNoClosedContourOp;
    std::vector<ThresholdOp> vecThresholdOp;
    std::vector<NodeOutputOp> vecOutputOp;

    std::string strSearchBy;
    bool fSearchByMinima;
};

// --------------------------------------------------------------------------
/**
 * Determine if the given field has a closed contour about this point.
 */
template <typename real>
bool HasClosedContour(
    const SimpleGrid & grid,
    const DataArray1D<real> & dataState,
    const int ix0,
    double dDeltaAmt,
    double dDeltaDist,
    double dMinMaxDist)
{
//   dDeltaAmt = 0.0;

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
       for (int n = 0; n < grid.m_vecConnectivity[ix].size(); n++)
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
bool SatisfiesThreshold(
    const SimpleGrid & grid,
    const DataArray1D<real> & dataState,
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
       for (int n = 0; n < grid.m_vecConnectivity[ix].size(); n++)
       {
          queueNodes.push(grid.m_vecConnectivity[ix][n]);
       }
    }

    return false;
}

// --------------------------------------------------------------------------
int teca_detect_nodes::DetectCyclonesUnstructured(
    const_p_teca_cartesian_mesh mesh,
    SimpleGrid & grid,
    std::set<int> & setCandidates)
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

       DataArray1D<double> vecLat(y->size(), false);
       auto [sp_y, p_y] = get_cpu_accessible<CTT>(y);
       vecLat.AttachToData((void*)p_y);

       for (int j = 0; j < y->size(); j++)
       {
          vecLat[j] *= M_PI / 180.0;
       }

       assert_type<CTT>(x);
       DataArray1D<double> vecLon(x->size(), false);
       auto [sp_x, p_x] = get_cpu_accessible<CTT>(x);
       vecLon.AttachToData((void*)p_x);

       for (int i = 0; i < x->size(); i++)
       {
          vecLon[i] *= M_PI / 180.0;
       }

       // No connectivity file; check for latitude/longitude dimension
       if (this->in_connect == "")
       {
          AnnounceStartBlock("Generating RLL grid data");
          grid.GenerateLatitudeLongitude(
                    vecLat, vecLon, this->regional, this->diag_connect, true);
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

    this->minlon *= M_PI / 180.0;
    this->maxlon *= M_PI / 180.0;
    this->minlat *= M_PI / 180.0;
    this->maxlat *= M_PI / 180.0;
    this->minabslat *= M_PI / 180.0;

    // get SearchBy array
    const_p_teca_variant_array SearchBy =
       mesh->get_point_arrays()->get(this->internals->strSearchBy);

    if (!SearchBy)
    {
       TECA_FATAL_ERROR("Dataset missing SearchBy variable \""
           << this->internals->strSearchBy << "\"")
       return -1;
    }

    int nRejectedMerge = 0;
    VARIANT_ARRAY_DISPATCH_FP(SearchBy.get(),

       DataArray1D<float> dataSearch(SearchBy->size(), false);
       auto [sp_SearchBy, p_SearchBy] = get_cpu_accessible<CTT>(SearchBy);
       dataSearch.AttachToData((void*)p_SearchBy);

       // Tag all minima
       AnnounceStartBlock("FindAllLocalMinMax");
       if (this->searchbythreshold == "")
       {
          if (this->internals->fSearchByMinima)
             FindAllLocalMinima<float>(grid, dataSearch, setCandidates);
          else
             FindAllLocalMaxima<float>(grid, dataSearch, setCandidates);
       }
       else
       {
          FindAllLocalMinMaxWithThreshold<float>(
                                              grid,
                                              dataSearch,
                                              this->internals->fSearchByMinima,
                                              this->searchbythreshold,
                                              setCandidates);
       }
       AnnounceEndBlock("Done");

       // Eliminate based on merge distance
       AnnounceStartBlock("Eliminate based on merge distance");
       if (this->mergedist != 0.0)
       {
          std::set<int> setNewCandidates;

          // Calculate chord distance
          double dSphDist = 2.0 * sin(0.5 * this->mergedist / 180.0 * M_PI);

          // Create a new KD Tree containing all nodes
          kdtree * kdMerge = kd_create(3);
          if (kdMerge == NULL)
          {
             _EXCEPTIONT("kd_create(3) failed");
          }

          std::set<int>::const_iterator iterCandidate = setCandidates.begin();
          for (; iterCandidate != setCandidates.end(); iterCandidate++)
          {
             double dLat = grid.m_dLat[*iterCandidate];
             double dLon = grid.m_dLon[*iterCandidate];

             double dX = cos(dLon) * cos(dLat);
             double dY = sin(dLon) * cos(dLat);
             double dZ = sin(dLat);

             kd_insert3(kdMerge, dX, dY, dZ, (void*)(&(*iterCandidate)));
          }

          // Loop through all candidates find set of nearest neighbors
          iterCandidate = setCandidates.begin();
          for (; iterCandidate != setCandidates.end(); iterCandidate++)
          {
             double dLat = grid.m_dLat[*iterCandidate];
             double dLon = grid.m_dLon[*iterCandidate];

             double dX = cos(dLon) * cos(dLat);
             double dY = sin(dLon) * cos(dLat);
             double dZ = sin(dLat);

             // Find all neighbors within dSphDist
             kdres * kdresMerge =
                kd_nearest_range3(kdMerge, dX, dY, dZ, dSphDist);

             // Number of neighbors
             int nNeighbors = kd_res_size(kdresMerge);
             if (nNeighbors == 0)
             {
                setNewCandidates.insert(*iterCandidate);
             }
             else
             {
                double dValue =
                   static_cast<double>(dataSearch[*iterCandidate]);

                bool fExtrema = true;
                for (;;)
                {
                   int * ppr = (int *)(kd_res_item_data(kdresMerge));

                   if (this->internals->fSearchByMinima)
                   {
                      if (static_cast<double>(dataSearch[*ppr]) < dValue)
                      {
                         fExtrema = false;
                         break;
                      }

                   }
                   else
                   {
                      if (static_cast<double>(dataSearch[*ppr]) > dValue)
                      {
                         fExtrema = false;
                         break;
                      }
                   }

                   int iHasMore = kd_res_next(kdresMerge);
                   if (!iHasMore)
                   {
                      break;
                   }
                }

                if (fExtrema)
                {
                   setNewCandidates.insert(*iterCandidate);
                }
                else
                {
                   nRejectedMerge++;
                }
             }

             kd_res_free(kdresMerge);
          }

          // Destroy the KD Tree
          kd_free(kdMerge);

          // Update set of pressure minima
          setCandidates = setNewCandidates;
       }
       AnnounceEndBlock("Done");
    )

    // Eliminate based on interval
    AnnounceStartBlock("Eliminate based on interval");
    int nRejectedLocation = 0;
    if ((this->minlat != this->maxlat) ||
        (this->minlon != this->maxlon) ||
	     (this->minabslat != 0.0))
    {
       std::set<int> setNewCandidates;

       std::set<int>::const_iterator iterCandidate = setCandidates.begin();
       for (; iterCandidate != setCandidates.end(); iterCandidate++)
       {
          double dLat = grid.m_dLat[*iterCandidate];
          double dLon = grid.m_dLon[*iterCandidate];

          if (this->minlat != this->maxlat)
          {
             if (dLat < this->minlat)
             {
                nRejectedLocation++;
                continue;
             }
             if (dLat > this->maxlat)
             {
                nRejectedLocation++;
                continue;
             }
          }
          if (this->minlon != this->maxlon)
          {
             if (dLon < 0.0)
             {
                int iLonShift = static_cast<int>(dLon / (2.0 * M_PI));
                dLon += static_cast<double>(iLonShift + 1) * 2.0 * M_PI;
             }
             if (dLon >= 2.0 * M_PI)
             {
                int iLonShift = static_cast<int>(dLon / (2.0 * M_PI));
                dLon -= static_cast<double>(iLonShift - 1) * 2.0 * M_PI;
             }
             if (this->minlon < this->maxlon)
             {
                if (dLon < this->minlon)
                {
                   nRejectedLocation++;
                   continue;
                }
                if (dLon > this->maxlon)
                {
                   nRejectedLocation++;
                   continue;
                }
             }
             else
             {
                if ((dLon > this->maxlon) &&
                    (dLon < this->minlon))
                {
                   nRejectedLocation++;
                   continue;
                }
             }
          }
          if (this->minabslat != 0.0)
          {
             if (fabs(dLat) < this->minabslat)
             {
                nRejectedLocation++;
                continue;
             }
          }
          setNewCandidates.insert(*iterCandidate);
       }
       setCandidates = setNewCandidates;
    }
    AnnounceEndBlock("Done");

    // Eliminate based on thresholds
    AnnounceStartBlock("Eliminate based on thresholds");
    DataArray1D<int> vecRejectedThreshold(
                                       this->internals->vecThresholdOp.size());
    for (int tc = 0; tc < this->internals->vecThresholdOp.size(); tc++)
    {
       std::set<int> setNewCandidates;

       // Load the search variable data
       Variable & var =
          this->internals->varreg.Get(
                                 this->internals->vecThresholdOp[tc].m_varix);
       const_p_teca_variant_array ThresholdVar =
          mesh->get_point_arrays()->get(var.GetName());

       if (!ThresholdVar)
       {
          TECA_FATAL_ERROR("Dataset missing variable \""
              << var.GetName() << "\"")
          return -1;
       }

       VARIANT_ARRAY_DISPATCH_FP(ThresholdVar.get(),

          DataArray1D<float> dataState(ThresholdVar->size(), false);
          auto [sp_ThresholdVar, p_ThresholdVar] = get_cpu_accessible<CTT>(ThresholdVar);
          dataState.AttachToData((void*)p_ThresholdVar);

          // Loop through all pressure minima
          std::set<int>::const_iterator iterCandidate = setCandidates.begin();
          for (; iterCandidate != setCandidates.end(); iterCandidate++)
          {
              // Determine if the threshold is satisfied
              bool fSatisfiesThreshold = SatisfiesThreshold<float>(
                       grid, dataState, *iterCandidate,
                       this->internals->vecThresholdOp[tc].m_eOp,
                       this->internals->vecThresholdOp[tc].m_dValue,
                       this->internals->vecThresholdOp[tc].m_dDistance);

              // If not rejected, add to new pressure minima array
              if (fSatisfiesThreshold)
              {
                 setNewCandidates.insert(*iterCandidate);
              }
              else
              {
                 vecRejectedThreshold[tc]++;
              }
          }

          setCandidates = setNewCandidates;
       )
    }
    AnnounceEndBlock("Done");

    // Eliminate based on closed contours
    AnnounceStartBlock("Eliminate based on closed contours");
    DataArray1D<int> vecRejectedClosedContour(
                                   this->internals->vecClosedContourOp.size());
    for (int ccc = 0; ccc < this->internals->vecClosedContourOp.size(); ccc++)
    {
       std::set<int> setNewCandidates;

       // Load the search variable data
       Variable & var =
          this->internals->varreg.Get(
                            this->internals->vecClosedContourOp[ccc].m_varix);
       const_p_teca_variant_array ClosedContourVar =
          mesh->get_point_arrays()->get(var.GetName());

       if (!ClosedContourVar)
       {
          TECA_FATAL_ERROR("Dataset missing variable \""
              << var.GetName() << "\"")
          return -1;
       }

       VARIANT_ARRAY_DISPATCH_FP(ClosedContourVar.get(),

          DataArray1D<float> dataState(ClosedContourVar->size(), false);
          auto [sp_ClosedContourVar, p_ClosedContourVar] = get_cpu_accessible<CTT>(ClosedContourVar);
          dataState.AttachToData((void*)p_ClosedContourVar);

          // Loop through all pressure minima
          std::set<int>::const_iterator iterCandidate = setCandidates.begin();
          for (; iterCandidate != setCandidates.end(); iterCandidate++)
          {
             // Determine if a closed contour is present
             bool fHasClosedContour = HasClosedContour<float>(
                    grid, dataState, *iterCandidate,
                    this->internals->vecClosedContourOp[ccc].m_dDeltaAmount,
                    this->internals->vecClosedContourOp[ccc].m_dDistance,
                    this->internals->vecClosedContourOp[ccc].m_dMinMaxDist);

             // If not rejected, add to new pressure minima array
             if (fHasClosedContour)
             {
                setNewCandidates.insert(*iterCandidate);
             }
             else
             {
                vecRejectedClosedContour[ccc]++;
             }
          }

          setCandidates = setNewCandidates;
       )
    }
    AnnounceEndBlock("Done");

    // Eliminate based on no closed contours
    AnnounceStartBlock("Eliminate based on no closed contours");
    DataArray1D<int> vecRejectedNoClosedContour(
                                 this->internals->vecNoClosedContourOp.size());
    for (int ccc = 0; ccc < this->internals->vecNoClosedContourOp.size(); ccc++)
    {
       std::set<int> setNewCandidates;

       // Load the search variable data
       Variable & var =
          this->internals->varreg.Get(
                          this->internals->vecNoClosedContourOp[ccc].m_varix);
       const_p_teca_variant_array NoClosedContourVar =
          mesh->get_point_arrays()->get(var.GetName());

       if (!NoClosedContourVar)
       {
          TECA_FATAL_ERROR("Dataset missing variable \""
              << var.GetName() << "\"")
          return -1;
       }

       VARIANT_ARRAY_DISPATCH_FP(NoClosedContourVar.get(),

          DataArray1D<float> dataState(NoClosedContourVar->size(), false);
          auto [sp_NoClosedContourVar, p_NoClosedContourVar] = get_cpu_accessible<CTT>(NoClosedContourVar);
          dataState.AttachToData((void*)p_NoClosedContourVar);

          // Loop through all pressure minima
          std::set<int>::const_iterator iterCandidate = setCandidates.begin();
          for (; iterCandidate != setCandidates.end(); iterCandidate++)
          {
             // Determine if a closed contour is present
             bool fHasClosedContour = HasClosedContour<float>(
                   grid,
                   dataState,
                   *iterCandidate,
                   this->internals->vecNoClosedContourOp[ccc].m_dDeltaAmount,
                   this->internals->vecNoClosedContourOp[ccc].m_dDistance,
                   this->internals->vecNoClosedContourOp[ccc].m_dMinMaxDist);

             // If a closed contour is present, reject this candidate
             if (fHasClosedContour)
             {
                vecRejectedNoClosedContour[ccc]++;
             }
             else
             {
                setNewCandidates.insert(*iterCandidate);
             }
          }

          setCandidates = setNewCandidates;
       )
    }
    AnnounceEndBlock("Done");

    Announce("Total candidates: %i", setCandidates.size());
    Announce("Rejected (  location): %i", nRejectedLocation);
    Announce("Rejected (    merged): %i", nRejectedMerge);

    for (int tc = 0; tc < vecRejectedThreshold.GetRows(); tc++)
    {
       Announce("Rejected (thresh. %s): %i",
          this->internals->varreg.GetVariableString(
                          this->internals->vecThresholdOp[tc].m_varix).c_str(),
                                                     vecRejectedThreshold[tc]);
    }

    for (int ccc = 0; ccc < vecRejectedClosedContour.GetRows(); ccc++)
    {
       Announce("Rejected (contour %s): %i",
          this->internals->varreg.GetVariableString(
                     this->internals->vecClosedContourOp[ccc].m_varix).c_str(),
                                                vecRejectedClosedContour[ccc]);
    }

    for (int ccc = 0; ccc < vecRejectedNoClosedContour.GetRows(); ccc++)
    {
       Announce("Rejected (nocontour %s): %i",
          this->internals->varreg.GetVariableString(
                   this->internals->vecNoClosedContourOp[ccc].m_varix).c_str(),
                                              vecRejectedNoClosedContour[ccc]);
    }

    return 0;
}

// --------------------------------------------------------------------------
teca_detect_nodes::teca_detect_nodes() :
    in_connect(""),
    searchbymin(""),
    searchbymax(""),
    closedcontourcmd(""),
    noclosedcontourcmd(""),
    thresholdcmd(""),
    outputcmd(""),
//    outputcmd("MSL,min,0;_VECMAG(VAR_10U,VAR_10V),max,2;ZS,min,0"),
    searchbythreshold(""),
    minlon(0.0),
    maxlon(10.0),
    minlat(-20.0),
    maxlat(20.0),
    minabslat(0.0),
    mergedist(6.0),
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
        TECA_POPTS_GET(std::string, prefix, searchbymin, "")
        TECA_POPTS_GET(std::string, prefix, searchbymax, "")
        TECA_POPTS_GET(std::string, prefix, closedcontourcmd, "")
        TECA_POPTS_GET(std::string, prefix, noclosedcontourcmd, "")
        TECA_POPTS_GET(std::string, prefix, thresholdcmd, "")
        TECA_POPTS_GET(std::string, prefix, outputcmd, "")
        TECA_POPTS_GET(std::string, prefix, searchbythreshold, "")
        TECA_POPTS_GET(double, prefix, minlon, "")
        TECA_POPTS_GET(double, prefix, maxlon, "")
        TECA_POPTS_GET(double, prefix, minlat, "")
        TECA_POPTS_GET(double, prefix, maxlat, "")
        TECA_POPTS_GET(double, prefix, minabslat, "")
        TECA_POPTS_GET(double, prefix, mergedist, "")
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
    TECA_POPTS_SET(opts, std::string, prefix, searchbymin)
    TECA_POPTS_SET(opts, std::string, prefix, searchbymax)
    TECA_POPTS_SET(opts, std::string, prefix, closedcontourcmd)
    TECA_POPTS_SET(opts, std::string, prefix, noclosedcontourcmd)
    TECA_POPTS_SET(opts, std::string, prefix, thresholdcmd)
    TECA_POPTS_SET(opts, std::string, prefix, outputcmd)
    TECA_POPTS_SET(opts, std::string, prefix, searchbythreshold)
    TECA_POPTS_SET(opts, double, prefix, minlon)
    TECA_POPTS_SET(opts, double, prefix, maxlon)
    TECA_POPTS_SET(opts, double, prefix, minlat)
    TECA_POPTS_SET(opts, double, prefix, maxlat)
    TECA_POPTS_SET(opts, double, prefix, minabslat)
    TECA_POPTS_SET(opts, double, prefix, mergedist)
    TECA_POPTS_SET(opts, bool, prefix, diag_connect)
    TECA_POPTS_SET(opts, bool, prefix, regional)
    TECA_POPTS_SET(opts, bool, prefix, out_header)
}
#endif

// --------------------------------------------------------------------------
int teca_detect_nodes::get_active_extent(const const_p_teca_variant_array &lat,
    const const_p_teca_variant_array &lon, std::vector<unsigned long> &extent) const
{
    extent = {1, 0, 1, 0, 0, 0};

    unsigned long high_i = lon->size() - 1;
    if (this->minlon > this->maxlon)
    {
       extent[0] = 0l;
       extent[1] = high_i;
    }
    else
    {
       VARIANT_ARRAY_DISPATCH_FP(lon.get(),

          auto [sp_lon, p_lon] = get_cpu_accessible<CTT>(lon);

          if (teca_coordinate_util::index_of(p_lon, 0, high_i, static_cast<NT>(this->minlon), false, extent[0]) ||
              teca_coordinate_util::index_of(p_lon, 0, high_i, static_cast<NT>(this->maxlon), true, extent[1]))
          {
              TECA_ERROR(
                  << "requested longitude ["
                  << this->minlon << ", " << this->maxlon << ", "
                  << "] is not contained in the current dataset bounds ["
                  << p_lon[0] << ", " << p_lon[high_i] << "]")
              return -1;
          }
       )
    }
    if (extent[0] > extent[1])
    {
       TECA_ERROR("invalid longitude coordinate array type")
       return -1;
    }

    unsigned long high_j = lat->size() - 1;
    if (this->minlat > this->maxlat)
    {
       extent[2] = 0l;
       extent[3] = high_j;
    }
    else
    {
       VARIANT_ARRAY_DISPATCH_FP(lat.get(),

          auto [sp_lat, p_lat] = get_cpu_accessible<CTT>(lat);

          if (teca_coordinate_util::index_of(p_lat, 0, high_j, static_cast<NT>(this->minlat), false, extent[2]) ||
              teca_coordinate_util::index_of(p_lat, 0, high_j, static_cast<NT>(this->maxlat), true, extent[3]))
          {
             TECA_ERROR(
                   << "requested latitude ["
                   << this->minlat << ", " << this->maxlat
                   << "] is not contained in the current dataset bounds ["
                   << p_lat[0] << ", " << p_lat[high_j] << "]")
             return -1;
          }
       )
    }
    if (extent[2] > extent[3])
    {
       TECA_ERROR("invalid latitude coordinate array type")
       return -1;
    }

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

    p_teca_variant_array lat;
    p_teca_variant_array lon;
    if (!(lat = coords.get("y")) || !(lon = coords.get("x")))
    {
       TECA_FATAL_ERROR("metadata missing lat lon coordinates")
       return up_reqs;
    }

    std::vector<unsigned long> extent(6, 0l);
    if (this->get_active_extent(lat, lon, extent))
    {
       TECA_FATAL_ERROR("failed to determine the active extent")
       return up_reqs;
    }

#if TECA_DEBUG > 1
    cerr << teca_parallel_id() << "active_bound = "
        << this->minlon<< ", " << this->maxlon
        << ", " << this->minlat << ", " << this->maxlat
        << endl;
    cerr << teca_parallel_id() << "active_extent = "
        << extent[0] << ", " << extent[1] << ", " << extent[2] << ", "
        << extent[3] << ", " << extent[4] << ", " << extent[5] << endl;
#endif

    // get the requested arrays
    std::set<std::string> req_arrays;
    req_in.get("arrays", req_arrays);

    // Only one of search by min or search by max should be specified
    if ((this->searchbymin == "") && (this->searchbymax == ""))
    {
    	 this->internals->strSearchBy = "PSL";
    }
    if ((this->searchbymin != "") && (this->searchbymax != ""))
    {
    	 TECA_ERROR("Only one of --searchbymin or --searchbymax can"
    		" be specified");
       return up_reqs;
    }

    this->internals->fSearchByMinima = true;
    if (this->searchbymin != "")
    {
       this->internals->strSearchBy = this->searchbymin;
    	 this->internals->fSearchByMinima = true;
    }
    if (this->searchbymax != "")
    {
       this->internals->strSearchBy = this->searchbymax;
    	 this->internals->fSearchByMinima = false;
    }
    req_arrays.insert(this->internals->strSearchBy);

    // Parse the closed contour command string
    if (this->closedcontourcmd != "")
    {
       int iLast = 0;
       for (int i = 0; i <= this->closedcontourcmd.length(); i++)
       {
          if ((i == this->closedcontourcmd.length()) ||
              (this->closedcontourcmd[i] == ';') ||
              (this->closedcontourcmd[i] == ':'))
          {
             std::string strSubStr =
                this->closedcontourcmd.substr(iLast, i - iLast);

             int iNextOp = (int)(this->internals->vecClosedContourOp.size());
             this->internals->vecClosedContourOp.resize(iNextOp + 1);
             this->internals->vecClosedContourOp[iNextOp].Parse(
                                          this->internals->varreg, strSubStr);

             // Load the search variable data
             Variable & var =
                this->internals->varreg.Get(
                        this->internals->vecClosedContourOp[iNextOp].m_varix);
             // Get the data directly from a variable
             if (!var.IsOp())
             {
                req_arrays.insert(var.GetName());
             }
             // Evaluate a data operator to get the contents of this variable
             else
             {
                TECA_ERROR("Data operator is not supported")
             }

             iLast = i + 1;
          }
       }
    }

    // Parse the no closed contour command string
    if (this->noclosedcontourcmd != "")
    {
       int iLast = 0;
       for (int i = 0; i <= noclosedcontourcmd.length(); i++)
       {
          if ((i == this->noclosedcontourcmd.length()) ||
              (noclosedcontourcmd[i] == ';') ||
              (noclosedcontourcmd[i] == ':'))
          {
             std::string strSubStr =
                this->noclosedcontourcmd.substr(iLast, i - iLast);

             int iNextOp = (int)(this->internals->vecNoClosedContourOp.size());
             this->internals->vecNoClosedContourOp.resize(iNextOp + 1);
             this->internals->vecNoClosedContourOp[iNextOp].Parse(
                                          this->internals->varreg, strSubStr);

             // Load the search variable data
             Variable & var =
                this->internals->varreg.Get(
                      this->internals->vecNoClosedContourOp[iNextOp].m_varix);
             // Get the data directly from a variable
             if (!var.IsOp())
             {
                req_arrays.insert(var.GetName());
             }
             // Evaluate a data operator to get the contents of this variable
             else
             {
                TECA_ERROR("Data operator is not supported")
             }

             iLast = i + 1;
          }
       }
    }

    // Parse the threshold operator command string
    if (this->thresholdcmd != "")
    {
       int iLast = 0;
       for (int i = 0; i <= this->thresholdcmd.length(); i++)
       {
          if ((i == this->thresholdcmd.length()) ||
              (this->thresholdcmd[i] == ';') ||
              (this->thresholdcmd[i] == ':'))
          {
             std::string strSubStr =
                this->thresholdcmd.substr(iLast, i - iLast);

             int iNextOp = (int)(this->internals->vecThresholdOp.size());
             this->internals->vecThresholdOp.resize(iNextOp + 1);
             this->internals->vecThresholdOp[iNextOp].Parse(
                                          this->internals->varreg, strSubStr);

             // Load the search variable data
             Variable & var =
                this->internals->varreg.Get(
                            this->internals->vecThresholdOp[iNextOp].m_varix);
             // Get the data directly from a variable
             if (!var.IsOp())
             {
                req_arrays.insert(var.GetName());
             }
             // Evaluate a data operator to get the contents of this variable
             else
             {
                TECA_ERROR("Data operator is not supported")
             }

             iLast = i + 1;
          }
       }
    }

    // Parse the output operator command string
    if (outputcmd != "")
    {
       int iLast = 0;
       for (int i = 0; i <= this->outputcmd.length(); i++)
       {
          if ((i == this->outputcmd.length()) ||
              (this->outputcmd[i] == ';') ||
              (this->outputcmd[i] == ':'))
          {
             std::string strSubStr = this->outputcmd.substr(iLast, i - iLast);

             int iNextOp = (int)(this->internals->vecOutputOp.size());
             this->internals->vecOutputOp.resize(iNextOp + 1);
             this->internals->vecOutputOp[iNextOp].Parse(
                                          this->internals->varreg, strSubStr);

             // Load the search variable data
             Variable & var =
                this->internals->varreg.Get(
                               this->internals->vecOutputOp[iNextOp].m_varix);
             // Get the data directly from a variable
             if (!var.IsOp())
             {
                req_arrays.insert(var.GetName());
             }
             // Evaluate a data operator to get the contents of this variable
             else
             {
                TECA_ERROR("Data operator is not supported")
             }

             iLast = i + 1;
          }
       }
    }

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

    std::set<int> setCandidates;

    SimpleGrid grid;

    if (this->DetectCyclonesUnstructured(mesh, grid, setCandidates))
    {
       TECA_FATAL_ERROR("TC detector encountered an error")
       return nullptr;
    }

    // Write candidate information
    int iCandidateIx = 0;

    // Apply output operators
    std::vector< std::vector<std::string> > vecOutputValue;
    vecOutputValue.resize(setCandidates.size());
    for (int i = 0; i < setCandidates.size(); i++)
    {
       vecOutputValue[i].resize(this->internals->vecOutputOp.size());
    }

    for (int outc = 0; outc < this->internals->vecOutputOp.size(); outc++)
    {
       Variable & var =
          this->internals->varreg.Get(
                                  this->internals->vecOutputOp[outc].m_varix);
       const_p_teca_variant_array OutputVar =
          mesh->get_point_arrays()->get(var.GetName());

       if (!OutputVar)
       {
          TECA_FATAL_ERROR("Dataset missing variable \"" <<
              var.GetName() << "\"")
          return nullptr;
       }

       VARIANT_ARRAY_DISPATCH_FP(OutputVar.get(),

          DataArray1D<float> dataState(OutputVar->size(), false);
          auto [sp_OutputVar, p_OutputVar] = get_cpu_accessible<CTT>(OutputVar);
          dataState.AttachToData((void*)p_OutputVar);

          // Loop through all pressure minima
          iCandidateIx = 0;
          std::set<int>::const_iterator iterCandidate = setCandidates.begin();
          for (; iterCandidate != setCandidates.end(); iterCandidate++)
          {
             ApplyNodeOutputOp<float>(
                    this->internals->vecOutputOp[outc],
                    grid,
                    dataState,
                    *iterCandidate,
                    vecOutputValue[iCandidateIx][outc]);

             iCandidateIx++;
          }
       )
    }

    // Output all candidates
    iCandidateIx = 0;

    double lat_array[setCandidates.size()];
    double lon_array[setCandidates.size()];
    int i_array[setCandidates.size()];
    int j_array[setCandidates.size()];

    std::set<int>::const_iterator iterCandidate = setCandidates.begin();
    for (; iterCandidate != setCandidates.end(); iterCandidate++)
    {
       if (grid.m_nGridDim.size() == 1)
       {
          i_array[iCandidateIx] = *iterCandidate;
       }
       else if (grid.m_nGridDim.size() == 2)
       {
          i_array[iCandidateIx] =
             (*iterCandidate) % static_cast<int>(grid.m_nGridDim[1]);
          j_array[iCandidateIx] =
             (*iterCandidate) / static_cast<int>(grid.m_nGridDim[1]);
       }
       lat_array[iCandidateIx] = grid.m_dLat[*iterCandidate] * 180.0 / M_PI;
       lon_array[iCandidateIx] = grid.m_dLon[*iterCandidate] * 180.0 / M_PI;

       iCandidateIx++;
    }

    // build the output
    p_teca_table out_table = teca_table::New();
    out_table->set_calendar(calendar);
    out_table->set_time_units(time_units);

    // add time stamp
    out_table->declare_columns("step", long(), "time", double());
    for (unsigned long i = 0; i < setCandidates.size(); ++i)
       out_table << time_step << time_offset;

    // put the arrays into the table
    p_teca_variant_array_impl<double> latitude =
       teca_variant_array_impl<double>::New(setCandidates.size(), lat_array);
    p_teca_variant_array_impl<double> longitude =
       teca_variant_array_impl<double>::New(setCandidates.size(), lon_array);
    p_teca_variant_array_impl<int> i =
       teca_variant_array_impl<int>::New(setCandidates.size(), i_array);

    out_table->append_column("i", i);
    if (grid.m_nGridDim.size() == 2)
    {
       p_teca_variant_array_impl<int> j =
          teca_variant_array_impl<int>::New(setCandidates.size(), j_array);
       out_table->append_column("j", j);
    }
    out_table->append_column("lat", latitude);
    out_table->append_column("lon", longitude);

    for (int outc = 0; outc < this->internals->vecOutputOp.size(); outc++)
    {
       iCandidateIx = 0;
       std::string output_array[setCandidates.size()];
       std::set<int>::const_iterator iterCandidate = setCandidates.begin();
       for (; iterCandidate != setCandidates.end(); iterCandidate++)
       {
          output_array[iCandidateIx] = vecOutputValue[iCandidateIx][outc];
          iCandidateIx++;
       }

       p_teca_variant_array_impl<std::string> output =
          teca_variant_array_impl<std::string>::New(setCandidates.size(),
                                                                output_array);

       Variable & var = this->internals->varreg.Get(
          this->internals->vecOutputOp[outc].m_varix);
       out_table->append_column(var.ToString(this->internals->varreg).c_str(),
                                                                      output);
    }

#if TECA_DEBUG > 1
    out_table->to_stream(cerr);
    cerr << std::endl;
#endif

    return out_table;
}
