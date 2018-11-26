#include "vtkTECATCWindRadiiReader.h"

#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkCellArray.h"
#include "vtkIdTypeArray.h"

#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkCharArray.h"
#include "vtkShortArray.h"
#include "vtkIntArray.h"
#include "vtkLongArray.h"
#include "vtkLongLongArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnsignedShortArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkUnsignedLongLongArray.h"
#include "vtkStringArray.h"

#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <vector>

#include "teca_coordinate_util.h"

namespace internals {

unsigned int count_cells(vtkIdTypeArray *cells)
{
  vtkIdType *pcells = cells->GetPointer(0);
  unsigned int len = cells->GetNumberOfTuples();

  unsigned int n = 0;
  for (unsigned int i = 0; i < len; i = i + pcells[i] + 1)
    n += 1;

  return n;
}

void append(unsigned int pos, unsigned int n_vals,
    vtkDoubleArray *dest, double val)
{
  double *pdest = dest->WritePointer(pos, n_vals);
  for (unsigned int j = 0; j < n_vals; ++j)
    pdest[j] = val;
}

template<typename NT_MESH>
unsigned int  add_curve(const NT_MESH *r_crit, const NT_MESH *theta,
    const NT_MESH *x, const NT_MESH *y, unsigned int theta_resolution,
    unsigned int arc_resolution, vtkDoubleArray *pts, vtkIdTypeArray *cells)
{
  unsigned int n_pts = pts->GetNumberOfTuples();
  unsigned int pts_per_cell = arc_resolution*theta_resolution;
  unsigned int pts_per_cell_1 = pts_per_cell + 1;
  unsigned int pts_per_cell_2 = pts_per_cell + 2;

  NT_MESH dtheta = 2.0*M_PI/static_cast<NT_MESH>(theta_resolution);
  NT_MESH dtheta_2 = dtheta/2.0;
  NT_MESH dtheta_arc = dtheta/(arc_resolution - 1);

  // points
  double *ppts = pts->WritePointer(3*n_pts, 3*pts_per_cell);
  for (unsigned int j = 0; j < theta_resolution; ++j)
    {
    NT_MESH rj = r_crit[j];
    NT_MESH xj = x[j];
    NT_MESH yj = y[j];

    NT_MESH theta_0 = theta[j] - dtheta_2;

    for (unsigned int i = 0; i < arc_resolution; ++i)
      {
      NT_MESH theta_i = theta_0 + i*dtheta_arc;
      NT_MESH xi = xj + rj*cos(theta_i);
      NT_MESH yi = yj + rj*sin(theta_i);

      unsigned int ii = 3*i;
      ppts[ii  ] = xi;
      ppts[ii+1] = yi;
      ppts[ii+2] = 0.0;
      }

    ppts += 3*arc_resolution;
    }

  // cells
  vtkIdType *pcells = cells->WritePointer
    (cells->GetNumberOfTuples(), pts_per_cell_2);

  pcells[0] = pts_per_cell_1;
  for (unsigned int j = 0; j < pts_per_cell; ++j)
      pcells[j+1] = n_pts + j;
  pcells[pts_per_cell_1] = n_pts;

  n_pts += pts_per_cell;

  return pts_per_cell;
}

template<typename NT_MESH>
unsigned int add_wedge(const NT_MESH *r_crit, const NT_MESH *theta,
    const NT_MESH *x, const NT_MESH *y, unsigned int theta_resolution,
    unsigned int arc_resolution, vtkDoubleArray *pts, vtkIdTypeArray *cells)
{
  unsigned int pts_per_cell = arc_resolution + 1;
  unsigned int pts_per_cell_1 = pts_per_cell + 1;
  unsigned int pts_per_cell_2 = pts_per_cell + 2;
  unsigned int n_pts = pts->GetNumberOfTuples();
  unsigned int n_new_pts = pts_per_cell*theta_resolution;

  NT_MESH dtheta = 2.0*M_PI/static_cast<NT_MESH>(theta_resolution);
  NT_MESH dtheta_2 = dtheta/2.0;
  NT_MESH dtheta_arc = dtheta/(arc_resolution - 1);

  // points
  double *ppts = pts->WritePointer(3*n_pts, 3*n_new_pts);
  for (unsigned int j = 0; j < theta_resolution; ++j)
    {
    NT_MESH rj = r_crit[j];
    NT_MESH xj = x[j];
    NT_MESH yj = y[j];

    ppts[0] = xj;
    ppts[1] = yj;
    ppts[2] = 0.0;

    NT_MESH theta_0 = theta[j] - dtheta_2;

    for (unsigned int i = 0; i < arc_resolution; ++i)
      {
      NT_MESH theta_i = theta_0 + i*dtheta_arc;
      NT_MESH xi = xj + rj*cos(theta_i);
      NT_MESH yi = yj + rj*sin(theta_i);

      unsigned int ii = 3*(i + 1);
      ppts[ii  ] = xi;
      ppts[ii+1] = yi;
      ppts[ii+2] = 0.0;
      }

    ppts += 3*pts_per_cell;
    }

  // cells
  for (unsigned int j = 0; j < theta_resolution; ++j)
    {
    vtkIdType *pcells = cells->WritePointer
        (cells->GetNumberOfTuples(), pts_per_cell_2);

    pcells[0] = pts_per_cell_1;
    for (unsigned int i = 0; i < pts_per_cell; ++i)
      {
      unsigned int ii = i + 1;
      pcells[ii] = n_pts + pts_per_cell*j + i;
      }
    pcells[pts_per_cell_1] = n_pts + pts_per_cell*j;
    }

  return n_new_pts;
}
};

//-----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTECATCWindRadiiReader);

//-----------------------------------------------------------------------------
vtkTECATCWindRadiiReader::vtkTECATCWindRadiiReader() : XCoordinate(nullptr),
  YCoordinate(nullptr), TrackCoordinate(nullptr), CurveCoordinate(nullptr),
  GeometryMode(GEOMETRY_MODE_CURVE)
{
  // Initialize pipeline.
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  this->SetXCoordinate("lon");
  this->SetYCoordinate("lat");
  this->SetTrackCoordinate("track_id");
  this->SetCurveCoordinate("track_point_id");
}

//-----------------------------------------------------------------------------
vtkTECATCWindRadiiReader::~vtkTECATCWindRadiiReader()
{
  this->SetXCoordinate(nullptr);
  this->SetYCoordinate(nullptr);
  this->SetTrackCoordinate(nullptr);
  this->SetCurveCoordinate(nullptr);
}

//-----------------------------------------------------------------------------
int vtkTECATCWindRadiiReader::RequestInformation(vtkInformation *req,
  vtkInformationVector **inInfos, vtkInformationVector* outInfos)
{
  // table shoule have been read by now
  if (!this->Table)
    {
    vtkErrorMacro("Failed to read the table")
    return 1;
    }

  // base class handles common tasks like reporting time
  this->vtkTECATableReader::RequestInformation(req, inInfos, outInfos);

  // get curve ids
  if (!this->CurveCoordinate)
    {
    vtkErrorMacro("Must set the track coordinate")
    return 1;
    }

  if (!this->Table->has_column(this->CurveCoordinate))
    {
    vtkErrorMacro("Curve coordinate \""
      << this->CurveCoordinate << "\" is invalid")
    return 1;
    }

  std::vector<unsigned long> curve_id;
  this->Table->get_column(this->CurveCoordinate)->get(curve_id);

  // build the track map, has first and last indices of each track
  unsigned long nm1 = curve_id.size() - 1;

  this->CurveMap[curve_id[0]].first = 0;
  this->CurveMap[curve_id[nm1]].second = nm1;

  for (unsigned long i = 0; i < nm1; ++i)
    {
    if (curve_id[i] != curve_id[i+1])
      {
      this->CurveMap[curve_id[i]].second = i;
      this->CurveMap[curve_id[i+1]].first = i+1;
      }
    }

  return 1;
}

//-----------------------------------------------------------------------------
int vtkTECATCWindRadiiReader::RequestData(vtkInformation *req,
  vtkInformationVector **inInfo, vtkInformationVector *outInfos)
{
  (void)req;
  (void)inInfo;

  // table should have been read by now
  if (!this->Table)
    {
    vtkErrorMacro("Failed to read the table")
    return 1;
    }

  vtkInformation *outInfo = outInfos->GetInformationObject(0);

  // Get the output dataset.
  vtkPolyData *output = dynamic_cast<vtkPolyData*>(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (!output)
    {
    vtkErrorMacro("Output data has not been configured correctly.");
    return 1;
    }

  // quick check on the validity of the input
  unsigned int nCols = this->Table->get_number_of_columns();
  if (nCols < 1)
    {
    vtkErrorMacro("The file has 0 columns")
    return 1;
    }

  // determine the requested time
  double req_time = -1.0;
  if (outInfo->Has(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP()))
    req_time = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

  // set time on the output
  vtkInformation *dataInfo = output->GetInformation();
  dataInfo->Set(vtkDataObject::DATA_TIME_STEP(), req_time);
  outInfo->Set(vtkDataObject::DATA_TIME_STEP(), req_time);

  // get track time
  const_p_teca_variant_array time_coord =
    this->Table->get_column(this->TimeCoordinate);

  if (!time_coord)
    {
    vtkErrorMacro("Missing " << this->TimeCoordinate)
    return 1;
    }

  // get track ids
  const_p_teca_variant_array track_ids =
   this->Table->get_column(this->TrackCoordinate);

  if (!track_ids)
    {
    vtkErrorMacro("Missing " <<  this->TrackCoordinate)
    return 1;
    }

  // get track centers
  const_p_teca_variant_array track_x =
    this->Table->get_column(this->XCoordinate);

  if (!track_x)
    {
    vtkErrorMacro("Missing " <<  this->XCoordinate)
    return 1;
    }

  const_p_teca_variant_array track_y =
    this->Table->get_column(this->YCoordinate);

  if (!track_y)
    {
    vtkErrorMacro("Missing " <<  this->YCoordinate)
    return 1;
    }

  // get theta
  const_p_teca_variant_array theta =
    this->Table->get_column("theta");

  if (!theta)
    {
    vtkErrorMacro("Missing theta")
    return 1;
    }

  // get theta resolution
  teca_metadata md = this->Table->get_metadata();
  if (!md.has("theta_resolution"))
    {
    vtkErrorMacro("Missing theta resolution")
    return 1;
    }
  unsigned int theta_resolution = 0;
  md.get("theta_resolution", 0, theta_resolution);

  // get the critical wind values
  if (!md.has("critical_wind_speeds"))
    {
    vtkErrorMacro("Missing critical wind speeds")
    return 1;
    }
  std::vector<double> w_crit;
  md.get("critical_wind_speeds", w_crit);
  unsigned int n_crit = w_crit.size();

  // and the wind radii
  std::vector<const_p_teca_variant_array> r_crit(n_crit);
  for (unsigned int i = 0; i < n_crit; ++i)
    {
    std::ostringstream oss;
    oss << "r_" << i;

    r_crit[i] = this->Table->get_column(oss.str());
    if (!r_crit[i])
      {
      vtkErrorMacro("Missing " << oss.str())
      return 1;
      }
    }

  // get peak radius
  const_p_teca_variant_array r_peak = this->Table->get_column("r_peak");
  if (!r_peak)
    {
    vtkErrorMacro("Missing r_peak")
    return 1;
    }

  // get peak wind
  const_p_teca_variant_array w_peak = this->Table->get_column("w_peak");
  if (!w_peak)
    {
    vtkErrorMacro("Missing w_peak")
    return 1;
    }

  // vtk data structures describing wind radii curves
  vtkDoubleArray *pts = vtkDoubleArray::New();
  pts->SetNumberOfComponents(3);
  pts->SetName("coords");

  vtkIdTypeArray *cells = vtkIdTypeArray::New();
  cells->SetName("cells");

  vtkDoubleArray *track_id = vtkDoubleArray::New();
  track_id->SetName("track_id");

  vtkDoubleArray *curve_id = vtkDoubleArray::New();
  curve_id->SetName("curve_id");

  vtkDoubleArray *wind = vtkDoubleArray::New();
  wind->SetName("wind");

  // for each track, determine if the track exists at this time point.
  // if it does then add it to the output
  NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
    track_x.get(), _MESH,

    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
      w_peak.get(), _WIND,

      const NT_MESH *px = static_cast<TT_MESH*>(track_x.get())->get();
      const NT_MESH *py = static_cast<TT_MESH*>(track_y.get())->get();
      const NT_MESH *ptheta = static_cast<TT_MESH*>(theta.get())->get();

      std::vector<const NT_MESH*> pr_crit(n_crit);
      for (unsigned int i = 0; i < n_crit; ++i)
        pr_crit[i] = static_cast<TT_MESH*>(r_crit[i].get())->get();

      const NT_MESH *pr_peak = static_cast<TT_MESH*>(r_peak.get())->get();
      const NT_WIND *pw_peak = static_cast<TT_WIND*>(w_peak.get())->get();

      CurveMapT::iterator it = this->CurveMap.begin();
      CurveMapT::iterator end = this->CurveMap.end();

      for (; it != end; ++it)
        {
        unsigned long i = it->first;
        unsigned long q0 = it->second.first;
        //unsigned long q1 = it->second.second;

        // time range spanned by the track
        double tq0;
        time_coord->get(q0, tq0);

        // track point is not visible now
        if (!teca_coordinate_util::equal(req_time, tq0,
            8.0*std::numeric_limits<double>::epsilon()))
            continue;

        // if there are no wind radii then skip it
        bool have_wind_radii = false;
        for (unsigned int k = 0; k < n_crit; ++k)
          {
              const NT_MESH *pr_crit_k = pr_crit[k] + q0;
              for (unsigned int j = 0; j < theta_resolution; ++j)
                  if (pr_crit_k[j] > NT_MESH(1.0e-6))
                      have_wind_radii = true;
          }
        if (!have_wind_radii)
            continue;

        // track id
        double tid;
        track_ids->get(i, tid);

        const NT_MESH *ptheta_q0 = ptheta + q0;
        const NT_MESH *px_q0 = px + q0;
        const NT_MESH *py_q0 = py + q0;
        const NT_MESH *pr_peak_q0 = pr_peak + q0;
        const NT_WIND *pw_peak_q0 = pw_peak + q0;

        unsigned int pos = 0;
        unsigned int n_new_pts = 0;
        constexpr unsigned int arc_resolution = 32;

        // generate geometry
        // wind radii
        for (unsigned int k = 0; k < n_crit; ++k)
          {
          const NT_MESH *pr_crit_k_q0 = pr_crit[k] + q0;

          if (this->GeometryMode == GEOMETRY_MODE_CURVE)
            n_new_pts = internals::add_curve(pr_crit_k_q0, ptheta_q0, px_q0,
                 py_q0, theta_resolution, arc_resolution, pts, cells);
          else
            n_new_pts = internals::add_wedge(pr_crit_k_q0, ptheta_q0, px_q0,
                 py_q0, theta_resolution, arc_resolution, pts, cells);

          internals::append(pos, n_new_pts, track_id, tid);
          internals::append(pos, n_new_pts, curve_id, k);
          internals::append(pos, n_new_pts, wind, w_crit[k]);

          pos += n_new_pts;
          }


        n_new_pts = internals::add_curve(pr_peak_q0, ptheta_q0, px_q0,
            py_q0, theta_resolution, arc_resolution, pts, cells);

        internals::append(pos, n_new_pts, track_id, tid);
        internals::append(pos, n_new_pts, curve_id, n_crit);
        for (unsigned int j = 0; j < theta_resolution; ++j)
          internals::append(pos+j*arc_resolution,
             arc_resolution, wind, pw_peak_q0[j]);

        pos += n_new_pts;
        }
      )
    )

  // add the points to the output
  vtkPoints *points = vtkPoints::New();
  points->SetData(pts);
  pts->Delete();

  output->SetPoints(points);
  points->Delete();

  // add the cells to the output
  vtkCellArray *cellArray = vtkCellArray::New();
  cellArray->SetCells(internals::count_cells(cells), cells);
  cells->Delete();

  output->SetLines(cellArray);
  cellArray->Delete();

  // add track_id to the output
  output->GetPointData()->AddArray(track_id);
  track_id->Delete();

  // add curve_id to the output
  output->GetPointData()->AddArray(curve_id);
  curve_id->Delete();

  // add wind to output
  output->GetPointData()->AddArray(wind);
  wind->Delete();

  // add calendaring information
  std::string calendar;
  this->Table->get_calendar(calendar);
  vtkStringArray *sarr = vtkStringArray::New();
  sarr->SetName("calendar");
  sarr->SetNumberOfTuples(1);
  sarr->SetValue(0, calendar);
  output->GetFieldData()->AddArray(sarr);
  sarr->Delete();

  std::string timeUnits;
  this->Table->get_time_units(timeUnits);
  sarr = vtkStringArray::New();
  sarr->SetName("time_units");
  sarr->SetNumberOfTuples(1);
  sarr->SetValue(0, timeUnits);
  output->GetFieldData()->AddArray(sarr);
  sarr->Delete();

  return 1;
}

constexpr const char *safestr(const char *ptr)
{ return ptr?ptr:"nullptr"; }

//-----------------------------------------------------------------------------
void vtkTECATCWindRadiiReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "XCoordinate = " << safestr(this->XCoordinate) << endl
    << indent << "YCoordinate = " << safestr(this->YCoordinate) << endl
    << indent << "TrackCoordinate = " << safestr(this->TrackCoordinate) << endl
    << indent << "CurveCoordinate = " << safestr(this->CurveCoordinate) << endl;
}
