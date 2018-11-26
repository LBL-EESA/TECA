#include "vtkTECATCTrackReader.h"

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

template<typename tt> struct vtk_tt;
#define DECLARE_VTK_TT(_c_type, _vtk_type) \
template <> \
struct vtk_tt<_c_type> \
{ typedef vtk ## _vtk_type ## Array VTK_TT; }

DECLARE_VTK_TT(float, Float);
DECLARE_VTK_TT(double, Double);
DECLARE_VTK_TT(char, Char);
DECLARE_VTK_TT(short, Short);
DECLARE_VTK_TT(int, Int);
DECLARE_VTK_TT(long, Long);
DECLARE_VTK_TT(long long, LongLong);
DECLARE_VTK_TT(unsigned char, UnsignedChar);
DECLARE_VTK_TT(unsigned short, UnsignedShort);
DECLARE_VTK_TT(unsigned int, UnsignedInt);
DECLARE_VTK_TT(unsigned long, UnsignedLong);
DECLARE_VTK_TT(unsigned long long, UnsignedLongLong);

//-----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTECATCTrackReader);

//-----------------------------------------------------------------------------
vtkTECATCTrackReader::vtkTECATCTrackReader() : XCoordinate(nullptr),
  YCoordinate(nullptr), TrackCoordinate(nullptr)
{
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  this->SetXCoordinate("lon");
  this->SetYCoordinate("lat");
  this->SetTrackCoordinate("track_id");
}

//-----------------------------------------------------------------------------
vtkTECATCTrackReader::~vtkTECATCTrackReader()
{
  this->SetXCoordinate(nullptr);
  this->SetYCoordinate(nullptr);
  this->SetTrackCoordinate(nullptr);
}

//-----------------------------------------------------------------------------
int vtkTECATCTrackReader::RequestInformation(vtkInformation *req,
  vtkInformationVector **inInfos, vtkInformationVector* outInfos)
{
  // table should be read by now
  if (!this->Table)
    {
    vtkErrorMacro("Failed to read the table")
    return 1;
    }

  // base class handles some common operations, such as reporting time
  this->vtkTECATableReader::RequestInformation(req, inInfos, outInfos);

  // extract the tracks
  if (!this->TrackCoordinate)
    {
    vtkErrorMacro("Must set the track coordinate")
    return 1;
    }

  if (!this->Table->has_column(this->TrackCoordinate))
    {
    vtkErrorMacro("Track coordinate \""
      << this->TrackCoordinate << "\" is invalid")
    return 1;
    }

  // get first and last indices of each track
  std::vector<unsigned long> track_ids;
  this->Table->get_column(this->TrackCoordinate)->get(track_ids);

  unsigned long nm1 = track_ids.size() - 1;

  this->TrackRows[track_ids[0]].first = 0;
  this->TrackRows[track_ids[nm1]].second = nm1;

  for (unsigned long i = 0; i < nm1; ++i)
    {
    if (track_ids[i] != track_ids[i+1])
      {
      this->TrackRows[track_ids[i]].second = i;
      this->TrackRows[track_ids[i+1]].first = i+1;
      }
    }

  return 1;
}

//-----------------------------------------------------------------------------
int vtkTECATCTrackReader::RequestData(vtkInformation *req,
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

  // determine the requested time range
  double time = 0.0;
  if (outInfo->Has(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP()))
    time = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

  // set time on the output
  vtkInformation *dataInfo = output->GetInformation();
  dataInfo->Set(vtkDataObject::DATA_TIME_STEP(), time);
  outInfo->Set(vtkDataObject::DATA_TIME_STEP(), time);

  const_p_teca_variant_array track_times =
    this->Table->get_column(this->TimeCoordinate);
  if (!track_times)
    {
    vtkErrorMacro("Time coordinate \"" << this->TimeCoordinate << "\" not in table")
    return 1;
    }

  // for simplicity we're just gonna dump all the points in the table.
  // the cells we define below will pick out the ones that are actually
  // in use
  unsigned long nPts = this->Table->get_number_of_rows();
  unsigned long first = 0;
  unsigned long last = nPts - 1;

  // get coordinate arrays
  vtkDoubleArray *pts = vtkDoubleArray::New();
  pts->SetNumberOfComponents(3);
  pts->SetNumberOfTuples(nPts);
  pts->SetName("coords");
  double *ppts = pts->GetPointer(0);

  // build the VTK points structure from the columns that
  // the user has named
  std::vector<double> tmp;
  tmp.resize(nPts);

  const char *axesNames[] = {this->XCoordinate, this->YCoordinate, "."};

  for (long i = 0; i < 3; ++i, ++ppts)
    {
    if (axesNames[i] && axesNames[i][0] != '.')
      {
      // not a nullptr and not '.'. note: '.' tells us to fill this
      // column with zeros.
      if (!this->Table->has_column(axesNames[i]))
        {
        // missing ther equested column
        const char *axesLabels = "XYZ";
        std::ostringstream oss;
        oss << "The column requested for the " << axesLabels[i]
          << " coordinate axis \"" << axesNames[i] << "\" is not in the table. "
          " The available columns are: {";
        if (nCols > 0)
          {
          oss << this->Table->get_column_name(0);
          for (unsigned int j = 1; j < nCols; ++j)
            oss << ", " << this->Table->get_column_name(j);
          }
        oss << "}";
        vtkErrorMacro(<< oss.str())
        return 1;
        }

      // copy it over into a VTK data structure
      const_p_teca_variant_array axis = this->Table->get_column(axesNames[i]);
      axis->get(first, last, tmp.data());

      for (unsigned long i = 0; i < nPts; ++i)
        ppts[3*i] = tmp[i];
      }
    else
      {
      // fill this column with zeros
      for (unsigned long i = 0; i < nPts; ++i)
        ppts[3*i] = 0.0;
      }
    }

  // copy all of the columns in
  for (unsigned int i = 0; i < nCols; ++i)
    {
    const_p_teca_variant_array ar = this->Table->get_column(i);

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
      ar.get(),

      vtk_tt<NT>::VTK_TT *da = vtk_tt<NT>::VTK_TT::New();
      da->SetName(this->Table->get_column_name(i).c_str());
      da->SetNumberOfTuples(nPts);

      TT *tar = dynamic_cast<TT*>(ar.get());
      tar->get(first, last, da->GetPointer(0));

      output->GetPointData()->AddArray(da);
      da->Delete();
      )
    }

  tmp.clear();

  vtkPoints *points = vtkPoints::New();
  points->SetData(pts);
  pts->Delete();

  output->SetPoints(points);
  points->Delete();

  unsigned long nLines = 0;
  vtkIdTypeArray *lines = vtkIdTypeArray::New();

  unsigned long nVerts = 0;
  vtkIdTypeArray *verts = vtkIdTypeArray::New();

  // for each track, determine if the track exists at this time point.
  // if it does then add it to the output
  TrackRowMapT::iterator it = this->TrackRows.begin();
  TrackRowMapT::iterator end = this->TrackRows.end();

  for (; it != end; ++it)
    {
    first = it->second.first;
    last = it->second.second;

    // time range spanned by the track
    double t0, t1;
    track_times->get(first, t0);
    track_times->get(last, t1);

    // the track is not visible at this instant in time
    if (!((time >= t0) && (time <= t1)))
      continue;

    // determine the current end of the track
    for (unsigned long q = first; q <= last; ++q)
    {
        double tq;
        track_times->get(q, tq);
        if (tq > time)
        {
            last = q - 1;
            break;
        }
    }

    // append track geometry
    unsigned long nPts = last - first + 1;
    if (nPts > 1)
      {
      // add a line segment connecting the points
      ++nLines;

      unsigned long insLoc = lines->GetNumberOfTuples();
      vtkIdType *pCells = lines->WritePointer(insLoc, nPts+1);

      pCells[0] = nPts;

      for (unsigned long i = 0; i < nPts; ++i)
        pCells[i+1] = first + i;
      }
    else
      {
      // add a point
      ++nVerts;
      unsigned long insLoc = verts->GetNumberOfTuples();
      vtkIdType *pCells = verts->WritePointer(insLoc, nPts+1);
      pCells[0] = 1;
      pCells[1] = first;
      }

    }

  // add the geometry to the output
  vtkCellArray *lineArray = vtkCellArray::New();
  lineArray->SetCells(nLines, lines);
  lines->Delete();

  output->SetLines(lineArray);
  lineArray->Delete();

  vtkCellArray *vertArray = vtkCellArray::New();
  vertArray->SetCells(nVerts, verts);
  verts->Delete();

  output->SetVerts(vertArray);
  vertArray->Delete();

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
void vtkTECATCTrackReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "XCoordinate = " << safestr(this->XCoordinate) << endl
    << indent << "YCoordinate = " << safestr(this->YCoordinate) << endl
    << indent << "TrackCoordinate = " << safestr(this->TrackCoordinate) << endl;
}
