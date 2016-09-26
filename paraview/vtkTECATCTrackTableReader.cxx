#include "vtkTECATCTrackTableReader.h"

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
vtkStandardNewMacro(vtkTECATCTrackTableReader);

//-----------------------------------------------------------------------------
vtkTECATCTrackTableReader::vtkTECATCTrackTableReader() :
  FileName(nullptr), XCoordinate(nullptr), YCoordinate(nullptr),
  ZCoordinate(nullptr), TimeCoordinate(nullptr), TrackCoordinate(nullptr)
{
  // Initialize pipeline.
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  this->SetXCoordinate("lon");
  this->SetYCoordinate("lat");
  this->SetZCoordinate(".");
  this->SetTimeCoordinate("time");
  this->SetTrackCoordinate("track_id");
}

//-----------------------------------------------------------------------------
vtkTECATCTrackTableReader::~vtkTECATCTrackTableReader()
{
  this->SetFileName(nullptr);
  this->SetXCoordinate(nullptr);
  this->SetYCoordinate(nullptr);
  this->SetZCoordinate(nullptr);
  this->SetTimeCoordinate(nullptr);
  this->SetTrackCoordinate(nullptr);
}

//-----------------------------------------------------------------------------
int vtkTECATCTrackTableReader::CanReadFile(const char *file_name)
{
  // open the file
  teca_binary_stream bs;
  FILE* fd = fopen(file_name, "rb");
  if (!fd)
    {
    vtkErrorMacro("Failed to open " << file_name << endl)
    return 0;
    }

  // check if this is really ours
  int canRead = 0;
  char id[11] = {'\0'};
  if ((fread(id, 1, 10, fd) == 10)
    && !strncmp(id, "teca_table", 10))
    {
    canRead = 1;
    }

  fclose(fd);
  return canRead;
}

//-----------------------------------------------------------------------------
int vtkTECATCTrackTableReader::RequestInformation(
  vtkInformation *req, vtkInformationVector **inInfos,
  vtkInformationVector* outInfos)
{
  (void)req;
  (void)inInfos;

  if (!this->FileName)
    {
    vtkErrorMacro("FileName has not been set.")
    return 1;
    }

  // for now just read the whole thing here
  // open the file
  teca_binary_stream bs;
  FILE* fd = fopen(this->FileName, "rb");
  if (fd == NULL)
    {
    vtkErrorMacro("Failed to open " << this->FileName << endl)
    return 1;
    }

  // get its length, we'll read it in one go and need to create
  // a bufffer for it's contents
  long start = ftell(fd);
  fseek(fd, 0, SEEK_END);
  long end = ftell(fd);
  fseek(fd, 0, SEEK_SET);
  long nbytes = end - start - 10;

  // check if this is really ours
  char id[11] = {'\0'  };
  if (fread(id, 1, 10, fd) != 10)
    {
    const char *estr = (ferror(fd) ? strerror(errno) : "");
    fclose(fd);
    vtkErrorMacro("Failed to read \"" << this->FileName << "\". " << estr)
    return 1;
    }

  if (strncmp(id, "teca_table", 10))
    {
    fclose(fd);
    vtkErrorMacro("Not a teca_table. \"" << this->FileName << "\"")
    return 1;
    }

  // create the buffer
  bs.resize(static_cast<size_t>(nbytes));

  // read the stream
  long bytes_read = fread(bs.get_data(), sizeof(unsigned char), nbytes, fd);
  if (bytes_read != nbytes)
    {
    const char *estr = (ferror(fd) ? strerror(errno) : "");
    fclose(fd);
    vtkErrorMacro("Failed to read \"" << this->FileName << "\". Read only "
      << bytes_read << " of the requested " << nbytes << ". " << estr)
    return 1;
    }
  fclose(fd);

  // deserialize the binary rep
  this->Table = teca_table::New();
  this->Table->from_stream(bs);

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
  std::vector<long> tracks;
  this->Table->get_column(this->TrackCoordinate)->get(tracks);

  size_t nm1 = tracks.size() - 1;

  this->TrackRows.resize(tracks[nm1]+1);

  this->TrackRows[tracks[0]].first = 0;
  this->TrackRows[tracks[nm1]].second = nm1;

  for (size_t i = 0; i < nm1; ++i)
    {
    if (tracks[i] != tracks[i+1])
      {
      this->TrackRows[tracks[i]].second = i;
      this->TrackRows[tracks[i+1]].first = i+1;
      }
    }

  // get the time values
  std::vector<double> unique_times;
  if (!this->TimeCoordinate)
    {
    vtkErrorMacro("Must set the time coordinate")
    return 1;
    }

  if (this->TimeCoordinate[0] != '.')
    {
    if (!this->Table->has_column(this->TimeCoordinate))
      {
      vtkErrorMacro("Time coordinate \""
        << this->TimeCoordinate << "\" is invalid")
      return 1;
      }

    std::vector<double> tmp_vec;
    this->Table->get_column(this->TimeCoordinate)->get(tmp_vec);

    // make the list unique and sorted
    std::set<double> tmp_set(tmp_vec.begin(), tmp_vec.end());
    unique_times.assign(tmp_set.begin(), tmp_set.end());
    }
  else
    {
    unique_times = {0.0};
    }

  // pass into pipeline.
  vtkInformation *outInfo = outInfos->GetInformationObject(0);

  size_t n_unique = unique_times.size();

  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(),
    unique_times.data(), static_cast<int>(n_unique));

  double timeRange[2] = {unique_times[0], unique_times[n_unique - 1]};
  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), timeRange, 2);

  return 1;
}

//-----------------------------------------------------------------------------
int vtkTECATCTrackTableReader::RequestData(
        vtkInformation *req, vtkInformationVector **inInfo,
        vtkInformationVector *outInfos)
{
  (void)req;
  (void)inInfo;

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
    {
    time = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
    }


  // set time on the output
  vtkInformation *dataInfo = output->GetInformation();
  dataInfo->Set(vtkDataObject::DATA_TIME_STEP(), time);

  outInfo->Set(vtkDataObject::DATA_TIME_STEP(), time);


  p_teca_variant_array track_times = this->Table->get_column(this->TimeCoordinate);
  if (!track_times)
    {
    vtkErrorMacro("Time coordinate \"" << this->TimeCoordinate << "\" not in table")
    return 1;
    }

  // for simplicity we're just gonna dump all the points in the table.
  // the cells we define below will pick out the ones that are actually
  // in use
  size_t nPts = this->Table->get_number_of_rows();
  size_t first = 0;
  size_t last = nPts - 1;

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

  const char *axesNames[] =
    {this->XCoordinate, this->YCoordinate, this->ZCoordinate};

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

      // copyt it over into a VTK data structure
      // TODO -- this could be made to be a zero-copy transfer when
      // Utkarsh merges the new zero copy api.
      // TODO -- implement a spherical coordinate transform
      p_teca_variant_array axis = this->Table->get_column(axesNames[i]);
      axis->get(first, last, tmp.data());

      for (size_t i = 0; i < nPts; ++i)
        ppts[3*i] = tmp[i];
      }
    else
      {
      // fill this column with zeros
      for (size_t i = 0; i < nPts; ++i)
        ppts[3*i] = 0.0;
      }
    }

  // TODO -- this could be zero copy
  // copy all of the columns in
  for (unsigned int i = 0; i < nCols; ++i)
    {
    p_teca_variant_array ar = this->Table->get_column(i);

    TEMPLATE_DISPATCH(teca_variant_array_impl,
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

  size_t nLines = 0;
  vtkIdTypeArray *lines = vtkIdTypeArray::New();

  size_t nVerts = 0;
  vtkIdTypeArray *verts = vtkIdTypeArray::New();

  // for each track, determine if the track exists at this time point.
  // if it does then add it to the output
  size_t nTracks = this->TrackRows.size();
  for (size_t i = 0; i < nTracks; ++i)
    {
    first = this->TrackRows[i].first;
    last = this->TrackRows[i].second;

    // time range spanned by the track
    double t0, t1;
    track_times->get(first, t0);
    track_times->get(last, t1);

    // the track is not visible at this instant in time
    if (!((time >= t0) && (time <= t1)))
      continue;

    // determine the current end of the track
    for (size_t q = first; q <= last; ++q)
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
    size_t nPts = last - first + 1;
    if (nPts > 1)
      {
      // add a line segment connecting the points
      ++nLines;

      size_t insLoc = lines->GetNumberOfTuples();
      vtkIdType *pCells = lines->WritePointer(insLoc, nPts+1);

      pCells[0] = nPts;

      for (size_t i = 0; i < nPts; ++i)
        pCells[i+1] = first + i;
      }
    else
      {
      // add a point
      ++nVerts;
      size_t insLoc = verts->GetNumberOfTuples();
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
void vtkTECATCTrackTableReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "FileName = " << safestr(this->FileName) << endl
    << indent << "XCoordinate = " << safestr(this->XCoordinate) << endl
    << indent << "YCoordinate = " << safestr(this->YCoordinate) << endl
    << indent << "ZCoordinate = " << safestr(this->ZCoordinate) << endl
    << indent << "TimeCoordinate = " << safestr(this->TimeCoordinate) << endl
    << indent << "TrackCoordinate = " << safestr(this->TrackCoordinate) << endl;
}
