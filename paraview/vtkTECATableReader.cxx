#include "vtkTECATableReader.h"

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
vtkStandardNewMacro(vtkTECATableReader);

//-----------------------------------------------------------------------------
vtkTECATableReader::vtkTECATableReader() :
  FileName(nullptr), XCoordinate(nullptr), YCoordinate(nullptr),
  ZCoordinate(nullptr), TimeCoordinate(nullptr)
{
  // Initialize pipeline.
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  this->SetXCoordinate("lon");
  this->SetYCoordinate("lat");
  this->SetZCoordinate(".");
  this->SetTimeCoordinate("time");
}

//-----------------------------------------------------------------------------
vtkTECATableReader::~vtkTECATableReader()
{
  this->SetFileName(nullptr);
  this->SetXCoordinate(nullptr);
  this->SetYCoordinate(nullptr);
  this->SetZCoordinate(nullptr);
  this->SetTimeCoordinate(nullptr);
}

//-----------------------------------------------------------------------------
int vtkTECATableReader::CanReadFile(const char *file_name)
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
int vtkTECATableReader::RequestInformation(
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

    std::vector<double> times;
    this->Table->get_column(this->TimeCoordinate)->get(times);

    // give paraview a unique list, and store range of
    // indices into the table for each time step
    unique_times.reserve(times.size());
    unique_times.push_back(times[0]);
    size_t nm1 = times.size() - 1;
    this->TimeRows[times[0]].first = 0;
    this->TimeRows[times[nm1]].second = nm1;
    for (size_t i = 0; i < nm1; ++i)
      {
      if (std::fabs(times[i] - times[i+1]) > 1.e-6)
        {
        unique_times.push_back(times[i+1]);
        this->TimeRows[times[i]].second = i;
        this->TimeRows[times[i+1]].first = i+1;
        }
      }
    }
  else
    {
    unique_times = {0.0};
    this->TimeRows.insert(std::make_pair(0.0, std::make_pair(0,0)));
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
int vtkTECATableReader::RequestData(
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
  size_t first = 0;
  size_t last = 0;
  double time = 0.0;

  if (outInfo->Has(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP()))
    {
    time = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
    }

  std::map<double, std::pair<size_t, size_t>>::iterator it =
     this->TimeRows.find(time);

  if (it == this->TimeRows.end())
    {
    vtkErrorMacro("Invalid time " << time << " requested")
    return 1;
    }

  vtkInformation *dataInfo = output->GetInformation();
  dataInfo->Set(vtkDataObject::DATA_TIME_STEP(), time);

  outInfo->Set(vtkDataObject::DATA_TIME_STEP(), time);

  first = it->second.first;
  last = it->second.second;
  size_t nPts = last - first + 1;

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

  vtkPoints *points = vtkPoints::New();
  points->SetData(pts);
  pts->Delete();

  output->SetPoints(points);
  points->Delete();

  vtkIdTypeArray *cells = vtkIdTypeArray::New();
  cells->SetNumberOfTuples(2*nPts);

  vtkIdType *pcells = cells->GetPointer(0);

  for (size_t i = 0; i < nPts; ++i)
    pcells[2*i] = 1;

  pcells += 1;
  for (size_t i = 0; i < nPts; ++i)
    pcells[2*i] = i;

  vtkCellArray *cellArray = vtkCellArray::New();
  cellArray->SetCells(nPts, cells);
  cells->Delete();

  output->SetVerts(cellArray);
  cellArray->Delete();

  // copy all of the columns in
  // TODO -- enable user selection of specific columns
  // TODO -- make this zero copy, this can be done now
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

  return 1;
}


constexpr const char *safestr(const char *ptr)
{ return ptr?ptr:"nullptr"; }

//-----------------------------------------------------------------------------
void vtkTECATableReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "FileName = " << safestr(this->FileName) << endl
    << indent << "XCoordinate = " << safestr(this->XCoordinate) << endl
    << indent << "YCoordinate = " << safestr(this->YCoordinate) << endl
    << indent << "ZCoordinate = " << safestr(this->ZCoordinate) << endl
    << indent << "TimeCoordinate = " << safestr(this->TimeCoordinate) << endl;
}
