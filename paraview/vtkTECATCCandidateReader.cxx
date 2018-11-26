#include "vtkTECATCCandidateReader.h"

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
vtkStandardNewMacro(vtkTECATCCandidateReader);

//-----------------------------------------------------------------------------
vtkTECATCCandidateReader::vtkTECATCCandidateReader() : XCoordinate(nullptr),
  YCoordinate(nullptr), ZCoordinate(nullptr)
{
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  this->SetXCoordinate("lon");
  this->SetYCoordinate("lat");
  this->SetZCoordinate(".");

  this->SetSortTimeCoordinate(1);
}

//-----------------------------------------------------------------------------
vtkTECATCCandidateReader::~vtkTECATCCandidateReader()
{
  this->SetXCoordinate(nullptr);
  this->SetYCoordinate(nullptr);
  this->SetZCoordinate(nullptr);
}

//-----------------------------------------------------------------------------
int vtkTECATCCandidateReader::RequestData(vtkInformation *req,
  vtkInformationVector **inInfo, vtkInformationVector *outInfos)
{
  (void)req;
  (void)inInfo;

  // table shoule have been read by now
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
  size_t first = 0;
  size_t last = 0;
  double time = 0.0;

  if (outInfo->Has(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP()))
    time = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

  std::map<double, std::pair<size_t, size_t>>::iterator it =
     this->TimeRows.find(time);

  if (it == this->TimeRows.end())
    {
    // not an error
    // vtkErrorMacro("Invalid time " << time << " requested")
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
      // TODO -- implement a spherical coordinate transform
      const_p_teca_variant_array axis = this->Table->get_column(axesNames[i]);
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
  for (unsigned int i = 0; i < nCols; ++i)
    {
    const_p_teca_variant_array ar = this->Table->get_column(i);

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
      ar.get(),

      vtk_tt<NT>::VTK_TT *da = vtk_tt<NT>::VTK_TT::New();
      da->SetName(this->Table->get_column_name(i).c_str());
      da->SetNumberOfTuples(nPts);

      const TT *tar = dynamic_cast<const TT*>(ar.get());
      tar->get(first, last, da->GetPointer(0));

      output->GetPointData()->AddArray(da);
      da->Delete();
      )
    }

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
void vtkTECATCCandidateReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "FileName = " << safestr(this->FileName) << endl
    << indent << "XCoordinate = " << safestr(this->XCoordinate) << endl
    << indent << "YCoordinate = " << safestr(this->YCoordinate) << endl
    << indent << "ZCoordinate = " << safestr(this->ZCoordinate) << endl
    << indent << "TimeCoordinate = " << safestr(this->TimeCoordinate) << endl;
}
