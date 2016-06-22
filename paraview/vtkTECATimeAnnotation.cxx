#include "vtkTECATimeAnnotation.h"

#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStringArray.h"
#include "vtkTable.h"
#include "vtkDataSet.h"
#include "vtkFieldData.h"
#include "vtkDataArray.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "calcalcs.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTECATimeAnnotation);

//----------------------------------------------------------------------------
vtkTECATimeAnnotation::vtkTECATimeAnnotation() :
  IncludeYear(1), IncludeMonth(1), IncludeDay(1),
  IncludeHour(1), IncludeMinute(1), IncludeSecond(1),
  DateSeparator(nullptr), TimeSeparator(nullptr)
{
  this->SetNumberOfOutputPorts(1);
  this->SetDateSeparator("-");
  this->SetTimeSeparator(":");
}

//----------------------------------------------------------------------------
vtkTECATimeAnnotation::~vtkTECATimeAnnotation()
{}

//----------------------------------------------------------------------------
int vtkTECATimeAnnotation::FillInputPortInformation(
  int port, vtkInformation* info)
{
  (void)port;
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataObject");
  return 1;
}

//----------------------------------------------------------------------------
int vtkTECATimeAnnotation::FillOutputPortInformation(
  int port, vtkInformation *info)
{
  (void)port;
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkTable");
  return 1;
}

//----------------------------------------------------------------------------
int vtkTECATimeAnnotation::RequestData(vtkInformation *req,
    vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  (void)req;

  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkDataSet *input = dynamic_cast<vtkDataSet*>(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  if (!input)
    {
    vtkErrorMacro("empty input")
    return 1;
    }

  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkTable *output = dynamic_cast<vtkTable*>(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));
  if (!output)
    {
    vtkErrorMacro("empty output")
    return 1;
    }

  // get calendar system
  vtkStringArray *sarr = dynamic_cast<vtkStringArray*>(
    input->GetFieldData()->GetAbstractArray("calendar"));
  if (!sarr)
    {
    vtkErrorMacro("missing calendar")
    return 1;
    }
  std::string calendar = sarr->GetValue(0);

  sarr = dynamic_cast<vtkStringArray*>(
    input->GetFieldData()->GetAbstractArray("time_units"));
  if (!sarr)
    {
    vtkErrorMacro("missing time_units")
    return 1;
    }
  std::string time_units = sarr->GetValue(0);

  // get requested time
  double time = 0.0;
  if (outInfo->Has(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP()))
    {
    time = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
    }

  // convert time to date
  int year = 0;
  int month = 0;
  int day = 0;
  int hour = 0;
  int minute = 0;
  double second = 0;

  if (calcalcs::date(time, &year, &month, &day, &hour, &minute, &second,
      time_units.c_str(), calendar.c_str()))
  {
      vtkErrorMacro("Failed to compute the date time=" <<  time)
      return 1;
  }

  std::ostringstream oss;
  oss << std::setfill('0');
  if (this->IncludeYear)
    oss << std::setw(4) << year;

  if (this->IncludeYear && (this->IncludeMonth || this->IncludeDay))
    oss << std::setw(1) << this->DateSeparator;

  if (this->IncludeMonth)
    oss << std::setw(2) << month;

  if ((this->IncludeYear || this->IncludeMonth) && this->IncludeDay)
    oss << std::setw(1) << this->DateSeparator;

  if (this->IncludeDay)
    oss << std::setw(2) << day;

  if ((this->IncludeYear || this->IncludeMonth || this->IncludeDay) &&
    (this->IncludeHour || this->IncludeMinute || this->IncludeSecond))
    oss << std::setw(1) << " ";

  if (this->IncludeHour)
    oss << std::setw(2) << hour;

  if (this->IncludeHour && (this->IncludeMinute || this->IncludeSecond))
    oss << std::setw(1) << this->TimeSeparator;

  if (this->IncludeMinute)
    oss << std::setw(2) << minute;

  if ((this->IncludeMinute || this->IncludeHour) && this->IncludeSecond)
    oss << std::setw(1) << this->TimeSeparator;

  if (this->IncludeSecond)
    oss << std::setw(2) << second;

  vtkStringArray *data = vtkStringArray::New();
  data->SetName("time");
  data->SetNumberOfComponents(1);
  data->InsertNextValue(oss.str());
  output->AddColumn(data);
  data->Delete();

  return 1;
}

//----------------------------------------------------------------------------
void vtkTECATimeAnnotation::PrintSelf(ostream& os, vtkIndent indent)
{
  os << indent << "IncludeYear=" << IncludeYear << endl
    << indent << "IncludeMonth=" << IncludeMonth << endl
    << indent << "IncludeDay=" << IncludeDay << endl
    << indent << "IncludeHour=" << IncludeHour << endl
    << indent << "IncludeMinute=" << IncludeMinute << endl
    << indent << "IncludeSecond=" << IncludeSecond << endl
    << indent << "DateSeparator=" << DateSeparator << endl
    << indent << "TimeSeparator=" << TimeSeparator << endl;
}
