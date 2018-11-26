#include "vtkTECATableReader.h"

#include "vtkObjectFactory.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPolyData.h"

#include <cstring>
#include <cstdio>

#include <teca_file_util.h>
#include <teca_table_reader.h>
#include <teca_table_sort.h>
#include <teca_programmable_algorithm.h>

namespace internal
{
// helper class that interfaces to TECA's pipeline
// and extracts the dataset.
struct teca_pipeline_bridge
{
    teca_pipeline_bridge() = delete;

    teca_pipeline_bridge(const_p_teca_table *output)
        : m_output(output) {}

    const_p_teca_dataset operator()(unsigned int,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &)
    {
        if (!(*m_output = std::dynamic_pointer_cast
            <const teca_table>(input_data[0])))
        {
            TECA_ERROR("empty input")
        }
        return *m_output;
    }

    const_p_teca_table *m_output;
};
};

//-----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTECATableReader);

//-----------------------------------------------------------------------------
vtkTECATableReader::vtkTECATableReader() : FileName(nullptr),
  TimeCoordinate(nullptr), SortTimeCoordinate(0)
{
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  this->SetTimeCoordinate("time");
}

//-----------------------------------------------------------------------------
vtkTECATableReader::~vtkTECATableReader()
{
  this->SetFileName(nullptr);
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
void vtkTECATableReader::SetFileName(const char *fn)
{
  if (this->FileName && fn && !strcmp(fn, this->FileName))
    return;

  if (this->FileName || fn)
    this->Modified();

  free(this->FileName);
  this->FileName = fn ? strdup(fn) : nullptr;

  if (this->FileName)
    {
    // read and sort the data
    internal::teca_pipeline_bridge br(&this->Table);

    p_teca_table_reader tr = teca_table_reader::New();
    tr->set_file_name(this->FileName);

    p_teca_programmable_algorithm pa = teca_programmable_algorithm::New();
    if (this->SortTimeCoordinate && (this->TimeCoordinate[0] != '.'))
      {
      p_teca_table_sort ts = teca_table_sort::New();
      ts->set_input_connection(tr->get_output_port());
      ts->set_index_column(this->TimeCoordinate);
      pa->set_input_connection(ts->get_output_port());
      }
    else
      {
      pa->set_input_connection(tr->get_output_port());
      }
    pa->set_execute_callback(br);
    pa->update();
    }
}

//-----------------------------------------------------------------------------
int vtkTECATableReader::RequestInformation(vtkInformation *req,
  vtkInformationVector **inInfos, vtkInformationVector* outInfos)
{
  (void)req;
  (void)inInfos;

  // table should have been read by now
  if (!this->Table)
    {
    vtkErrorMacro("Failed to read the table")
    return 1;
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

  unsigned long n_unique = unique_times.size();

  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(),
    unique_times.data(), static_cast<int>(n_unique));

  double timeRange[2] = {unique_times[0], unique_times[n_unique - 1]};
  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), timeRange, 2);

  return 1;
}

constexpr const char *safestr(const char *ptr)
{ return ptr?ptr:"nullptr"; }

//-----------------------------------------------------------------------------
void vtkTECATableReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "FileName = " << safestr(this->FileName) << endl
    << indent << "TimeCoordinate = " << safestr(this->TimeCoordinate) << endl;
}
