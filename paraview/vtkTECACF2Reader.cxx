#include "vtkTECACF2Reader.h"

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
#include "vtkRectilinearGrid.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "teca_common.h"
#include "teca_file_util.h"
#include "teca_vtk_util.h"
#include "teca_coordinate_util.h"
#include "teca_cartesian_mesh.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"
#include "teca_programmable_algorithm.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <sstream>

namespace internal
{
// helper class that interfaces to TECA's pipeline
// and extracts and converts the dataset.
struct teca_vtk_dataset_bridge
{
    teca_vtk_dataset_bridge() = delete;

    teca_vtk_dataset_bridge(const std::vector<int> &ext,
        double time, const std::vector<std::string> &active_arrays,
        vtkRectilinearGrid *output) : m_ext(ext), m_time(time),
        m_active_arrays(active_arrays), m_output(output) {}

    ~teca_vtk_dataset_bridge() {}

    // get_upstream_request
    std::vector<teca_metadata> operator()(unsigned int,
        const std::vector<teca_metadata> &input_md, const teca_metadata &)
    {
        teca_metadata coords;
        if (input_md[0].get("coordinates", coords))
        {
            TECA_ERROR("metadata missing \"coorindates\"")
            input_md[0].to_stream(cerr);
            return std::vector<teca_metadata>(1);
        }

        std::vector<double> times;
        if (coords.get("t", times))
        {
            TECA_ERROR("metadata missing time")
            input_md[0].to_stream(cerr);
            return std::vector<teca_metadata>(1);
        }

        unsigned long time_step = 0;
        if (teca_coordinate_util::index_of(times.data(), 0,
            times.size()-1, m_time, time_step))
        {
            TECA_ERROR("failed to locate the time step for " << m_time)
            return std::vector<teca_metadata>(1);
        }

        teca_metadata up_req;
        up_req.insert("time_step", time_step);
        up_req.insert("arrays", m_active_arrays);
        up_req.insert("extent", m_ext);

        return std::vector<teca_metadata>(1, up_req);
    }

    // execute
    const_p_teca_dataset operator()(unsigned int,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &)
    {
        const_p_teca_cartesian_mesh mesh = std::dynamic_pointer_cast
            <const teca_cartesian_mesh>(input_data[0]);

        if (!mesh)
        {
            TECA_ERROR("empty input")
            return nullptr;
        }

        if (teca_vtk_util::deep_copy(m_output, mesh))
        {
            TECA_ERROR("Failed to copy to vtkRecilinearGrid")
            return nullptr;
        }

        return mesh;
    }

    std::vector<int> m_ext;
    double m_time;
    std::vector<std::string> m_active_arrays;
    vtkRectilinearGrid *m_output;
};
};

//-----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTECACF2Reader);

//-----------------------------------------------------------------------------
vtkTECACF2Reader::vtkTECACF2Reader() :
  FileName(nullptr), BaseDir(nullptr), InputRegex(nullptr),
  XCoordinate(nullptr), YCoordinate(nullptr), ZCoordinate(nullptr),
  TimeCoordinate(nullptr), Reader(teca_cf_reader::New()),
  UpdateMetadata(1)
{
  // Initialize pipeline.
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);

  this->SetBaseDir("");
  this->SetInputRegex(".*\\.nc$");
  this->SetXCoordinate("lon");
  this->SetYCoordinate("lat");
  this->SetZCoordinate(".");
  this->SetTimeCoordinate("time");
}

//-----------------------------------------------------------------------------
vtkTECACF2Reader::~vtkTECACF2Reader()
{
  this->SetFileName(nullptr);
  this->SetBaseDir(nullptr);
  this->SetInputRegex(nullptr);
  this->SetXCoordinate(nullptr);
  this->SetYCoordinate(nullptr);
  this->SetZCoordinate(nullptr);
  this->SetTimeCoordinate(nullptr);
}

//-----------------------------------------------------------------------------
int vtkTECACF2Reader::CanReadFile(const char *file)
{
  // accept any file. we won't be reading this file at all.
  // this is simply to work around the fact that ParaView has
  // no good way for the user to specify anything but a file.
  if (!teca_file_util::file_exists(file))
    {
    return 0;
    }
  return 1;
}

//-----------------------------------------------------------------------------
void vtkTECACF2Reader::SetFileName(const char *fileName)
{
  // no change
  if (fileName && this->FileName && !strcmp(fileName, this->FileName))
    {
    return;
    }

  // delete the old
  free(this->FileName);

  if (fileName)
    {
    // update new file and base dir
    size_t n = strlen(fileName);
    this->FileName = static_cast<char*>(malloc(n+1));
    strcpy(this->FileName, fileName);

    std::string base = teca_file_util::path(fileName);
    this->SetBaseDir(base.c_str());
    }
  else
    {
    // clear out old info
    this->FileName = nullptr;
    this->SetBaseDir("");
    }

  this->Modified();
  this->UpdateMetadata = 1;
}

//-----------------------------------------------------------------------------
#define SET_STR_IMPL(_name) \
void vtkTECACF2Reader::Set ## _name (const char *val)       \
{                                                           \
  /* no change */                                           \
  if (val && this-> _name  && !strcmp(val, this-> _name))   \
    {                                                       \
    return;                                                 \
    }                                                       \
                                                            \
  /* delete the old */                                      \
  free(this-> _name);                                       \
                                                            \
  if (val)                                                  \
    {                                                       \
    /* update new value */                                  \
    size_t n = strlen(val);                                 \
    this-> _name = static_cast<char*>(malloc(n+1));         \
    strcpy(this-> _name, val);                              \
    }                                                       \
  else                                                      \
    {                                                       \
    /* clear out old info */                                \
    this-> _name = nullptr;                                 \
    }                                                       \
                                                            \
  this->Modified();                                         \
  this->UpdateMetadata = 1;                                 \
}

SET_STR_IMPL(InputRegex)
SET_STR_IMPL(XCoordinate)
SET_STR_IMPL(YCoordinate)
SET_STR_IMPL(ZCoordinate)
SET_STR_IMPL(TimeCoordinate)

//-----------------------------------------------------------------------------
void vtkTECACF2Reader::SetPointArrayStatus(const char *name, int status)
{
  if (!name) return;
  this->ActiveArrays[name] = status;
  this->Modified();
}

//-----------------------------------------------------------------------------
int vtkTECACF2Reader::GetPointArrayStatus(const char *name)
{
  if (!name) return 0;
  return this->ActiveArrays[name];
}

//-----------------------------------------------------------------------------
int vtkTECACF2Reader::GetNumberOfPointArrays()
{
  return this->ActiveArrays.size();
}

//-----------------------------------------------------------------------------
const char* vtkTECACF2Reader::GetPointArrayName(int idx)
{
  std::map<std::string,int>::iterator it = this->ActiveArrays.begin();
  std::map<std::string,int>::iterator itEnd = this->ActiveArrays.end();

  for (int i = 0; (it != itEnd) && (i < idx); ++it, ++i);

  if (it == itEnd)
    {
    vtkErrorMacro(<< idx << " out of bounds");
    return 0;
    }

  return (*it).first.c_str();
}

//-----------------------------------------------------------------------------
void vtkTECACF2Reader::ClearPointArrayStatus()
{
  std::map<std::string,int>::iterator it = this->ActiveArrays.begin();
  std::map<std::string,int>::iterator itEnd = this->ActiveArrays.end();

  for (; it != itEnd; ++it)
    {
    it->second = 0;
    }

  this->Modified();
  this->UpdateMetadata = 1;                                  \
}

//-----------------------------------------------------------------------------
int vtkTECACF2Reader::RequestInformation(
  vtkInformation *req, vtkInformationVector **inInfos,
  vtkInformationVector* outInfos)
{
  (void)req;
  (void)inInfos;

  vtkInformation *outInfo = outInfos->GetInformationObject(0);

  if (this->UpdateMetadata)
    {

    if (!this->FileName)
      {
      vtkErrorMacro("FileName has not been set.")
      return 1;
      }

    // TODO : reading the metadata is expensive.  should I be
    // protecting this with a check to Modified??
    std::string files_regex = std::string(this->BaseDir) +
      PATH_SEP + std::string(this->InputRegex);

    this->Reader->set_files_regex(files_regex);

    if (this->XCoordinate[0] == '.')
      {
      this->Reader->set_x_axis_variable("");
      }
    else
      {
      this->Reader->set_x_axis_variable(this->XCoordinate);
      }

    if (this->YCoordinate[0] == '.')
      {
      this->Reader->set_y_axis_variable("");
      }
    else
      {
      this->Reader->set_y_axis_variable(this->YCoordinate);
      }

    if (this->ZCoordinate[0] == '.')
      {
      this->Reader->set_z_axis_variable("");
      }
    else
      {
      this->Reader->set_z_axis_variable(this->ZCoordinate);
      }

    if (this->TimeCoordinate[0] == '.')
      {
      this->Reader->set_t_axis_variable("");
      }
    else
      {
      this->Reader->set_t_axis_variable(this->TimeCoordinate);
      }

    this->Metadata.clear();
    this->Metadata = this->Reader->update_metadata();

    // pass metadata into ParaView pipeline
    // get list of variables
    this->ActiveArrays.clear();

    std::vector<std::string> vars;
    if (this->Metadata.get("variables", vars))
      {
      this->Metadata.to_stream(cerr);
      vtkErrorMacro("metadata is missing \"variables\"")
      return 1;
      }

    size_t nVars = vars.size();
    for (size_t i = 0; i < nVars; ++i)
      {
      this->ActiveArrays[vars[i]] =  0;
      }
    }

  // pass time axis into ParaView
  teca_metadata coords;
  if (this->Metadata.get("coordinates", coords))
    {
    this->Metadata.to_stream(cerr);
    vtkErrorMacro("metadata is missing \"coordinates\"")
    return 1;
    }

  std::vector<double> time_axis;
  if (coords.get("t", time_axis))
    {
    time_axis.push_back(0.0);
    }

  size_t nTimes = time_axis.size();

  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(),
    time_axis.data(), static_cast<int>(nTimes));

  double timeRange[2] = {time_axis[0], time_axis[nTimes - 1]};
  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), timeRange, 2);

  // Pass the data set extents
  std::vector<int> wholeExt;
  if (this->Metadata.get("whole_extent", wholeExt))
    {
    this->Metadata.to_stream(cerr);
    vtkErrorMacro("metadata missing \"whole_extent\"")
    return  1;
    }

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
    wholeExt.data(), 6);

  // mark last complete update
  this->UpdateMetadata = 0;

  return 1;
}

//-----------------------------------------------------------------------------
int vtkTECACF2Reader::RequestData(
        vtkInformation *req, vtkInformationVector **inInfo,
        vtkInformationVector *outInfos)
{
  (void)req;
  (void)inInfo;

  vtkInformation *outInfo = outInfos->GetInformationObject(0);

  // Get the output dataset.
  vtkRectilinearGrid *output = dynamic_cast<vtkRectilinearGrid*>(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  if (!output)
    {
    vtkErrorMacro("Output data has not been configured correctly.");
    return 1;
    }

  // get the update extent
  std::vector<int> ext(6, 0);
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), ext.data());

  // get the list of active arrays
  std::vector<std::string> active_arrays;
  std::map<std::string, int>::iterator it = this->ActiveArrays.begin();
  std::map<std::string, int>::iterator end = this->ActiveArrays.end();
  for (; it != end; ++it)
    {
    if (it->second)
      {
      active_arrays.push_back(it->first);
      }
    }

  // get the requested time step
  double time = 0.0;
  if (outInfo->Has(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP()))
    {
    time = outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());
    }
  outInfo->Set(vtkDataObject::DATA_TIME_STEP(), time);

  // hook up the TECA pipeline
  internal::teca_vtk_dataset_bridge bridge(ext, time, active_arrays, output);

  p_teca_programmable_algorithm alg = teca_programmable_algorithm::New();
  alg->set_request_callback(bridge);
  alg->set_execute_callback(bridge);
  alg->set_input_connection(this->Reader->get_output_port());
  alg->update();

  return 1;
}

constexpr const char *safestr(const char *ptr)
{ return ptr?ptr:"nullptr"; }

//-----------------------------------------------------------------------------
void vtkTECACF2Reader::PrintSelf(ostream& os, vtkIndent indent)
{
  os << indent << "FileName = " << safestr(this->FileName) << endl
    << indent << "XCoordinate = " << safestr(this->XCoordinate) << endl
    << indent << "YCoordinate = " << safestr(this->YCoordinate) << endl
    << indent << "ZCoordinate = " << safestr(this->ZCoordinate) << endl
    << indent << "TimeCoordinate = " << safestr(this->TimeCoordinate) << endl;
}
