/*=========================================================================

  Program:   cvtn
  Module:    vtkCvtNeuronReader.h

  Copyright (c) B. Loring, V. K. Buochard
  All rights reserved.

=========================================================================*/
#include "vtkCvtNeuronReader.h"

#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "neuron.h"

#include <hdf5.h>
#include <hdf5_hl.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/queuing_mutex.h>
#include <tbb/task_group.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <mpi.h>

using index_t = long;
using coord_t = float;
using data_t = float;

struct vtkCvtNeuronReader::vtkInternals
{
  vtkInternals() {}
  ~vtkInternals()
  {
    int nNeurons = this->Neurons.size();
    for (int i = 0; i < nNeurons; ++i)
      this->Neurons[i].freeMem();
    this->Neurons.clear();
    this->TimeSeries.freeMem();
  }

  std::vector<neuron::Neuron<index_t,coord_t,data_t>> Neurons;
  neuron::TimeSeries<index_t, coord_t, data_t> TimeSeries;
};

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkCvtNeuronReader);

//----------------------------------------------------------------------------
vtkCvtNeuronReader::vtkCvtNeuronReader() :
  NumberOfThreads(-1), Directory(nullptr), HasCoords(0), HasData(0),
  NeuronIds{0,0}, ReadNeuronIds{0,-1}, CurrentThreshold(0.07),
  VoltageThreshold(std::numeric_limits<float>::lowest()),
  Internals(nullptr)
{
  std::cerr << " ==== vtkCvtNeuronReader::vtkCvtNeuronReader" << std::endl;

  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(2);

  // disable hdf5 error spew. we need to probe the files to see which of
  // the file formats we have in hand.
  //H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
}

//----------------------------------------------------------------------------
vtkCvtNeuronReader::~vtkCvtNeuronReader()
{
  std::cerr << " ==== vtkCvtNeuronReader::~vtkCvtNeuronReader" << std::endl;
  std::cerr << " ==== cleaning up...";
  auto t0 = std::chrono::high_resolution_clock::now();

  delete this->Internals;
  free(this->Directory);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cerr << "done! ("
    << 1.e-6*std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
    << "s)" << std::endl;
}

//----------------------------------------------------------------------------
void vtkCvtNeuronReader::SetReadNeuronIds(int firstNeuron, int lastNeuron)
{
  this->SetFirstNeuron(firstNeuron);
  this->SetLastNeuron(lastNeuron);
}

//----------------------------------------------------------------------------
void vtkCvtNeuronReader::SetFirstNeuron(int val)
{
  if (this->ReadNeuronIds[0] == val)
    return;

  // this will cause geometry to be regenerated
  delete this->Internals;
  this->Internals = nullptr;

  // clamp at the high end of the available range
  if (val > this->NeuronIds[1])
  {
    vtkErrorMacro("FirstNeuron " << val
      << " out of bounds in [0, " << this->NeuronIds[1] << "]");
    this->ReadNeuronIds[0] = this->NeuronIds[1];
    return;
  }

  // clamp on the low side to 0
  if (val < -1)
      val = 0;

  this->ReadNeuronIds[0] = val;

  this->Modified();
}

//----------------------------------------------------------------------------
void vtkCvtNeuronReader::SetLastNeuron(int val)
{
  if (this->ReadNeuronIds[1] == val)
    return;

  // this will cause geometry to be regenerated
  delete this->Internals;
  this->Internals = nullptr;

  // clamp at the high end of the available range
  if (val > this->NeuronIds[1])
  {
    vtkErrorMacro("FirstNeuron " << val
      << " out of bounds in [0, " << this->NeuronIds[1] << "]");
    this->ReadNeuronIds[1] = this->NeuronIds[1];
    return;
  }

  // -1 results in automatic detection of the range
  if (val < -1)
      val = -1;

  this->ReadNeuronIds[1] = val;

  this->Modified();
}

//----------------------------------------------------------------------------
const char *vtkCvtNeuronReader::GetFileName() const
{
  return this->Directory;
}

//----------------------------------------------------------------------------
void vtkCvtNeuronReader::SetFileName(const char *fn)
{
  std::cerr << " ==== vtkCvtNeuronReader::SetFileName" << std::endl;
  std::cerr << " ==== " << (fn ? fn : "nullptr") << std::endl;

  // ParavIew will call this to pass a file name from the file open dialog.
  // convert the file name to a directory.
  if (!fn)
  {
    delete this->Internals;
    this->Internals = nullptr;
    free(this->Directory);
    this->Directory = nullptr;
    this->NeuronIds[0] = 0;
    this->NeuronIds[1] = 0;
    this->ReadNeuronIds[0] = 0;
    this->ReadNeuronIds[1] = -1;
    this->HasCoords = 0;
    this->HasData = 0;
    return;
  }

  char *dir = nullptr;
  size_t n = strlen(fn);
  for (size_t i = n - 1; i > 0; --i)
  {
    if (fn[i] == '/')
    {
       dir = strndup(fn, i);
       this->SetDirectory(dir);
       free(dir);
       return;
    }
  }

  this->SetDirectory(".");
}

//----------------------------------------------------------------------------
const char *vtkCvtNeuronReader::GetDirectory() const
{
  return this->Directory;
}

//----------------------------------------------------------------------------
void vtkCvtNeuronReader::SetDirectory(const char *dirName)
{
  std::cerr << " ==== vtkCvtNeuronReader::SetDirectory" << std::endl;
  std::cerr << " ==== " << (dirName ? dirName : "nullptr") << std::endl;

  if (dirName && this->Directory && (strcmp(this->Directory, dirName) == 0))
    return;

  delete this->Internals;
  this->Internals = nullptr;

  free(this->Directory);
  this->Directory = nullptr;
  this->NeuronIds[0] = 0;
  this->NeuronIds[1] = 0;
  this->ReadNeuronIds[0] = 0;
  this->ReadNeuronIds[1] = -1;
  this->HasCoords = 0;
  this->HasData = 0;

  this->Modified();

  if (!dirName)
    return;

  // rank 0 scans
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
  {
    std::cerr << " ==== scanning " << dirName << " ... ";
    auto rt0 = std::chrono::high_resolution_clock::now();

    // detect the segment geometry
    if (neuron::scanForNeurons(dirName, this->NeuronIds[1]))
    {
      vtkErrorMacro("No neuron geometry was found in \"" << dirName << "\"");
      return;
    }
    this->HasCoords = 1;

    // detect the presence of time series data
    this->HasData = neuron::imFileExists(dirName);

    auto rt1 = std::chrono::high_resolution_clock::now();
    std::cerr << "done! ("
      << 1.e-6*std::chrono::duration_cast<std::chrono::microseconds>(rt1 - rt0).count()
      << "s)" << std::endl
      << " ==== found " << this->NeuronIds[1] + 1 << " neurons in " << dirName << std::endl;
  }

  this->Directory = strdup(dirName);

  // other ranks receive results without touching the disk.
  MPI_Bcast(this->NeuronIds, 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&this->HasCoords, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&this->HasData, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

//----------------------------------------------------------------------------
int vtkCvtNeuronReader::CanReadFile()
{
  std::cerr << " ==== vtkCvtNeuronReader::CanReadFile" << std::endl;

  if (!this->Directory)
  {
    vtkErrorMacro("Directory was not set");
    return 0;
  }

  return this->HasCoords;
}

//----------------------------------------------------------------------------
int vtkCvtNeuronReader::InitializeGeometry(const char *inputDir,
  int neuron0, int neuron1)
{
  std::cerr << " ==== vtkCvtNeuronReader::InitializeGeometry" << std::endl;

  int nNeuron = neuron1 - neuron0 + 1;

  std::cerr << " ==== importing " << nNeuron << " neurons " << neuron0
    << " to " << neuron1 << " from " << inputDir << " ... ";
  auto rt0 = std::chrono::high_resolution_clock::now();

  std::vector<neuron::Neuron<index_t,coord_t,data_t>> neurons(nNeuron);

  tbb::parallel_for(tbb::blocked_range<int>(0,nNeuron),
    neuron::Importer<index_t,coord_t,data_t>(neuron0, inputDir, &neurons));

  this->Internals->Neurons.swap(neurons);

  auto rt1 = std::chrono::high_resolution_clock::now();
  std::cerr << "done! ("
    << 1.e-6*std::chrono::duration_cast<std::chrono::microseconds>(rt1 - rt0).count()
    << "s)" << std::endl;

  // initailze time series handlers
  if (this->HasData)
  {
    std::cerr << " ==== intializing time series, mesher, and exporter...";
    rt0 = std::chrono::high_resolution_clock::now();

    neuron::TimeSeries<index_t, coord_t, data_t>
        timeSeries(inputDir, &this->Internals->Neurons);

    data_t threshold[2] = {this->CurrentThreshold, this->VoltageThreshold};
    if (timeSeries.initialize(threshold))
    {
      vtkErrorMacro("Failed to initialize the time series object");
      return -1;
    }

    rt1 = std::chrono::high_resolution_clock::now();
    std::cerr << "done! ("
      << 1.e-6*std::chrono::duration_cast<std::chrono::microseconds>(rt1 - rt0).count()
      << "s)" << std::endl;

    //timeSeries.print();

    this->Internals->TimeSeries.swap(timeSeries);
  }

  return 0;
}

//----------------------------------------------------------------------------
int vtkCvtNeuronReader::FillOutputPortInformation(int port, vtkInformation *info)
{
  std::cerr << " ==== vtkCvtNeuronReader::FillOutputPortInformation" << std::endl;
  (void)port;
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
  return 1;
}

//----------------------------------------------------------------------------
int vtkCvtNeuronReader::RequestData(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** vtkNotUsed(inputVector), vtkInformationVector* outputVector)
{
  std::cerr << " ==== vtkCvtNeuronReader::RequestData" << std::endl;

  if (!this->HasCoords)
  {
    vtkErrorMacro("No geometry was detected");
    return 0;
  }

  // get the info objects
  vtkInformation* outInfo0 = outputVector->GetInformationObject(0);
  vtkInformation* outInfo1 = outputVector->GetInformationObject(1);

  // TODO
  // get the domain decomp
  int piece = outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER());
  int numPieces = outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES());

  // import and cache the neuron geometry, and set up the time series importer
  if (!this->Internals)
  {
    // TODO: this probably isn't the rtight thing to do. since VTK may use
    // TBB capabilities as well, there's probably a VTK API to intialize TBB.
    tbb::task_scheduler_init init(this->NumberOfThreads);
    std::cerr << " ==== initializing TBB with " << this->NumberOfThreads
        << " threads" << std::endl;

    this->Internals = new vtkInternals;

    // this is globaly requested subset across all MPI ranks
    int neuron0 = this->ReadNeuronIds[0] < 0 ? 0 : this->ReadNeuronIds[0];
    int neuron1 = this->ReadNeuronIds[1] < 0 ? this->NeuronIds[1] : this->ReadNeuronIds[1];

    // figure out the subset to load on this rank. piece is the rank and
    // numPieces is the number of ranks when running MPI parallel.
    int nTotal = neuron1 - neuron0 + 1;
    int nLocal = nTotal / numPieces;
    int nLarge = nTotal % numPieces;

    neuron0 = nLocal * piece;
    if (piece < nLarge)
      neuron0 += piece;
    else
      neuron0 += nLarge;

    neuron1 = neuron0 + nLocal - 1;
    if (piece < nLarge)
      neuron1 += 1;

    if (this->InitializeGeometry(this->Directory, neuron0, neuron1))
    {
      vtkErrorMacro("Failed to read the geometry from \"" << this->Directory << "\"");
      delete this->Internals;
      this->Internals = nullptr;
      return 0;
    }
  }

  // get the requested time value
  double timeVal = outInfo0->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

  // from that get the time step
  double t0 = this->Internals->TimeSeries.t0;
  double dt = this->Internals->TimeSeries.dt;
  long  timeStep = (timeVal - t0) / dt;

  std::cerr << " ==== processing step " << timeStep << " ... ";
  auto rt0 = std::chrono::high_resolution_clock::now();

  // update to this time step and interpolate new values to the neurons
  if (this->HasData && this->Internals->TimeSeries.importTimeStep(timeStep))
  {
    vtkErrorMacro("Failed to import time step " << timeStep);
    return 0;
  }

  // convert the data into a VTK object
  vtkPolyData *cells = nullptr;
  vtkPolyData *nodes = nullptr;
  using Exporter = neuron::Exporter<index_t,coord_t,data_t>;
  if (Exporter::packageCellsAndNodes(&this->Internals->Neurons, cells, nodes))
  {
    vtkErrorMacro("Failed to construct the output datasets");
    return 0;
  }

  auto rt1 = std::chrono::high_resolution_clock::now();
  std::cerr << "done! ("
      << 1.e-6*std::chrono::duration_cast<std::chrono::microseconds>(rt1 - rt0).count()
      << "s)" << std::endl;

  // get the output
  vtkPolyData* cellsOut = vtkPolyData::SafeDownCast(outInfo0->Get(vtkDataObject::DATA_OBJECT()));
  cellsOut->ShallowCopy(cells);
  cells->Delete();

  vtkPolyData* nodesOut = vtkPolyData::SafeDownCast(outInfo1->Get(vtkDataObject::DATA_OBJECT()));
  nodesOut->ShallowCopy(nodes);
  nodes->Delete();

  return 1;
}

//----------------------------------------------------------------------------
int vtkCvtNeuronReader::RequestInformation(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** vtkNotUsed(inputVector), vtkInformationVector* outputVector)
{
  std::cerr << " ==== vtkCvtNeuronReader::RequestInformation" << std::endl;

  if (!this->HasCoords)
  {
    vtkErrorMacro("No geometry was detected");
    return 0;
  }

  std::vector<double> timeValues;

  if (this->HasData)
  {
    double t0 = 0.0;
    double dt = 0.0;
    int nSteps = 0;

    // rank 0 scans
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
      // extract only the time series info. the converter code does this
      // with a bunch of allocations and initializations that should be
      // deffered, so re-implement that code here.
      hid_t fh = -1;

      char fn[256];
      snprintf(fn, 256, "%s/im.h5", this->Directory);

      fh = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
      if (fh < 0)
      {
        H5Eprint1(stderr);
        vtkErrorMacro("Failed to open " << fn);
        return 0;
      }

      // read time metadata
      hsize_t dims[2] = {0};
      data_t tmd[3] = {data_t(0)};
      if (neuron::readDimensions(fh, "/mapping/time", dims, 1, 3) ||
          neuron::cppH5Tt<data_t>::readDataset(fh, "/mapping/time", tmd))
      {
        vtkErrorMacro("Failed to read time metadata");
        H5Fclose(fh);
        return 0;
      }

      t0 = tmd[0];
      //t1 = tmd[1];
      dt = tmd[2];

      // attempt to detect the file format. Vyassa's using the bmtk code
      // and we need to attempt to work through a number of cases depending on
      // what that code has done.
      int nDims = 0;
      const char *dsetName = nullptr;
      if (neuron::readNumDimensions(fh, "/data", nDims) == 0)
      {
        dsetName = "/data";
      }
      else if ((neuron::readNumDimensions(fh, "/im/data", nDims) == 0) &&
        (neuron::readNumDimensions(fh, "/v/data", nDims) == 0))
      {
        dsetName = "/im/data";

      }
      else
      {
        vtkErrorMacro("Failed to detect file format. Expected either \"/data\" "
          "or \"/data/im\" and \"/data/v\"");
        H5Fclose(fh);
        return 0;
      }

      // get size of buffers for time series
      if (neuron::readDimensions(fh, dsetName, dims, nDims))
      {
        H5Fclose(fh);
        return 0;
      }

      H5Fclose(fh);

      nSteps = dims[0];
      //long stepSize = dims[1];
    }

    // all others receive the data without touching the disk
    MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // construct and share an explicit list of time values for VTK.
    // as far as I know this is required even though we have a uniform time
    // step.
    timeValues.resize(nSteps);
    for (long i = 0; i < nSteps; ++i)
    {
      timeValues[i] = t0 + i * dt;
    }
  }
  else
  {
    timeValues.resize(1);
  }

  // get the VTK pipeline info object
  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  // tell that this reader supports parallel domain decomp.
  // Notes: there are a number of ways to parallelize the data, this can
  // be implemted fairly easily by parallelizing over neurons.
  // we have experimented with VTK multiblock data and found that rendering
  // speed is prohibative, so we will stick with the basic poly/unstructured
  // mesh when parallelizing.
  outInfo->Set(CAN_HANDLE_PIECE_REQUEST(), 1);

  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_STEPS(),
    timeValues.data(), timeValues.size());

  // pass the range, even though we already passed it in the above array.
  // as far as I know VTK needs this to function.
  double timeRange[2] = {timeValues[0], timeValues.back()};
  outInfo->Set(vtkStreamingDemandDrivenPipeline::TIME_RANGE(), timeRange, 2);

  return 1;
}
