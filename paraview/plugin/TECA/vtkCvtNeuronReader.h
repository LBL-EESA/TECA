/*=========================================================================

  Program:   cvtn
  Module:    vtkCvtNeuronReader.h

  Copyright (c) B. Loring, V. K. Buochard
  All rights reserved.

=========================================================================*/
/**
 * @class   vtkCvtNeuronReader
 * @brief   a reader of neuron simulation data
 */
#ifndef vtkCvtNeuronReader_h
#define vtkCvtNeuronReader_h

#include "vtkFiltersSourcesModule.h" // For export macro
#include "vtkPolyDataAlgorithm.h"

class VTKFILTERSSOURCES_EXPORT vtkCvtNeuronReader : public vtkPolyDataAlgorithm
{
public:
  vtkTypeMacro(vtkCvtNeuronReader, vtkPolyDataAlgorithm);
  void PrintSelf(ostream&, vtkIndent) override {}

  static vtkCvtNeuronReader* New();

  // Set the file name to read. this method is required by ParaView.
  // For normal operation use the SetDirectory method instead.
  void SetFileName(const char *fn);
  const char *GetFileName() const;

  // Set the directory containing seg_coords directory and im.h5
  // file. The seg_coords directory is scanned to determine the
  // number of neurons.
  void SetDirectory(const char *dn);
  const char *GetDirectory() const;

  // return 1 if the data can be read from the current directory.
  // This method is requried by ParaView.
  int CanReadFile();

  // Set the range of neurons to read. The defaults of 0,-1 results in
  // all neurons being read.
  void SetFirstNeuron(int val);
  void SetLastNeuron(int val);

  // these are for the PV gui to display the ids of available neurons
  // and set the range of neurons to read.
  void SetReadNeuronIds(int firstNeuron, int lastNeuron);
  vtkGetVector2Macro(NeuronIds, int);

  // Set the minimum voltage to consider the neuron active.
  // an array named active will hold the boolean result.
  vtkGetMacro(VoltageThreshold, float)
  vtkSetMacro(VoltageThreshold, float)

  // Set the minimum voltage to consider the neuron active.
  // an array named active will hold the boolean result.
  vtkGetMacro(CurrentThreshold, float)
  vtkSetMacro(CurrentThreshold, float)

  // Tells if the active directory has geometry and/or time series data
  // Note: it is an error when geometry is not found.
  vtkGetMacro(HasCoords, int)
  vtkGetMacro(HasData, int)

  // set the number of TBB threads to use, -1 will result in an automatic
  // detection. this may be inapropriate when running with MPI.
  vtkSetMacro(NumberOfThreads, int)
  vtkGetMacro(NumberOfThreads, int)

protected:
  vtkCvtNeuronReader();
  ~vtkCvtNeuronReader() override;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int FillOutputPortInformation(int port, vtkInformation *info) override;

  // read the range of neurons from disk. this pre-processes the data into a format
  // ammenable to visualization and sets up the time series interpolation routines.
  int InitializeGeometry(const char *inputDir, int neuron0, int neuron1);

  int NumberOfThreads;
  char *Directory;
  int HasCoords;
  int HasData;
  int NeuronIds[2];
  int ReadNeuronIds[2];
  float CurrentThreshold;
  float VoltageThreshold;

  struct vtkInternals;
  vtkInternals *Internals;

private:
  vtkCvtNeuronReader(const vtkCvtNeuronReader&) = delete;
  void operator=(const vtkCvtNeuronReader&) = delete;
};

#endif
