// .NAME vtkTECATCCandidateTableReader --
// .SECTION Description
//
// .SECTION See Also

#ifndef vtkTECATCCandidateTableReader_h
#define vtkTECATCCandidateTableReader_h

#include "vtkPolyDataAlgorithm.h"
#include <map>
#include <utility>
#include <teca_table.h> // for table

class vtkTECATCCandidateTableReader : public vtkPolyDataAlgorithm
{
public:
  static vtkTECATCCandidateTableReader *New();
  vtkTypeMacro(vtkTECATCCandidateTableReader,vtkPolyDataAlgorithm);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  vtkSetStringMacro(FileName);
  vtkGetStringMacro(FileName);

  int CanReadFile(const char *file);

  int GetNumberOfTimeSteps();
  void GetTimeSteps(double *times);

  vtkSetStringMacro(XCoordinate);
  vtkGetStringMacro(XCoordinate);

  vtkSetStringMacro(YCoordinate);
  vtkGetStringMacro(YCoordinate);

  vtkSetStringMacro(ZCoordinate);
  vtkGetStringMacro(ZCoordinate);

  vtkSetStringMacro(TimeCoordinate);
  vtkGetStringMacro(TimeCoordinate);

protected:
  vtkTECATCCandidateTableReader();
  ~vtkTECATCCandidateTableReader();

  virtual int RequestInformation(
    vtkInformation *req, vtkInformationVector **inInfos,
    vtkInformationVector *outInfos);

  virtual int RequestData(
    vtkInformation *req, vtkInformationVector **inInfos,
    vtkInformationVector *outInfos);

  int GetTimeStepId(vtkInformation *inInfo,
    vtkInformation *outInfo);

private:
  char *FileName;
  char *XCoordinate;
  char *YCoordinate;
  char *ZCoordinate;
  char *TimeCoordinate;

  const_p_teca_table Table;
  std::map<double, std::pair<size_t, size_t>> TimeRows;

private:
  vtkTECATCCandidateTableReader(const vtkTECATCCandidateTableReader &); // Not implemented
  void operator=(const vtkTECATCCandidateTableReader &); // Not implemented
};

#endif
