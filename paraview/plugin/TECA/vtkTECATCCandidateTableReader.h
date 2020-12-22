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
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetStringMacro(FileName);
  vtkGetStringMacro(FileName);

  int CanReadFile(const char *file);

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

  int RequestInformation(vtkInformation *req, vtkInformationVector **inInfos,
    vtkInformationVector *outInfos) override;

  int RequestData(vtkInformation *req, vtkInformationVector **inInfos,
    vtkInformationVector *outInfos) override;

  int GetTimeStepId(vtkInformation *inInfo, vtkInformation *outInfo);

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
