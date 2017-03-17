// .NAME vtkTECATCCandidateReader --
// .SECTION Description
//
// .SECTION See Also

#ifndef vtkTECATCCandidateReader_h
#define vtkTECATCCandidateReader_h

#include "vtkTECATableReader.h"

class vtkTECATCCandidateReader : public vtkTECATableReader
{
public:
  static vtkTECATCCandidateReader *New();
  vtkTypeMacro(vtkTECATCCandidateReader,vtkTECATableReader);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  vtkSetStringMacro(XCoordinate);
  vtkGetStringMacro(XCoordinate);

  vtkSetStringMacro(YCoordinate);
  vtkGetStringMacro(YCoordinate);

  vtkSetStringMacro(ZCoordinate);
  vtkGetStringMacro(ZCoordinate);

protected:
  vtkTECATCCandidateReader();
  ~vtkTECATCCandidateReader();

  virtual int RequestData(vtkInformation *req,
    vtkInformationVector **inInfos, vtkInformationVector *outInfos);

  int GetTimeStepId(vtkInformation *inInfo,
    vtkInformation *outInfo);

private:
  char *FileName;
  char *XCoordinate;
  char *YCoordinate;
  char *ZCoordinate;
  char *TimeCoordinate;

  std::map<double, std::pair<size_t, size_t>> TimeRows;

private:
  vtkTECATCCandidateReader(const vtkTECATCCandidateReader &); // Not implemented
  void operator=(const vtkTECATCCandidateReader &); // Not implemented
};

#endif
