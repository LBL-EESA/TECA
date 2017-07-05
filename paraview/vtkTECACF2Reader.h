// .NAME vtkTECACF2Reader --
// .SECTION Description
//
// .SECTION See Also

#ifndef vtkTECACF2Reader_h
#define vtkTECACF2Reader_h

#include "vtkRectilinearGridAlgorithm.h"

#include "teca_metadata.h"
#include "teca_cf_reader.h"

#include <map>
#include <string>

class vtkTECACF2Reader : public vtkRectilinearGridAlgorithm
{
public:
  static vtkTECACF2Reader *New();
  vtkTypeMacro(vtkTECACF2Reader,vtkRectilinearGridAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  int CanReadFile(const char *file);

  void SetFileName(const char *file);
  vtkGetStringMacro(FileName);

  void SetInputRegex(const char *regex);
  vtkGetStringMacro(InputRegex);

  void SetXCoordinate(const char *xCoord);
  vtkGetStringMacro(XCoordinate);

  void SetYCoordinate(const char *yCoord);
  vtkGetStringMacro(YCoordinate);

  void SetZCoordinate(const char *zCoord);
  vtkGetStringMacro(ZCoordinate);

  void SetTimeCoordinate(const char *timeCoord);
  vtkGetStringMacro(TimeCoordinate);

  // Description:
  // Array selection.
  void SetPointArrayStatus(const char *name, int status);
  int GetPointArrayStatus(const char *name);
  int GetNumberOfPointArrays();
  const char* GetPointArrayName(int idx);
  void ClearPointArrayStatus();

protected:
  vtkTECACF2Reader();
  ~vtkTECACF2Reader();

  int RequestInformation(vtkInformation *req, vtkInformationVector **inInfos,
    vtkInformationVector *outInfos) override;

  int RequestData(vtkInformation *req, vtkInformationVector **inInfos,
    vtkInformationVector *outInfos) override;

  int GetTimeStepId(vtkInformation *inInfo, vtkInformation *outInfo);

  vtkSetStringMacro(BaseDir);
  vtkGetStringMacro(BaseDir);

private:
  char *FileName;
  char *BaseDir;
  char *InputRegex;
  char *XCoordinate;
  char *YCoordinate;
  char *ZCoordinate;
  char *TimeCoordinate;
  p_teca_cf_reader Reader;
  teca_metadata Metadata;
  std::map<std::string, int> ActiveArrays;
  int UpdateMetadata;

private:
  vtkTECACF2Reader(const vtkTECACF2Reader &); // Not implemented
  void operator=(const vtkTECACF2Reader &); // Not implemented
};

#endif
