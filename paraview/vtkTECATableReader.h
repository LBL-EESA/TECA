// .NAME vtkTECATableReader -- reads a table of cyclone tracks
// .SECTION Description
//
// .SECTION See Also

#ifndef vtkTECATableReader_h
#define vtkTECATableReader_h

#include "vtkPolyDataAlgorithm.h"
#include <teca_table.h> // for table

class vtkTECATableReader : public vtkPolyDataAlgorithm
{
public:
  static vtkTECATableReader *New();
  vtkTypeMacro(vtkTECATableReader,vtkPolyDataAlgorithm);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  void SetFileName(const char *fileName);
  vtkGetStringMacro(FileName);

  int CanReadFile(const char *file);

  int GetNumberOfTimeSteps();
  void GetTimeSteps(double *times);

  vtkSetStringMacro(TimeCoordinate);
  vtkGetStringMacro(TimeCoordinate);

  vtkSetMacro(SortTimeCoordinate, int);
  vtkGetMacro(SortTimeCoordinate, int);

protected:
  vtkTECATableReader();
  ~vtkTECATableReader();

  virtual int RequestInformation(vtkInformation *req,
    vtkInformationVector **inInfos, vtkInformationVector *outInfos);

  char *FileName;
  char *TimeCoordinate;
  int SortTimeCoordinate;

  const_p_teca_table Table;

private:
  vtkTECATableReader(const vtkTECATableReader &); // Not implemented
  void operator=(const vtkTECATableReader &); // Not implemented
};

#endif
