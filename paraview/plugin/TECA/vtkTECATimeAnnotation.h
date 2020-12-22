// .NAME vtkTECATimeAnnotation
// .SECTION Description
// Provides calendaring functionality for CF2 compliant readers.
// This converts a double precision CF2 time value into a human
// readable string that will be automatically rendered by ParaView.
#ifndef vtkTECATimeAnnotation_h
#define vtkTECATimeAnnotation_h

#include "vtkTableAlgorithm.h"

class VTK_EXPORT vtkTECATimeAnnotation : public vtkTableAlgorithm
{
public:
  static vtkTECATimeAnnotation* New();
  vtkTypeMacro(vtkTECATimeAnnotation, vtkTableAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetMacro(IncludeYear, int)
  vtkGetMacro(IncludeYear, int)

  vtkSetMacro(IncludeMonth, int)
  vtkGetMacro(IncludeMonth, int)

  vtkSetMacro(IncludeDay, int)
  vtkGetMacro(IncludeDay, int)

  vtkSetMacro(IncludeHour, int)
  vtkGetMacro(IncludeHour, int)

  vtkSetMacro(IncludeMinute, int)
  vtkGetMacro(IncludeMinute, int)

  vtkSetMacro(IncludeSecond, int)
  vtkGetMacro(IncludeSecond, int)

  vtkSetStringMacro(DateSeparator)
  vtkGetStringMacro(DateSeparator)

  vtkSetStringMacro(TimeSeparator)
  vtkGetStringMacro(TimeSeparator)

protected:
  vtkTECATimeAnnotation();
  ~vtkTECATimeAnnotation();

  int RequestData(vtkInformation* req, vtkInformationVector** inVec,
    vtkInformationVector* outVec) override;

  int FillInputPortInformation(int port, vtkInformation* info) override;
  int FillOutputPortInformation(int port, vtkInformation* info) override;

private:
  int IncludeYear;
  int IncludeMonth;
  int IncludeDay;
  int IncludeHour;
  int IncludeMinute;
  int IncludeSecond;
  char *DateSeparator;
  char *TimeSeparator;

private:
  vtkTECATimeAnnotation(const vtkTECATimeAnnotation&); // Not implemented
  void operator=(const vtkTECATimeAnnotation&); // Not implemented
};

#endif
