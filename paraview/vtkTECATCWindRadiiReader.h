// .NAME vtkTECATCWindRadiiReader --
// .SECTION Description
//
// .SECTION See Also

#ifndef vtkTECATCWindRadiiReader_h
#define vtkTECATCWindRadiiReader_h

#include "vtkTECATableReader.h"
#include <map>
#include <utility>

class vtkTECATCWindRadiiReader : public vtkTECATableReader
{
public:
  static vtkTECATCWindRadiiReader *New();
  vtkTypeMacro(vtkTECATCWindRadiiReader,vtkTECATableReader);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  vtkSetStringMacro(XCoordinate);
  vtkGetStringMacro(XCoordinate);

  vtkSetStringMacro(YCoordinate);
  vtkGetStringMacro(YCoordinate);

  vtkSetStringMacro(TrackCoordinate);
  vtkGetStringMacro(TrackCoordinate);

  vtkSetStringMacro(CurveCoordinate);
  vtkGetStringMacro(CurveCoordinate);

  enum {GEOMETRY_MODE_CURVE = 0,
    GEOMETRY_MODE_WEDGE = 1};
  vtkSetMacro(GeometryMode, int);
  vtkGetMacro(GeometryMode, int);

protected:
  vtkTECATCWindRadiiReader();
  ~vtkTECATCWindRadiiReader();

  virtual int RequestInformation(vtkInformation *req,
    vtkInformationVector **inInfos, vtkInformationVector *outInfos);

  virtual int RequestData(vtkInformation *req,
    vtkInformationVector **inInfos, vtkInformationVector *outInfos);

private:
  char *XCoordinate;
  char *YCoordinate;
  char *TrackCoordinate;
  char *CurveCoordinate;
  int GeometryMode;

  typedef std::map<unsigned long, std::pair<unsigned long, unsigned long>> CurveMapT;
  CurveMapT CurveMap;

private:
  vtkTECATCWindRadiiReader(const vtkTECATCWindRadiiReader &); // Not implemented
  void operator=(const vtkTECATCWindRadiiReader &); // Not implemented
};

#endif
