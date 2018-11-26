// .NAME vtkTECATCTrackReader -- reads a table of cyclone tracks
// .SECTION Description
//
// .SECTION See Also

#ifndef vtkTECATCTrackReader_h
#define vtkTECATCTrackReader_h

#include "vtkTECATableReader.h"
#include <map>
#include <utility>

class vtkTECATCTrackReader : public vtkTECATableReader
{
public:
  static vtkTECATCTrackReader *New();
  vtkTypeMacro(vtkTECATCTrackReader,vtkTECATableReader);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  vtkSetStringMacro(XCoordinate);
  vtkGetStringMacro(XCoordinate);

  vtkSetStringMacro(YCoordinate);
  vtkGetStringMacro(YCoordinate);

  vtkSetStringMacro(TrackCoordinate);
  vtkGetStringMacro(TrackCoordinate);

protected:
  vtkTECATCTrackReader();
  ~vtkTECATCTrackReader();

  virtual int RequestInformation(vtkInformation *req,
    vtkInformationVector **inInfos, vtkInformationVector *outInfos);

  virtual int RequestData(vtkInformation *req,
    vtkInformationVector **inInfos, vtkInformationVector *outInfos);

private:
  char *XCoordinate;
  char *YCoordinate;
  char *TrackCoordinate;

  typedef std::map<unsigned long, std::pair<unsigned long, unsigned long>> TrackRowMapT;
  TrackRowMapT TrackRows;

private:
  vtkTECATCTrackReader(const vtkTECATCTrackReader &); // Not implemented
  void operator=(const vtkTECATCTrackReader &); // Not implemented
};

#endif
