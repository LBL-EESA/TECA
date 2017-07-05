// .NAME vtkTECATCTrackTableReader -- reads a table of cyclone tracks
// .SECTION Description
//
// .SECTION See Also

#ifndef vtkTECATCTrackTableReader_h
#define vtkTECATCTrackTableReader_h

#include "vtkPolyDataAlgorithm.h"
#include <map>
#include <utility>
#include <teca_table.h> // for table

class vtkTECATCTrackTableReader : public vtkPolyDataAlgorithm
{
public:
  static vtkTECATCTrackTableReader *New();
  vtkTypeMacro(vtkTECATCTrackTableReader,vtkPolyDataAlgorithm);
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

  vtkSetStringMacro(TrackCoordinate);
  vtkGetStringMacro(TrackCoordinate);

protected:
  vtkTECATCTrackTableReader();
  ~vtkTECATCTrackTableReader();

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
  char *TrackCoordinate;

  p_teca_table Table;
  std::vector<std::pair<size_t, size_t>> TrackRows;

private:
  vtkTECATCTrackTableReader(const vtkTECATCTrackTableReader &); // Not implemented
  void operator=(const vtkTECATCTrackTableReader &); // Not implemented
};

#endif
