#ifndef teca_vtk_util_h
#define teca_vtk_util_h

/// @file

#include "teca_config.h"
#include "teca_cartesian_mesh.h"

#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkCharArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkShortArray.h"
#include "vtkUnsignedShortArray.h"
#include "vtkIntArray.h"
#include "vtkUnsignedIntArray.h"
#include "vtkLongArray.h"
#include "vtkUnsignedLongArray.h"
#include "vtkLongLongArray.h"
#include "vtkUnsignedLongLongArray.h"
#include "vtkPointData.h"
class vtkRectilinearGrid;
#else
using vtkFloatArray = void*;
using vtkDoubleArray = void*;
using vtkCharArray = void*;
using vtkUnsignedCharArray = void*;
using vtkShortArray = void*;
using vtkUnsignedShortArray = void*;
using vtkIntArray = void*;
using vtkUnsignedIntArray = void*;
using vtkLongArray = void*;
using vtkUnsignedLongArray = void*;
using vtkLongLongArray = void*;
using vtkUnsignedLongLongArray = void*;
class vtkRectilinearGrid;
#endif


/// Codes dealing with VTK
namespace teca_vtk_util
{

/// @cond

// traits class for naming and/or selecting
// the VTK type given a C++ type
template <typename T> struct TECA_EXPORT vtk_tt {};
#define VTK_TT_SPEC(_ctype, _ctypestr, _vtype, _fmt)    \
template <>                                             \
struct vtk_tt <_ctype>                                  \
{                                                       \
    using type = _vtype;                                \
                                                        \
    static constexpr const char *str()                  \
    { return #_ctypestr; }                              \
                                                        \
    static constexpr const char *fmt()                  \
    { return _fmt; }                                    \
};
VTK_TT_SPEC(float, float, vtkFloatArray, "%g")
VTK_TT_SPEC(double, double, vtkDoubleArray, "%g")
VTK_TT_SPEC(char, char, vtkCharArray, "%hhi")
VTK_TT_SPEC(unsigned char, unsigned_char, vtkUnsignedCharArray, "%hhu")
VTK_TT_SPEC(short, short, vtkShortArray, "%hi")
VTK_TT_SPEC(unsigned short, unsigned_short, vtkUnsignedShortArray, "%hu")
VTK_TT_SPEC(int, int, vtkIntArray, "%i")
VTK_TT_SPEC(unsigned int, unsigned_int, vtkUnsignedIntArray, "%u")
VTK_TT_SPEC(long, long, vtkLongArray, "%li")
VTK_TT_SPEC(unsigned long, unsigned_long, vtkUnsignedLongArray, "%lu")
VTK_TT_SPEC(long long, long_long, vtkLongLongArray, "%lli")
VTK_TT_SPEC(unsigned long long, unsigned_long_long, vtkUnsignedLongLongArray, "%llu")

/// @endcond

/// deep copy input mesh into the VTK object. @returns 0 if successful
TECA_EXPORT
int deep_copy(vtkRectilinearGrid *output,
    const_p_teca_cartesian_mesh input);


/** write spatio-temporal partitions as a VTK unstructured grid that can be
 * visualized in ParaView
 */
class TECA_EXPORT partition_writer
{
public:
    /// create the writer
    partition_writer(const std::string &file_name,
        double x0, double y0, double t0,
        double dx, double dy, double dt) :
        m_file_name(file_name),
        m_x0(x0), m_y0(y0), m_t0(t0),
        m_dx(dx), m_dy(dy), m_dt(dt) {}

    /// add a partition
    void add_partition(const unsigned long extent[6],
        const unsigned long temporal_extent[2], int owner);

    /// write the vtk dataset
    void write();

private:
    std::string m_file_name;
    std::vector<double> m_points;
    std::vector<long> m_cells;
    std::vector<unsigned char> m_types;
    std::vector<int> m_owner;
    double m_x0;
    double m_y0;
    double m_t0;
    double m_dx;
    double m_dy;
    double m_dt;
};


};

#endif
