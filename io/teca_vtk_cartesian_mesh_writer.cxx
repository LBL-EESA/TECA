#include "teca_vtk_cartesian_mesh_writer.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"

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
#include "vtkRectilinearGrid.h"
#include "vtkXMLRectilinearGridWriter.h"

#include <iostream>
#include <sstream>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// helper for selecting the right vtk type
template <typename T> struct vtk_tt {};
#define VTK_TT_SPEC(_ctype, _vtype) \
template <>                         \
struct vtk_tt <_ctype>              \
{                                   \
    typedef _vtype type;            \
};
VTK_TT_SPEC(float, vtkFloatArray)
VTK_TT_SPEC(double, vtkDoubleArray)
VTK_TT_SPEC(char, vtkCharArray)
VTK_TT_SPEC(unsigned char, vtkUnsignedCharArray)
VTK_TT_SPEC(short, vtkShortArray)
VTK_TT_SPEC(unsigned short, vtkUnsignedShortArray)
VTK_TT_SPEC(int, vtkIntArray)
VTK_TT_SPEC(unsigned int, vtkUnsignedIntArray)
VTK_TT_SPEC(long, vtkLongArray)
VTK_TT_SPEC(unsigned long, vtkUnsignedLongArray)
VTK_TT_SPEC(long long, vtkLongLongArray)
VTK_TT_SPEC(unsigned long long, vtkUnsignedLongLongArray)

// --------------------------------------------------------------------------
teca_vtk_cartesian_mesh_writer::teca_vtk_cartesian_mesh_writer()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_vtk_cartesian_mesh_writer::~teca_vtk_cartesian_mesh_writer()
{}

// --------------------------------------------------------------------------
p_teca_dataset teca_vtk_cartesian_mesh_writer::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;

    p_teca_cartesian_mesh mesh
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(input_data[0]);

    if (!mesh)
    {
        TECA_ERROR("empty input")
        return nullptr;
    }

    unsigned long time_step;
    if (request.get("time_step", time_step))
    {
        TECA_ERROR("request missing \"time_step\"")
        return nullptr;
    }

    vector<int> extent;
    if (request.get("extent", extent))
    {
        TECA_ERROR("request missing \"extent\"")
        return nullptr;
    }

    vtkRectilinearGrid *rg = vtkRectilinearGrid::New();
    rg->SetExtent(&extent[0]);

    // transfer coordinates
    p_teca_variant_array x = mesh->get_x_coordinates();
    TEMPLATE_DISPATCH(teca_variant_array_impl, x.get(),
        TT *xx = static_cast<TT*>(x.get());
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(xx->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, xx->get(), sizeof(NT)*xx->size());
        rg->SetXCoordinates(a);
        a->Delete();
        );

    p_teca_variant_array y = mesh->get_y_coordinates();
    TEMPLATE_DISPATCH(teca_variant_array_impl, y.get(),
        TT *yy = static_cast<TT*>(y.get());
         vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(yy->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, yy->get(), sizeof(NT)*yy->size());
        rg->SetYCoordinates(a);
        a->Delete();
        )

    p_teca_variant_array z = mesh->get_z_coordinates();
    TEMPLATE_DISPATCH(teca_variant_array_impl, z.get(),
        TT *zz = static_cast<TT*>(z.get());
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(zz->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, zz->get(), sizeof(NT)*zz->size());
        rg->SetZCoordinates(a);
        a->Delete();
        )

    // transfor point data
    p_teca_array_collection pd = mesh->get_point_arrays();
    unsigned int n_arrays = pd->size();
    for (unsigned int i = 0; i< n_arrays; ++i)
    {
        p_teca_variant_array a = pd->get(i);
        string name = pd->get_name(i);

        TEMPLATE_DISPATCH(teca_variant_array_impl, a.get(),
            TT *aa = static_cast<TT*>(a.get());
            vtk_tt<NT>::type *b = vtk_tt<NT>::type::New();
            b->SetNumberOfTuples(aa->size());
            b->SetName(name.c_str());
            NT *p_b = b->GetPointer(0);
            memcpy(p_b, aa->get(), sizeof(NT)*aa->size());
            rg->GetPointData()->AddArray(b);
            b->Delete();
            )
    }

    ostringstream file_name;
    file_name << this->base_file_name <<  "_" << time_step << ".vtr";

    vtkXMLRectilinearGridWriter *w = vtkXMLRectilinearGridWriter::New();
    w->SetFileName(file_name.str().c_str());
    w->SetInputData(rg);
    w->Write();

    w->Delete();
    rg->Delete();

    return p_teca_dataset();
}
