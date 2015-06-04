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

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

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

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_vtk_cartesian_mesh_writer::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for " + prefix + "(teca_vtk_cartesian_mesh_writer)");

    opts.add_options()
        TECA_POPTS_GET(string, prefix, base_file_name, "base path/name to write series to")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_vtk_cartesian_mesh_writer::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, string, prefix, base_file_name)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_vtk_cartesian_mesh_writer::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;

    const_p_teca_cartesian_mesh mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!mesh)
    {
        TECA_ERROR("empty input")
        return nullptr;
    }

    unsigned long time_step = 0;
    if (mesh->get_time_step(time_step) && request.get("time_step", time_step))
    {
        TECA_ERROR("request missing \"time_step\"")
        return nullptr;
    }

    vector<unsigned long> extent(6, 0);
    if (mesh->get_extent(extent) && request.get("extent", extent))
    {
        TECA_ERROR("request missing \"extent\"")
        return nullptr;
    }

    vtkRectilinearGrid *rg = vtkRectilinearGrid::New();
    rg->SetExtent(
        extent[0], extent[1],
        extent[2], extent[3],
        extent[4], extent[5]);

    // transfer coordinates
    const_p_teca_variant_array x = mesh->get_x_coordinates();
    TEMPLATE_DISPATCH(const teca_variant_array_impl, x.get(),
        const TT *xx = static_cast<const TT*>(x.get());
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(xx->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, xx->get(), sizeof(NT)*xx->size());
        rg->SetXCoordinates(a);
        a->Delete();
        );

    const_p_teca_variant_array y = mesh->get_y_coordinates();
    TEMPLATE_DISPATCH(const teca_variant_array_impl, y.get(),
        const TT *yy = static_cast<const TT*>(y.get());
         vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(yy->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, yy->get(), sizeof(NT)*yy->size());
        rg->SetYCoordinates(a);
        a->Delete();
        )

    const_p_teca_variant_array z = mesh->get_z_coordinates();
    TEMPLATE_DISPATCH( const teca_variant_array_impl, z.get(),
        const TT *zz = static_cast<const TT*>(z.get());
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(zz->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, zz->get(), sizeof(NT)*zz->size());
        rg->SetZCoordinates(a);
        a->Delete();
        )

    // transform point data
    const_p_teca_array_collection pd = mesh->get_point_arrays();
    unsigned int n_arrays = pd->size();
    for (unsigned int i = 0; i< n_arrays; ++i)
    {
        const_p_teca_variant_array a = pd->get(i);
        string name = pd->get_name(i);

        TEMPLATE_DISPATCH(const teca_variant_array_impl, a.get(),
            const TT *aa = static_cast<const TT*>(a.get());
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
