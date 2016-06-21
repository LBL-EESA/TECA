#include "teca_vtk_util.h"

#include "teca_common.h"
#include "teca_variant_array.h"

#include <vector>
#include <string>

#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
#include "vtkDataObject.h"
#include "vtkStringArray.h"
#include "vtkRectilinearGrid.h"
#include "vtkInformation.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#endif

namespace teca_vtk_util
{
// **************************************************************************
int deep_copy(vtkRectilinearGrid *output,
    const_p_teca_cartesian_mesh input)
{
#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
    // set the extent
    std::vector<unsigned long> extent(6, 0);
    if (input->get_extent(extent))
    {
        TECA_ERROR("input missing \"extent\"")
        return -1;
    }

    output->SetExtent(extent[0], extent[1],
        extent[2], extent[3], extent[4], extent[5]);

    // set the time
    double time;
    if (input->get_time(time))
    {
        TECA_ERROR("input missing \"time\"")
        return -1;
    }

    output->GetInformation()->Set(vtkDataObject::DATA_TIME_STEP(), time);

    // transfer coordinates
    const_p_teca_variant_array x = input->get_x_coordinates();
    TEMPLATE_DISPATCH(const teca_variant_array_impl, x.get(),
        const TT *xx = static_cast<const TT*>(x.get());
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(xx->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, xx->get(), sizeof(NT)*xx->size());
        output->SetXCoordinates(a);
        a->Delete();
        );

    const_p_teca_variant_array y = input->get_y_coordinates();
    TEMPLATE_DISPATCH(const teca_variant_array_impl, y.get(),
        const TT *yy = static_cast<const TT*>(y.get());
         vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(yy->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, yy->get(), sizeof(NT)*yy->size());
        output->SetYCoordinates(a);
        a->Delete();
        )

    const_p_teca_variant_array z = input->get_z_coordinates();
    TEMPLATE_DISPATCH( const teca_variant_array_impl, z.get(),
        const TT *zz = static_cast<const TT*>(z.get());
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(zz->size());
        NT *p_a = a->GetPointer(0);
        memcpy(p_a, zz->get(), sizeof(NT)*zz->size());
        output->SetZCoordinates(a);
        a->Delete();
        )

    // transform point data
    const_p_teca_array_collection pd = input->get_point_arrays();
    unsigned int n_arrays = pd->size();
    for (unsigned int i = 0; i< n_arrays; ++i)
    {
        const_p_teca_variant_array a = pd->get(i);
        std::string name = pd->get_name(i);

        TEMPLATE_DISPATCH(const teca_variant_array_impl, a.get(),
            const TT *aa = static_cast<const TT*>(a.get());
            vtk_tt<NT>::type *b = vtk_tt<NT>::type::New();
            b->SetNumberOfTuples(aa->size());
            b->SetName(name.c_str());
            NT *p_b = b->GetPointer(0);
            memcpy(p_b, aa->get(), sizeof(NT)*aa->size());
            output->GetPointData()->AddArray(b);
            b->Delete();
            )
    }

    // add calendaring metadata
    std::string calendar;
    input->get_calendar(calendar);
    vtkStringArray *sarr = vtkStringArray::New();
    sarr->SetName("calendar");
    sarr->SetNumberOfTuples(1);
    sarr->SetValue(0, calendar);
    output->GetFieldData()->AddArray(sarr);
    sarr->Delete();

    std::string timeUnits;
    input->get_time_units(timeUnits);
    sarr = vtkStringArray::New();
    sarr->SetName("time_units");
    sarr->SetNumberOfTuples(1);
    sarr->SetValue(0, timeUnits);
    output->GetFieldData()->AddArray(sarr);
    sarr->Delete();
#else
    (void)output;
    (void)input;
#endif
    return 0;
}
};
