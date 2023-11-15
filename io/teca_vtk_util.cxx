#include "teca_vtk_util.h"

#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

#include <vector>
#include <string>

#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
#include "vtkDataObject.h"
#include "vtkStringArray.h"
#include "vtkRectilinearGrid.h"
#include "vtkInformation.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#endif

using namespace teca_variant_array_util;

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
    VARIANT_ARRAY_DISPATCH(x.get(),
        auto [spx, px] = get_host_accessible<CTT>(x);
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(x->size());
        NT *p_a = a->GetPointer(0);
        sync_host_access_any(x);
        memcpy(p_a, px, sizeof(NT)*x->size());
        output->SetXCoordinates(a);
        a->Delete();
        );

    const_p_teca_variant_array y = input->get_y_coordinates();
    VARIANT_ARRAY_DISPATCH(y.get(),
        auto [spy, py] = get_host_accessible<CTT>(y);
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(y->size());
        NT *p_a = a->GetPointer(0);
        sync_host_access_any(y);
        memcpy(p_a, py, sizeof(NT)*y->size());
        output->SetYCoordinates(a);
        a->Delete();
        )

    const_p_teca_variant_array z = input->get_z_coordinates();
    VARIANT_ARRAY_DISPATCH(z.get(),
        auto [spz, pz] = get_host_accessible<CTT>(z);
        vtk_tt<NT>::type *a = vtk_tt<NT>::type::New();
        a->SetNumberOfTuples(z->size());
        NT *p_a = a->GetPointer(0);
        sync_host_access_any(z);
        memcpy(p_a, pzz, sizeof(NT)*z->size());
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

        VARIANT_ARRAY_DISPATCH(a.get(),
            auto [spa, pa] = get_host_accessible<CTT>(a);
            vtk_tt<NT>::type *b = vtk_tt<NT>::type::New();
            b->SetNumberOfTuples(a->size());
            b->SetName(name.c_str());
            NT *p_b = b->GetPointer(0);
            sync_host_access_any(a);
            memcpy(p_b, pa, sizeof(NT)*a->size());
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


// --------------------------------------------------------------------------
void partition_writer::write()
{
    int nc = m_types.size();
    int np = 8*nc;

    FILE *fh = fopen(m_file_name.c_str(), "w");

    fputs("# vtk DataFile Version 2.0\n", fh);
    fputs("TECA spatial parallel partitioning\n", fh);
    fputs("ASCII\n", fh);
    fputs("DATASET UNSTRUCTURED_GRID\n\n", fh);

    fprintf(fh, "POINTS %d double\n", 8*nc);
    for (int i = 0; i < np; ++i)
    {
        int ii = 3*i;
        fprintf(fh, "%g %g %g", m_points[ii], m_points[ii + 1], m_points[ii + 2]);
        if ((i + 1) % 8 == 0)
        {
            fputs("\n", fh);
        }
        else
        {
            fputs("  ", fh);
        }
    }
    if (np % 8)
    {
        fputs("\n", fh);
    }

    fprintf(fh, "\nCELLS %d %d\n", nc, int(m_cells.size()));
    for (int i = 0; i < nc; ++i)
    {
        int ii = 9*i;
        for (int j = 0; j < 9; ++j)
        {
            fprintf(fh, "%ld ", m_cells[ii + j]);
        }
        fputs("\n", fh);
    }

    fprintf(fh, "\nCELL_TYPES %d\n", nc);
    for (int i = 0; i < nc; ++i)
    {
        fprintf(fh, "%d\n", m_types[i]);
    }

    fprintf(fh, "\nCELL_DATA %d\n", nc);
    fputs("SCALARS owner float 1\n", fh);
    fputs("LOOKUP_TABLE default\n", fh);
    for (int i = 0; i < nc; ++i)
    {
        fprintf(fh, "%d ", m_owner[i]);
        if ((i + 1) % 18 == 0)
        {
            fputs("\n", fh);
        }
    }
    if (nc % 18)
    {
        fputs("\n", fh);
    }

    fclose(fh);
}

// --------------------------------------------------------------------------
void partition_writer::add_partition(const unsigned long extent[6],
    const unsigned long temporal_extent[2], int owner)
{
   /* VTK hexahedron cell winding
       7  +-------------------+ 6
         /|                  /|
        / |                 / |
       /  |                /  |
    4 +------------------+/5  |
      |   |              |    |
      | 3 +--------------|--- + 2
    Y |  /               |   /
      | /                |  /  T
      |/                 | /
    0 +------------------+/ 1
               X
    */
    int cid = m_types.size();

    m_owner.push_back(owner);

    m_types.push_back(12); // VTK hexahedron

    int pid = 8*cid;
    m_cells.push_back(8);
    m_cells.push_back(pid    );
    m_cells.push_back(pid + 1);
    m_cells.push_back(pid + 2);
    m_cells.push_back(pid + 3);
    m_cells.push_back(pid + 4);
    m_cells.push_back(pid + 5);
    m_cells.push_back(pid + 6);
    m_cells.push_back(pid + 7);

    // 0
    m_points.push_back(m_x0 + m_dx*extent[0]);
    m_points.push_back(m_y0 + m_dy*extent[2]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[0]);

    // 1
    m_points.push_back(m_x0 + m_dx*extent[1]);
    m_points.push_back(m_y0 + m_dy*extent[2]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[0]);

    // 2
    m_points.push_back(m_x0 + m_dx*extent[1]);
    m_points.push_back(m_y0 + m_dy*extent[2]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[1]);

    // 3
    m_points.push_back(m_x0 + m_dx*extent[0]);
    m_points.push_back(m_y0 + m_dy*extent[2]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[1]);

    // 4
    m_points.push_back(m_x0 + m_dx*extent[0]);
    m_points.push_back(m_y0 + m_dy*extent[3]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[0]);

    // 5
    m_points.push_back(m_x0 + m_dx*extent[1]);
    m_points.push_back(m_y0 + m_dy*extent[3]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[0]);

    // 6
    m_points.push_back(m_x0 + m_dx*extent[1]);
    m_points.push_back(m_y0 + m_dy*extent[3]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[1]);

    // 7
    m_points.push_back(m_x0 + m_dx*extent[0]);
    m_points.push_back(m_y0 + m_dy*extent[3]);
    m_points.push_back(m_t0 + m_dt*temporal_extent[1]);
}

}
