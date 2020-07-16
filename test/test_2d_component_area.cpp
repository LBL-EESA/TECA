#include "teca_cartesian_mesh_source.h"
#include "teca_dataset_source.h"
#include "teca_normalize_coordinates.h"
#include "teca_connected_components.h"
#include "teca_2d_component_area.h"
#include "teca_dataset_capture.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

// genrates a field with nxl by nyl tiles, each having a unique integer id
struct tile_labeler
{
    unsigned long m_nxl;
    unsigned long m_nyl;
    int m_consecutive_labels;

    p_teca_variant_array operator()(const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &,
        double)
    {
        // 64 random integers between 1000 and 10000 for use as non consecutive labels
        int labels[] = {9345, 2548, 5704, 5132, 9786, 8329, 3667, 4332, 6232,
            3775, 2593, 7716, 1212, 9638, 9499, 9284, 6736, 7504, 8273, 5808, 7613,
            1405, 8849, 4405, 4777, 2927, 5903, 5294, 7344, 8335, 8186, 3343, 5341,
            7718, 7614, 6608, 1518, 6246, 7647, 4254, 7719, 6879, 1706, 8408, 1489,
            7054, 9304, 7218, 1275, 4784, 3670, 8859, 8877, 5367, 5340, 1521, 5815,
            5717, 6189, 5342, 4709, 6740, 1804, 6772};

        unsigned long nx = x->size();
        unsigned long ny = y->size();
        unsigned long nxy = nx*ny;
        p_teca_int_array cc = teca_int_array::New(nxy);
        int *pcc = cc->get();

        memset(pcc,0, nxy*sizeof(int));

         TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
             x.get(),

             const NT *px = std::static_pointer_cast<TT>(x)->get();
             const NT *py = std::static_pointer_cast<TT>(y)->get();

             for (unsigned long j = 0; j < ny; ++j)
             {
                 int yl = int((py[j] + NT(90.)) / (NT(180.) / m_nyl)) % m_nyl;
                 for (unsigned long i = 0; i < nx; ++i)
                 {
                     int xl = int(px[i] / (NT(360.) / m_nxl)) % m_nxl;
                     int lab = yl*m_nxl + xl;
                     pcc[j*nx + i] = m_consecutive_labels ? lab : labels[lab];
                 }
             }
             )

         return cc;
    }
};

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 8)
    {
        cerr << "test_2d_component_area [nx] [ny] [flip y] [num labels x] "
            << "[num labels y] [consecutive labels] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    int flip_y = atoi(argv[3]);
    unsigned long nyl = atoi(argv[4]);
    unsigned long nxl = atoi(argv[5]);
    int consecutive_labels = atoi(argv[6]);
    string out_file = argv[7];

    if (!consecutive_labels && (nxl*nyl > 64))
    {
        TECA_ERROR("Max 64 non-consecutive labels")
        return -1;
    }

    // allocate a mesh
    p_teca_cartesian_mesh_source source = teca_cartesian_mesh_source::New();
    source->set_whole_extents({0l, nx-1l, 0l, ny-1l, 0, 0, 0, 0});

    double y0 = flip_y ? 90.0 : -90.0;
    double y1 = flip_y ? -90.0 : 90.0;
    source->set_bounds({0., 360., y0, y1, 0., 0., 1., 1.});

    tile_labeler labeler = {nxl, nyl, consecutive_labels};
    source->append_field_generator({"labels", labeler});

    p_teca_normalize_coordinates norm_coord = teca_normalize_coordinates::New();
    norm_coord->set_input_connection(source->get_output_port());

    long background_id = consecutive_labels ? 0 : -2;

    p_teca_2d_component_area ca = teca_2d_component_area::New();
    ca->set_component_variable("labels");
    ca->set_contiguous_component_ids(consecutive_labels);
    ca->set_background_id(background_id);
    ca->set_input_connection(norm_coord->get_output_port());

    p_teca_dataset_capture cao = teca_dataset_capture::New();
    cao->set_input_connection(ca->get_output_port());

    p_teca_index_executive exe = teca_index_executive::New();
    exe->set_start_index(0);
    exe->set_end_index(0);

    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(cao->get_output_port());
    wri->set_executive(exe);
    wri->set_file_name(out_file);

    wri->update();

    const_p_teca_dataset ds = cao->get_dataset();
    teca_metadata mdo = ds->get_metadata();

    std::vector<int> label_id;
    mdo.get("component_ids", label_id);

    std::vector<double> area;
    mdo.get("component_area", area);

    cerr << "component area" << endl;
    double total_area = 0.f;
    int n_labels = area.size();
    for (int i = 0; i < n_labels; ++i)
    {
        cerr << "label " << label_id[i] << " = " << area[i] << endl;
        total_area += area[i];
    }
    cerr << "total area = " << total_area << endl;

    double dx = 360./(nx - 1.);
    double dy = 180./(ny - 1.);
    double re = 6378.1370;
    double pi = M_PI;
    double cell_x = dx*pi/180.;
    double half_cell_y = dy*pi/360.;
    double total_area_true = re*re*(2*pi - cell_x)*(cos(half_cell_y) - cos(pi - half_cell_y));
    double diff = total_area_true - total_area;
    cerr << "total area true = " << total_area_true << endl
        << "diff = " << diff << endl;

    if (fabs(diff) > 1e-03)
    {
        TECA_ERROR("Area calculation failed!")
        return -1;
    }

    return 0;
}
