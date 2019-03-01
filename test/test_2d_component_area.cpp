#include "teca_cf_reader.h"
#include "teca_mask.h"
#include "teca_l2_norm.h"
#include "teca_connected_components.h"
#include "teca_2d_component_area.h"
#include "teca_dataset_capture.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_time_step_executive.h"
#include "teca_system_interface.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_dataset_source.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

// 64 random integers between 1000 and 10000 for use as non consecutive labels
int labels[] = {9345, 2548, 5704, 5132, 9786, 8329, 3667, 4332, 6232,
    3775, 2593, 7716, 1212, 9638, 9499, 9284, 6736, 7504, 8273, 5808, 7613,
    1405, 8849, 4405, 4777, 2927, 5903, 5294, 7344, 8335, 8186, 3343, 5341,
    7718, 7614, 6608, 1518, 6246, 7647, 4254, 7719, 6879, 1706, 8408, 1489,
    7054, 9304, 7218, 1275, 4784, 3670, 8859, 8877, 5367, 5340, 1521, 5815,
    5717, 6189, 5342, 4709, 6740, 1804, 6772};

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 7)
    {
        cerr << "test_2d_component_area [nx] [ny] [num labels x] "
            << "[num labels y] [consecutive labels] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long nyl = atoi(argv[3]);
    unsigned long nxl = atoi(argv[4]);
    int consecutive_labels = atoi(argv[5]);
    string out_file = argv[6];

    if (!consecutive_labels && (nxl*nyl > 64))
    {
        TECA_ERROR("Max 64 non-consecutive labels")
        return -1;
    }

    // allocate a mesh
    // coordinate axes
    using coord_t = double;
    coord_t dx = coord_t(360.)/coord_t(nx - 1);
    p_teca_variant_array_impl<coord_t> x = teca_variant_array_impl<coord_t>::New(nx);
    coord_t *px = x->get();
    for (unsigned long i = 0; i < nx; ++i)
        px[i] = i*dx;

    coord_t dy = coord_t(180.)/coord_t(ny - 1);
    p_teca_variant_array_impl<coord_t> y = teca_variant_array_impl<coord_t>::New(ny);
    coord_t *py = y->get();
    for (unsigned long i = 0; i < ny; ++i)
        py[i] = coord_t(-90.) + i*dy;

    p_teca_variant_array_impl<coord_t> z = teca_variant_array_impl<coord_t>::New(1);
    z->set(0, 0.f);

    p_teca_variant_array_impl<coord_t> t = teca_variant_array_impl<coord_t>::New(1);
    t->set(0, 1.f);

    // genrate nxl by nyl tiles
    unsigned long nxy = nx*ny;
    p_teca_int_array cc = teca_int_array::New(nxy);
    int *p_cc = cc->get();

    memset(p_cc,0, nxy*sizeof(int));

    for (unsigned long j = 0; j < ny; ++j)
    {
        int yl = int((py[j] + coord_t(90.)) / (coord_t(180.) / nyl)) % nyl;
        for (unsigned long i = 0; i < nx; ++i)
        {
            int xl = int(px[i] / (coord_t(360.) / nxl)) % nxl;
            int lab = yl*nxl + xl;
            p_cc[j*nx + i] = consecutive_labels ? lab : labels[lab];
        }
    }

    unsigned long wext[] = {0, nx - 1, 0, ny - 1, 0, 0};

    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates(x);
    mesh->set_y_coordinates(y);
    mesh->set_z_coordinates(z);
    mesh->set_whole_extent(wext);
    mesh->set_extent(wext);
    mesh->set_time(1.0);
    mesh->set_time_step(0ul);
    mesh->get_point_arrays()->append("labels", cc);

    teca_metadata md;
    md.insert("whole_extent", wext, 6);
    md.insert("time_steps", std::vector<unsigned long>({0}));
    md.insert("variables", std::vector<std::string>({"cc"}));
    md.insert("number_of_time_steps", 1);

    // build the pipeline
    p_teca_dataset_source source = teca_dataset_source::New();
    source->set_metadata(md);
    source->set_dataset(mesh);

    p_teca_2d_component_area ca = teca_2d_component_area::New();
    ca->set_label_variable("labels");
    ca->set_contiguous_label_ids(consecutive_labels);
    ca->set_input_connection(source->get_output_port());

    p_teca_dataset_capture cao = teca_dataset_capture::New();
    cao->set_input_connection(ca->get_output_port());

    p_teca_time_step_executive exe = teca_time_step_executive::New();
    exe->set_first_step(0);
    exe->set_last_step(0);

    p_teca_vtk_cartesian_mesh_writer wri = teca_vtk_cartesian_mesh_writer::New();
    wri->set_input_connection(cao->get_output_port());
    wri->set_executive(exe);
    wri->set_file_name(out_file);

    wri->update();

    const_p_teca_dataset ds = cao->get_dataset();
    teca_metadata mdo = ds->get_metadata();

    std::vector<int> label_id;
    mdo.get("teca_2d_component_area::label_id", label_id);

    std::vector<double> area;
    mdo.get("teca_2d_component_area::area", area);

    cerr << "component area" << endl;
    double total_area = 0.f;
    int n_labels = area.size();
    for (int i = 0; i < n_labels; ++i)
    {
        cerr << "label " << label_id[i] << " = " << area[i] << endl;
        total_area += area[i];
    }
    cerr << "total area = " << total_area << endl;

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
