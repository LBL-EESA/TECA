#include "teca_2d_component_area.h"
#include "teca_component_area_filter.h"
#include "teca_dataset_capture.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_system_interface.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_dataset_source.h"

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

    if (argc != 8)
    {
        cerr << "test_component_area_filter [nx] [ny] [num labels x] "
            << "[num labels y] [low thershold] [consecutive labels] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long nyl = atoi(argv[3]);
    unsigned long nxl = atoi(argv[4]);
    double low_threshold_value = atof(argv[5]);
    int consecutive_labels = atoi(argv[6]);
    string out_file = argv[7];

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

    std::string post_fix = "_area_filtered";

    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates("x", x);
    mesh->set_y_coordinates("y", y);
    mesh->set_z_coordinates("z", z);
    mesh->set_whole_extent(wext);
    mesh->set_extent(wext);
    mesh->set_time(1.0);
    mesh->set_time_step(0ul);
    mesh->get_point_arrays()->append("labels", cc);

    teca_metadata md;
    md.set("whole_extent", wext, 6);
    md.set("variables", std::vector<std::string>({"cc"}));
    md.set("number_of_time_steps", 1);
    md.set("index_initializer_key", std::string("number_of_time_steps"));
    md.set("index_request_key", std::string("time_step"));

    // build the pipeline
    p_teca_dataset_source source = teca_dataset_source::New();
    source->set_metadata(md);
    source->set_dataset(mesh);

    long background_id = consecutive_labels ? 0 : -2;

    p_teca_2d_component_area ca = teca_2d_component_area::New();
    ca->set_input_connection(source->get_output_port());
    ca->set_component_variable("labels");
    ca->set_contiguous_component_ids(consecutive_labels);
    ca->set_background_id(background_id);

    p_teca_component_area_filter caf = teca_component_area_filter::New();
    caf->set_input_connection(ca->get_output_port());
    caf->set_component_variable("labels");
    caf->set_component_ids_key("component_ids");
    caf->set_component_area_key("component_area");
    caf->set_low_area_threshold(low_threshold_value);
    caf->set_variable_post_fix(post_fix);
    caf->set_contiguous_component_ids(consecutive_labels);

    p_teca_dataset_capture cao = teca_dataset_capture::New();
    cao->set_input_connection(caf->get_output_port());

    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(cao->get_output_port());
    wri->set_file_name(out_file);

    wri->update();


    const_p_teca_dataset ds = cao->get_dataset();
    const_p_teca_cartesian_mesh cds = std::dynamic_pointer_cast<const teca_cartesian_mesh>(ds);
    const_p_teca_variant_array va = cds->get_point_arrays()->get("labels" + post_fix);

    teca_metadata mdo = ds->get_metadata();

    std::vector<int> filtered_label_id;

    std::vector<int> label_id;
    mdo.get("label_id", label_id);

    std::vector<double> area;
    mdo.get("area", area);

    std::vector<int> label_id_filtered;
    mdo.get("label_id" + post_fix, label_id_filtered);

    std::vector<double> area_filtered;
    mdo.get("area" + post_fix, area_filtered);

    cerr << "component areas" << endl;
    int n_labels = label_id.size();
    for (int i = 0; i < n_labels; ++i)
    {
        cerr << "label " << label_id[i] << " = " << area[i] << endl;
        if (area[i] < low_threshold_value)
        {
            filtered_label_id.push_back(label_id[i]);
        }
    }
    cerr << endl;

    cerr << "component areas filtered with low thershold area = " << low_threshold_value;
    cerr << endl;
    int n_labels_filtered = label_id_filtered.size();
    for (int i = 0; i < n_labels_filtered; ++i)
    {
        cerr << "label " << label_id_filtered[i] << " = " << area_filtered[i] << endl;
    }

    size_t n_filtered = filtered_label_id.size();
    size_t n_labels_total = va->size();

    NESTED_TEMPLATE_DISPATCH_I(const teca_variant_array_impl,
        va.get(),
        _LABEL,

        const NT_LABEL *p_labels_filtered = static_cast<TT_LABEL*>(va.get())->get();

        for (size_t i = 0; i < n_filtered; ++i)
        {
            NT_LABEL label = filtered_label_id[i];
            for (size_t j = 0; j < n_labels_total; j++)
            {
                if (label == p_labels_filtered[j])
                {
                    TECA_ERROR("Area filter failed!")
                    return -1;
                }
            }
        }
        
    )

    return 0;
}
