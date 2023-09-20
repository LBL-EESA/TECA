#include "teca_2d_component_area.h"
#include "teca_component_area_filter.h"
#include "teca_dataset_capture.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_system_interface.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_dataset_source.h"

#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace teca_variant_array_util;

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
    using array_t = teca_variant_array_impl<double>;

    coord_t dx = coord_t(360.)/coord_t(nx - 1);
    auto [x, px] = ::New<array_t>(nx);
    for (unsigned long i = 0; i < nx; ++i)
        px[i] = i*dx;

    coord_t dy = coord_t(180.)/coord_t(ny - 1);
    auto [y, py] = ::New<array_t>(ny);
    for (unsigned long i = 0; i < ny; ++i)
        py[i] = coord_t(-90.) + i*dy;

    auto z = array_t::New(1, coord_t(0));
    auto t = array_t::New(1, coord_t(1));

    // genrate nxl by nyl tiles
    unsigned long nxy = nx*ny;
    auto [cc, pcc] = ::New<teca_int_array>(nxy, int(0));
    for (unsigned long j = 0; j < ny; ++j)
    {
        int yl = int((py[j] + coord_t(90.)) / (coord_t(180.) / nyl)) % nyl;
        for (unsigned long i = 0; i < nx; ++i)
        {
            int xl = int(px[i] / (coord_t(360.) / nxl)) % nxl;
            int lab = yl*nxl + xl;
            pcc[j*nx + i] = consecutive_labels ? lab : labels[lab];
        }
    }

    unsigned long wext[] = {0, nx - 1, 0, ny - 1, 0, 0};

    std::string postfix = "_area_filtered";

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
    caf->set_variable_postfix(postfix);
    caf->set_contiguous_component_ids(consecutive_labels);

    p_teca_dataset_capture cao = teca_dataset_capture::New();
    cao->set_input_connection(caf->get_output_port());

    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(cao->get_output_port());
    wri->set_file_name(out_file);

    wri->update();


    const_p_teca_dataset ds = cao->get_dataset();
    const_p_teca_cartesian_mesh cds = std::dynamic_pointer_cast<const teca_cartesian_mesh>(ds);
    const_p_teca_variant_array va = cds->get_point_arrays()->get("labels" + postfix);

    teca_metadata mdo = ds->get_metadata();

    std::vector<int> comp_ids;
    if (mdo.get("component_ids", comp_ids))
        std::cerr << "ERROR: failed to get component_ids" << std::endl;

    std::vector<double> area;
    if (mdo.get("component_area", area))
        std::cerr << "ERROR: failed to get component areas" << std::endl;

    std::vector<int> comp_ids_filtered;
    if (mdo.get("component_ids" + postfix, comp_ids_filtered))
        std::cerr << "ERROR: failed to get filtered component_ids" << std::endl;

    std::vector<double> area_filtered;
    if (mdo.get("component_area" + postfix, area_filtered))
        std::cerr << "ERROR: failed to get filtered component areas" << std::endl;

#if defined(TECA_HAS_CUDA)
    cudaStreamSynchronize(cudaStreamPerThread);
#endif


    // get the output and check against the solution that we computed ourselves
    // below.
    std::set<int> base_in;
    std::set<int> base_out;

    // print the inputs
    int n_labels = comp_ids.size();

    std::cerr << "component areas" << std::endl;
    for (int i = 0; i < n_labels; ++i)
        std::cerr << "label " << comp_ids[i] << " = " << area[i] << std::endl;
    std::cerr << std::endl;

    // sort the inputs
    for (int i = 0; i < n_labels; ++i)
    {
        if (area[i] < low_threshold_value)
            base_out.insert(comp_ids[i]);
        else
            base_in.insert(comp_ids[i]);
    }

    // print the output
    int n_labels_filtered = comp_ids_filtered.size();

    std::cerr << "component areas filtered with low thershold area = "
        << low_threshold_value << std::endl;

    for (int i = 0; i < n_labels_filtered; ++i)
        std::cerr << "label " << comp_ids_filtered[i] << " = " << area_filtered[i] << std::endl;
    std::cerr << std::endl;

    // sort the output
    for (int i = 0; i < n_labels_filtered; ++i)
    {
        int label = comp_ids_filtered[i];
        if (!base_in.count(label) || base_out.count(label))
        {
            std::cerr << "ERROR: label " << i << " : " << comp_ids_filtered[i]
                << " was filtered incorrectly" << std::endl;
        }
    }

    std::cerr << base_out.size() << " removed. " << base_in.size() << " kept." << std::endl;

    return 0;
}
