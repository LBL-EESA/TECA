#include "teca_latitude_range_filter.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_dataset_capture.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_dataset_source.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>

using namespace std;
using namespace teca_variant_array_util;


bool isequal(double a, double b, double epsilon)
{
    return fabs(a - b) < epsilon;
}

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 6)
    {
        cerr << "test_latitude_range_filter [nx] [ny] [min_lat] [max_lat] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    double min_lat = atof(argv[3]);
    double max_lat = atof(argv[4]);
    string out_file = argv[5];

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

    unsigned long nxy = nx * ny;
    p_teca_double_array ones_grid = teca_double_array::New(nxy, double(1));

    unsigned long wext[] = {0, nx - 1, 0, ny - 1, 0, 0};

    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates("x", x);
    mesh->set_y_coordinates("y", y);
    mesh->set_z_coordinates("z", z);
    mesh->set_whole_extent(wext);
    mesh->set_extent(wext);
    mesh->set_time(1.0);
    mesh->set_time_step(0ul);
    mesh->get_point_arrays()->append("ones_grid", ones_grid);

    teca_metadata md;
    md.set("whole_extent", wext, 6);
    md.set("time_steps", std::vector<unsigned long>({0}));
    md.set("variables", std::vector<std::string>({"ones_grid"}));
    md.set("number_of_time_steps", 1);
    md.set("index_initializer_key", std::string("number_of_time_steps"));
    md.set("index_request_key", std::string("time_step"));

    // build the pipeline
    p_teca_dataset_source source = teca_dataset_source::New();
    source->set_metadata(md);
    source->set_dataset(mesh);

    p_teca_latitude_range_filter lrf = teca_latitude_range_filter::New();
    lrf->set_input_connection(source->get_output_port());
    lrf->set_min_lat(min_lat);
    lrf->set_max_lat(max_lat);
    lrf->append_filtered_variable("ones_grid");
    lrf->set_variable_postfix("_filtered");
    lrf->set_verbose(1);

    p_teca_dataset_capture dsc = teca_dataset_capture::New();
    dsc->set_input_connection(lrf->get_output_port());

    p_teca_index_executive exe = teca_index_executive::New();
    exe->set_verbose(1);
    exe->set_start_index(0);
    exe->set_end_index(0);

    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(dsc->get_output_port());
    wri->set_executive(exe);
    wri->set_file_name(out_file);

    wri->update();

    const_p_teca_dataset ds = dsc->get_dataset();
    const_p_teca_cartesian_mesh cds = std::dynamic_pointer_cast<const teca_cartesian_mesh>(ds);
    const_p_teca_variant_array va = cds->get_point_arrays()->get("ones_grid_filtered");

    auto [spda, p_filtered_array] = get_host_accessible<const array_t>(va);

    sync_host_access_any(va);

    // Iterate over all values and check if they match expected behavior
    // The filter uses abs(lat), so values should be 1.0 when min_lat <= abs(lat) <= max_lat
    // and 0.0 otherwise
    for (long j = 0; j < long(ny); ++j)
    {
        coord_t lat = py[j];
        coord_t abs_lat = fabs(lat);
        
        // Determine expected value based on latitude
        coord_t expected_val = (abs_lat >= min_lat && abs_lat <= max_lat) ? 1.0 : 0.0;
        
        // Check all longitude points at this latitude
        for (long i = 0; i < long(nx); ++i)
        {
            long idx = j * nx + i;
            coord_t actual_val = p_filtered_array[idx];
            
            if (!isequal(actual_val, expected_val, 1e-7))
            {
                TECA_ERROR("Mismatch at index (" << i << ", " << j << "), lat=" << lat
                    << ", abs_lat=" << abs_lat << ": expected " << expected_val
                    << " but got " << actual_val)
                return -1;
            }
        }
    }

    return 0;
}