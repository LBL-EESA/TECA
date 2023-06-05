#include "teca_cartesian_mesh_source.h"
#include "teca_shape_file_mask.h"
#include "teca_normalize_coordinates.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_index_executive.h"
#include "teca_dataset_diff.h"
#include "teca_system_util.h"

#include <string>
#include <cstring>


int main(int argc, char **argv)
{
    if (argc != 6)
    {
        std::cerr << "usage:" << std::endl
            << "test_shape_file_mask [shape file] [mask name] [nx] [ny] [out file]"
            << std::endl;
        return 0;
    }

    std::string shape_file = argv[1];
    std::string mask_name = argv[2];
    unsigned long nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    std::string out_file = argv[5];

    p_teca_cartesian_mesh_source src = teca_cartesian_mesh_source::New();
    src->set_whole_extents({0lu, nx - 1lu, 0, ny - 1lu, 0lu, 0lu, 0lu, 0lu});
    src->set_bounds({-180.0, 180.0, -90.0, 90.0, 0.0, 0.0, 0.0, 0.0});
    src->set_t_axis_variable("");

    p_teca_shape_file_mask mask = teca_shape_file_mask::New();
    mask->set_input_connection(src->get_output_port());
    mask->set_shape_file(shape_file);
    mask->set_mask_variables({mask_name});
    mask->set_normalize_coordinates(0);
    mask->set_verbose(1);

    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(mask->get_output_port());
    coords->set_enable_periodic_shift_x(1);

    p_teca_index_executive exec = teca_index_executive::New();

    // run the test or generate the baseline
    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test)
    {
        std::cerr << "running the test ..." << std::endl;

        p_teca_cf_reader cfr = teca_cf_reader::New();
        cfr->set_t_axis_variable("");
        cfr->set_files_regex(out_file);

        exec->set_arrays({mask_name});

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, cfr->get_output_port());
        diff->set_input_connection(1, coords->get_output_port());
        diff->set_executive(exec);
        diff->set_verbose(1);

        diff->update();
    }
    else
    {
        std::cerr << "writing the baseline ..." << std::endl;

        p_teca_cf_writer writer = teca_cf_writer::New();
        writer->set_input_connection(coords->get_output_port());
        writer->set_file_name(out_file);
        writer->set_point_arrays({mask_name});
        writer->set_thread_pool_size(1);
        writer->set_layout_to_number_of_steps();
        writer->set_steps_per_file(1);

        writer->set_executive(exec);

        writer->update();
    }

    return 0;
}
