#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"
#include "teca_mpi_manager.h"
#include "teca_bayesian_ar_detect.h"
#include "teca_bayesian_ar_detect_parameters.h"
#include "teca_binary_segmentation.h"
#include "teca_connected_components.cxx"
#include "teca_2d_component_area.h"
#include "teca_component_statistics.h"
#include "teca_table_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_to_stream.h"
#include "teca_cartesian_mesh_writer.h"

#include <vector>
#include <string>
#include <iostream>

using std::cerr;
using std::endl;

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // parse command line
    if (argc != 8)
    {
        if (rank == 0)
        {
            cerr << endl << "Usage error:" << endl
                << "test_bayesian_ar_detect [mesh data regex] "
                   "[baseline table] [water vapor var] [out file name] [num threads] "
                   "[first step] [last step]" << endl << endl;
        }
        return -1;
    }

    std::string mesh_data_regex = argv[1];
    std::string baseline_table = argv[2];
    std::string water_vapor_var = argv[3];
    std::string out_file_name = argv[4];
    int n_threads = atoi(argv[5]);
    int first_step =  atoi(argv[6]);
    int last_step = atoi(argv[7]);

    // create the pipeline
    p_teca_bayesian_ar_detect_parameters parameter_table =
        teca_bayesian_ar_detect_parameters::New();

    p_teca_cf_reader mesh_data_reader = teca_cf_reader::New();
    mesh_data_reader->set_files_regex(mesh_data_regex);
    //mesh_data_reader->set_periodic_in_x(1);

    p_teca_bayesian_ar_detect ar_detect = teca_bayesian_ar_detect::New();
    ar_detect->set_input_connection(0, parameter_table->get_output_port());
    ar_detect->set_input_connection(1, mesh_data_reader->get_output_port());
    ar_detect->set_water_vapor_variable(water_vapor_var);
    ar_detect->set_thread_pool_size(n_threads);

    p_teca_binary_segmentation seg = teca_binary_segmentation::New();
    seg->set_input_connection(ar_detect->get_output_port());
    seg->set_low_threshold_value(0.25);
    seg->set_threshold_variable("ar_probability");
    seg->set_segmentation_variable("ar_probability_0.25");

    p_teca_connected_components cc = teca_connected_components::New();
    cc->set_input_connection(seg->get_output_port());
    cc->set_segmentation_variable("ar_probability_0.25");
    cc->set_component_variable("ars");

    p_teca_2d_component_area ca = teca_2d_component_area::New();
    ca->set_input_connection(cc->get_output_port());
    ca->set_component_variable("ars");

#undef WRITE_RESULT
#if defined WRITE_RESULT
    std::cerr << "writing data to disk for manual verification" << std::endl;
    p_teca_cf_writer wri = teca_cf_writer::New();
    wri->set_input_connection(ca->get_output_port());
    wri->set_file_name(out_file_name);
    wri->set_thread_pool_size(1);
    wri->set_point_arrays({"ar_probability", "ar_probability_0.25", "ars"});
    wri->update();
    exit(0);
#endif

    p_teca_component_statistics cs = teca_component_statistics::New();
    cs->set_input_connection(ca->get_output_port());

    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->set_input_connection(cs->get_output_port());
    map_reduce->set_start_index(first_step);
    map_reduce->set_end_index(last_step);
    map_reduce->set_verbose(1);
    map_reduce->set_thread_pool_size(1);

    // sort results in time
    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("global_component_ids");

    if (teca_file_util::file_exists(baseline_table.c_str()))
    {
        // run the test
        p_teca_table_reader baseline_table_reader = teca_table_reader::New();
        baseline_table_reader->set_file_name(baseline_table);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_table_reader->get_output_port());
        diff->set_input_connection(1, sort->get_output_port());

        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            cerr << "generating baseline image " << baseline_table << endl;

        p_teca_table_to_stream tts = teca_table_to_stream::New();
        tts->set_input_connection(sort->get_output_port());

        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(tts->get_output_port());
        table_writer->set_file_name(baseline_table);

        table_writer->update();
        return -1;
    }

    return 0;
}
