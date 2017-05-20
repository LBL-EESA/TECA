#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_remove_rows.h"
#include "teca_table_sort.h"
#include "teca_cf_reader.h"
#include "teca_tc_wind_radii.h"
#include "teca_table_reduce.h"
#include "teca_table_to_stream.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_time_step_executive.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"
#include "teca_mpi_manager.h"

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

    // parse command line
    if (argc != 10)
    {
        if (rank == 0)
        {
            cerr << endl << "Usage error:" << endl
                << "test_tc_wind_radii [storm table] [mesh data regex] [test baseline] "
                   "[mask expression] [num bins] [profile type] [num threads] [first step] "
                   "[last step]"
                << endl << endl;
        }
        return -1;
    }

    std::string storm_table = argv[1];
    std::string mesh_data_regex = argv[2];
    std::string baseline_table = argv[3];
    std::string mask_expression = argv[4];
    int n_bins = atoi(argv[5]);
    int profile_type = atoi(argv[6]);
    int n_threads = atoi(argv[7]);
    int first_step =  atoi(argv[8]);
    int last_step = atoi(argv[9]);

    // create the pipeline
    p_teca_table_reader storm_reader = teca_table_reader::New();
    storm_reader->set_file_name(storm_table);

    p_teca_table_remove_rows eval_expr = teca_table_remove_rows::New();
    eval_expr->set_input_connection(storm_reader->get_output_port());
    eval_expr->set_mask_expression(mask_expression);

    p_teca_cf_reader mesh_data_reader = teca_cf_reader::New();
    mesh_data_reader->set_files_regex(mesh_data_regex);

    p_teca_tc_wind_radii wind_radii = teca_tc_wind_radii::New();
    wind_radii->set_input_connection(0, eval_expr->get_output_port());
    wind_radii->set_input_connection(1, mesh_data_reader->get_output_port());
    wind_radii->set_number_of_radial_bins(n_bins);
    wind_radii->set_profile_type(profile_type);

    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->set_input_connection(wind_radii->get_output_port());
    map_reduce->set_first_step(first_step);
    map_reduce->set_last_step(last_step);
    map_reduce->set_verbose(1);
    map_reduce->set_thread_pool_size(n_threads);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("track_id");
    sort->enable_stable_sort();

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
