#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_temporal_average.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_ar_detect.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_reduce.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"

#include <sys/time.h>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // parse command line
    string vapor_files;
    string mask_file;
    string baseline;
    int have_baseline = 0;
    long first_step = 0;
    long last_step = -1;
    unsigned int n_threads = 1;
    string vapor_var;
    string mask_var;

    if (argc != 9)
    {
        cerr << endl << "Usage error:" << endl
            << "test_ar_detect [input regex] [land mask file] [output] "
               "[first step] [last step] [n threads] [water vapor var] "
               "[mask var]" << endl << endl;
        return -1;
    }

    // parse command line
    vapor_files = argv[1];
    mask_file = argv[2];
    baseline = argv[3];
    if (teca_file_util::file_exists(baseline.c_str()))
        have_baseline = 1;
    first_step = atoi(argv[4]);
    last_step = atoi(argv[5]);
    n_threads = atoi(argv[6]);
    vapor_var = argv[7];
    mask_var = argv[8];

    // build the pipeline
    p_teca_cf_reader vapor_reader = teca_cf_reader::New();
    vapor_reader->set_files_regex(vapor_files);

    p_teca_normalize_coordinates vapor_coords = teca_normalize_coordinates::New();
    vapor_coords->set_input_connection(vapor_reader->get_output_port());

    p_teca_cf_reader mask_reader = teca_cf_reader::New();
    mask_reader->set_t_axis_variable("");
    mask_reader->append_file_name(mask_file);

    p_teca_normalize_coordinates mask_coords = teca_normalize_coordinates::New();
    mask_coords->set_input_connection(mask_reader->get_output_port());

    p_teca_cartesian_mesh_regrid mask_regrid = teca_cartesian_mesh_regrid::New();
    mask_regrid->set_input_connection(0, vapor_coords->get_output_port());
    mask_regrid->set_input_connection(1, mask_coords->get_output_port());
    mask_regrid->append_array(mask_var);

    p_teca_ar_detect ar_detect = teca_ar_detect::New();
    ar_detect->set_input_connection(mask_regrid->get_output_port());
    ar_detect->set_water_vapor_variable(vapor_var);
    ar_detect->set_land_sea_mask_variable(mask_var);

    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->set_input_connection(ar_detect->get_output_port());
    map_reduce->set_start_index(first_step);
    map_reduce->set_end_index(last_step);
    map_reduce->set_verbose(1);
    map_reduce->set_thread_pool_size(n_threads);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("time_step");

    p_teca_table_calendar cal = teca_table_calendar::New();
    cal->set_input_connection(sort->get_output_port());

    // regression test
    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && have_baseline)
    {
        // run the test
        p_teca_table_reader table_reader = teca_table_reader::New();
        table_reader->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, table_reader->get_output_port());
        diff->set_input_connection(1, cal->get_output_port());
        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            cerr << "generating baseline image " << baseline << endl;
        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(cal->get_output_port());
        table_writer->set_file_name(baseline.c_str());
        table_writer->update();
    }

    return 0;
}
