#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_dataset_diff.h"
#include "teca_descriptive_statistics.h"
#include "teca_table.h"
#include "teca_table_to_stream.h"
#include "teca_file_util.h"
#include "teca_table_reader.h"
#include "teca_programmable_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_writer.h"
#include "teca_test_util.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"
#include "teca_mpi.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;


struct reduce_callback
{
    // reduction operator
    p_teca_dataset operator()(int device_id,
        const const_p_teca_dataset &in_data_0,
        const const_p_teca_dataset &in_data_1)
    {
        (void) device_id;

        const_p_teca_table table_0 =
            std::dynamic_pointer_cast<const teca_table>(in_data_0);

        const_p_teca_table table_1 =
            std::dynamic_pointer_cast<const teca_table>(in_data_1);

        p_teca_table table_2 = nullptr;

        if (table_0 && table_1)
        {
            table_2 = std::dynamic_pointer_cast
                <teca_table>(table_0->new_copy());

            table_2->concatenate_rows(table_1);
        }
        else if (table_0)
        {
            table_2 = std::dynamic_pointer_cast
                <teca_table>(table_0->new_copy());
        }
        else if (table_1)
        {
            table_2 = std::dynamic_pointer_cast
                <teca_table>(table_1->new_copy());
        }

        return table_2;
    }
};


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();
    int nranks = mpi_man.get_comm_size();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    if (argc < 3)
    {
        cerr << endl << "Usage error:" << endl
            << "test_map_descriptive_statistics [number of files] [input regex] "
               "[test baseline] [first step = 0] [last step = -1] [num threads = 1] "
               "[array 0 =] ... [array n =]"
            << endl << endl;
        return -1;
    }

    // handle a list of files on the command line.
    // the first argument passed is the number of files
    // or 0 if a regex is to be used. this case was added
    // to test processing datasets that do not contain a
    // time axis, but have one stored externally.
    int i = 1;
    int files_num = atoi(argv[i++]);

    // get the list of files (if files_num != 0)
    vector<string> files;
    vector<double> time_values;
    for (; i <= (files_num + 1); ++i)
        files.push_back(argv[i]);

    // generate some time values for the passed files
    for (; i <= (2*files_num + 1); ++i)
        time_values.push_back(atof(argv[i]));

    // files_num == 0 so a regex should have been given
    if (files.empty())
        files.push_back(argv[i++]);

    string baseline = argv[i++];

    int j = i;

    int have_baseline = 0;
    if ((rank == 0) && teca_file_util::file_exists(baseline.c_str()))
        have_baseline = 1;
    teca_test_util::bcast(MPI_COMM_WORLD, have_baseline);
    long first_step = 0;
    if (argc > j)
        first_step = atoi(argv[i++]);
    long last_step = -1;
    if (argc > j + 1)
        last_step = atoi(argv[i++]);
    unsigned int n_threads = 1;
    if (argc > j + 2)
        n_threads = atoi(argv[i++]);
    vector<string> arrays;
    for (int i = j+3; i < argc; ++i)
        arrays.push_back(argv[i]);

    if (rank == 0)
        cerr << "Testing with " << nranks << " MPI ranks each with "
            << n_threads << " threads" << endl;

    // create the pipeline
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    if (files_num)
    {
        // we pass the time axis with a list of files
        cf_reader->set_t_axis_variable("");
        cf_reader->set_t_values(time_values);
        cf_reader->set_calendar("noleap");
        cf_reader->set_t_units("days since 1979-01-01");
        cf_reader->set_file_names(files);
    }
    else
    {
        // time axis is obtained from CF2 compliant dataset
        cf_reader->set_files_regex(files[0]);
    }

    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(cf_reader->get_output_port());

    p_teca_descriptive_statistics stats = teca_descriptive_statistics::New();
    stats->set_input_connection(coords->get_output_port());
    stats->set_dependent_variables(arrays);

    p_teca_programmable_reduce reduce = teca_programmable_reduce::New();
    reduce->set_name("table_reduce");
    reduce->set_input_connection(stats->get_output_port());
    reduce->set_start_index(first_step);
    reduce->set_end_index(last_step);
    reduce->set_thread_pool_size(n_threads);
    reduce->set_reduce_callback(reduce_callback());

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(reduce->get_output_port());
    sort->set_index_column("step");

    p_teca_table_calendar cal = teca_table_calendar::New();
    cal->set_input_connection(sort->get_output_port());

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

        p_teca_table_to_stream tts = teca_table_to_stream::New();
        tts->set_input_connection(cal->get_output_port());

        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(tts->get_output_port());
        table_writer->set_file_name(baseline.c_str());

        table_writer->update();
    }

    return 0;
}
