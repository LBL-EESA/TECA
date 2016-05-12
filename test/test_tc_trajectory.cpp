#include "teca_config.h"
#include "teca_tc_trajectory.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
#include "teca_test_util.h"
#include "teca_time_step_executive.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int mpi_thread_required = MPI_THREAD_SERIALIZED;
    int mpi_thread_provided = 0;
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);
    if (mpi_thread_provided < mpi_thread_required)
    {
        cerr << "ERROR: MPI does not support threads" << endl;
        return -1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    // parse command line
    string candidates;
    string baseline;
    int have_baseline = 0;
    int write_binary = 0;
    double max_daily_distance;
    double min_wind_speed;
    double min_wind_duration;

    if (rank == 0)
    {
        if (argc != 6)
        {
            cerr << endl << "Usage error:" << endl
                << "test_tc_trajectory [candidate table] [baseline] "
                   "[max daily dist] [min wind speed] [min wind duration]"
                << endl << endl;
            return -1;
        }

        // parse command line
        candidates = argv[1];
        baseline = argv[2];
        if (teca_file_util::file_exists(baseline.c_str()))
            have_baseline = 1;
        write_binary = teca_file_util::extension(baseline) == ".bin";
        max_daily_distance = atof(argv[3]);
        min_wind_speed = atof(argv[4]);
        min_wind_duration = atof(argv[5]);
    }

    teca_test_util::bcast(candidates);
    teca_test_util::bcast(baseline);
    teca_test_util::bcast(have_baseline);
    teca_test_util::bcast(write_binary);
    teca_test_util::bcast(max_daily_distance);
    teca_test_util::bcast(min_wind_speed);
    teca_test_util::bcast(min_wind_duration);

    p_teca_table_reader cand = teca_table_reader::New();
    cand->set_file_name(candidates);

    p_teca_tc_trajectory tracks = teca_tc_trajectory::New();
    tracks->set_input_connection(cand->get_output_port());

    p_teca_table_calendar cal = teca_table_calendar::New();
    cal->set_input_connection(tracks->get_output_port());

    // regression test
    if (have_baseline)
    {
        // run the test
        p_teca_table_reader base = teca_table_reader::New();
        base->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, base->get_output_port());
        diff->set_input_connection(1, cal->get_output_port());

        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            cerr << "generating baseline image " << baseline << endl;
        p_teca_table_writer base = teca_table_writer::New();
        base->set_input_connection(cal->get_output_port());
        base->set_file_name(baseline.c_str());
        if (write_binary)
            base->set_output_format_bin();
        else
            base->set_output_format_csv();

        base->update();
    }

#if defined(TECA_HAS_MPI)
    MPI_Finalize();
#endif
    return 0;
}
