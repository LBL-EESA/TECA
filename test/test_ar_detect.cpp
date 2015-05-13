#include "teca_cf_reader.h"
#include "teca_temporal_average.h"
#include "teca_ar_detect.h"
#include "teca_table_reduce.h"
#include "teca_table_writer.h"
#include "teca_time_step_executive.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

// example use
// ./test/test_cf_reader ~/work/teca/data/'cam5_1_amip_run2.cam2.h2.1991-10-.*' tmp 0 -1 PS

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << endl << "Usage error:" << endl
            << "test_cf_reader [input regex] [output base] [filter width] [first step] [last step]" << endl
            << endl;
        return -1;
    }

    // parse command line
    string regex = argv[1];
    string base = argv[2];
    int filter_width = 0;
    if (argc > 3)
        filter_width = atoi(argv[3]);
    long first_step = 0;
    if (argc > 4)
        first_step = atoi(argv[4]);
    long last_step = -1;
    if (argc > 5)
        last_step = atoi(argv[5]);

    // cf reader
    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(regex);

    // ar detect
    p_teca_ar_detect ard = teca_ar_detect::New();

    // temporal average
    if (filter_width)
    {
        p_teca_temporal_average avg = teca_temporal_average::New();
        avg->set_filter_type(teca_temporal_average::backward);
        avg->set_filter_width(filter_width);
        avg->set_input_connection(cfr->get_output_port());
        ard->set_input_connection(avg->get_output_port());
    }
    else
    {
        ard->set_input_connection(cfr->get_output_port());
    }

    // reduction
    p_teca_table_reduce tr = teca_table_reduce::New();
    tr->set_input_connection(ard->get_output_port());
    tr->set_first_step(first_step);
    tr->set_last_step(last_step);
    tr->set_thread_pool_size(1);

    // writer
    p_teca_table_writer twr = teca_table_writer::New();
    twr->set_input_connection(tr->get_output_port());
    twr->set_base_file_name(base);

    // run the pipeline
    twr->update();

    return 0;
}
