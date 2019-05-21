#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_binary_segmentation.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 6)
    {
        cerr << endl << "Usage error:" << endl
            << "test_binary_segmentation [input regex] [var] [low] [high] [output]"
            << endl << endl;
        return -1;
    }

    std::string in_files = argv[1];
    std::string var = argv[2];
    double low = atof(argv[3]);
    double high = atof(argv[4]);
    std::string out_file = argv[5];

    // build the pipeline
    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(in_files);

    p_teca_binary_segmentation seg = teca_binary_segmentation::New();
    seg->set_threshold_variable(var);
    seg->set_low_threshold_value(low);
    seg->set_high_threshold_value(high);
    seg->set_threshold_by_percentile();
    seg->set_input_connection(cfr->get_output_port());

    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(seg->get_output_port());
    wri->set_file_name(out_file);

    wri->update();

    return 0;
}
