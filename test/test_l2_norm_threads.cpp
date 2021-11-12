#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_l2_norm.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"
#include "teca_index_executive.h"
#include "teca_cf_writer.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 7)
    {
        std::cerr << std::endl << "Usage error:" << std::endl
            << "test_l2_norm [input regex] [comp 0] [comp 1] [comp 2] [l2 norm] [output]"
            << std::endl << std::endl;
        return -1;
    }

    std::string in_files = argv[1];
    std::string comp_0_var = argv[2];
    std::string comp_1_var = argv[3];
    std::string comp_2_var = argv[4];
    std::string norm_var = argv[5];
    std::string out_file = argv[6];

    std::vector<std::string> out_vars({comp_0_var, norm_var});

    // cmake can't pass "" so "." is used to indicate lack of a component
    if (comp_1_var == ".")
        comp_1_var = "";
    else
        out_vars.push_back(comp_1_var);

    if (comp_2_var == ".")
        comp_2_var = "";
    else
        out_vars.push_back(comp_2_var);

    // build the pipeline
    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(in_files);

    p_teca_l2_norm l2 = teca_l2_norm::New();
    l2->set_input_connection(cfr->get_output_port());
    l2->set_component_0_variable(comp_0_var);
    l2->set_component_1_variable(comp_1_var);
    l2->set_component_2_variable(comp_2_var);
    l2->set_l2_norm_variable(norm_var);

    p_teca_index_executive exec = teca_index_executive::New();


    p_teca_cf_writer wri = teca_cf_writer::New();
    wri->set_input_connection(l2->get_output_port());
    wri->set_thread_pool_size(1);
    wri->set_file_name(out_file);
    wri->set_layout_to_yearly();
    wri->set_executive(exec);
    wri->set_last_step(0);
    wri->set_point_arrays(out_vars);

    wri->update();

/*
    p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
    wri->set_input_connection(seg->get_output_port());
    wri->set_file_name(out_file);
    wri->set_executive(exec);
    wri->update();
*/

    return 0;
}
