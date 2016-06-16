#include "array_source.h"
#include "array_time_average.h"
#include "array_writer.h"
#include "array_executive.h"
#include "teca_system_interface.h"

#include <iostream>
using namespace std;

int main(int, char **)
{
    teca_system_interface::set_stack_trace_on_error();

    // create a pipeline
    cerr << "creating the pipeline..." << endl
      << endl
      << "  1 arrays" << endl
      << "  4 timesteps" << endl
      << "  array len 5         3 steps" << endl
      << "      |                  |" << endl
      << "array_source --> array_time_average --> array_writer" << endl
      << endl;

    p_array_source src = array_source::New();
    src->set_number_of_timesteps(4);
    src->set_number_of_arrays(1);
    src->set_array_size(5);

    p_array_time_average avg = array_time_average::New();
    avg->set_filter_width(3);

    p_array_writer wri = array_writer::New();

    avg->set_input_connection(src->get_output_port());
    wri->set_input_connection(avg->get_output_port());

    // execute
    cerr << "execute..." << endl;
    p_array_executive exec = array_executive::New();
    wri->set_executive(exec);
    wri->update();
    cerr << endl;

    return 0;
}
