#include "array_source.h"
#include "array_scalar_multiply.h"
#include "array_add.h"
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
    << "  5 arrays" << endl
    << "  2 timesteps            array_4" << endl
    << "  array len 5            array_1" << endl
    << "    |                        |" << endl
    << "array_source ----------> array_add --> array_writer" << endl
    << "  \\                            /" << endl
    << "   --> array_scalar_multiply --" << endl
    << "               |" << endl
    << "              10" << endl
    << endl;

    p_array_source src = array_source::New();
    src->set_number_of_timesteps(2);
    src->set_number_of_arrays(5);
    src->set_array_size(5);

    p_array_scalar_multiply mul = array_scalar_multiply::New();
    mul->set_scalar(10.0);

    p_array_add add = array_add::New();
    add->set_array_1("array_4");
    add->set_array_2("array_1");

    p_array_writer wri = array_writer::New();

    mul->set_input_connection(src->get_output_port());
    add->set_input_connection(0, src->get_output_port());
    add->set_input_connection(1, mul->get_output_port());
    wri->set_input_connection(add->get_output_port());

    // execute
    cerr << "execute..." << endl;
    p_array_executive exec = array_executive::New();
    wri->set_executive(exec);
    wri->update();
    cerr << endl;

    // TODO comapre output and return pass fail code

    return 0;
}
