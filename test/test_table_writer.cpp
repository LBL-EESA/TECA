#include "teca_table.h"
#include "teca_metadata.h"
#include "teca_table_writer.h"
#include "teca_test_util.h"
#include "teca_system_interface.h"

#include <iostream>
using namespace std;

int main(int, char **)
{
    teca_system_interface::set_stack_trace_on_error();

    p_teca_algorithm s = teca_test_util::test_table_server::New(4);

    // Write some .csv files.
    {
      p_teca_table_writer w = teca_table_writer::New();
      w->set_input_connection(s->get_output_port());
      w->set_file_name("table_writer_test_%t%.csv");

      w->update();
    }

    // Write some binary files.
    {
      p_teca_table_writer w = teca_table_writer::New();
      w->set_input_connection(s->get_output_port());
      w->set_file_name("table_writer_test_%t%.bin");

      w->update();
    }

    return 0;
}
