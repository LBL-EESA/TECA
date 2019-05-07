#include "teca_config.h"
#include "teca_programmable_algorithm.h"
#include "teca_table_writer.h"
#include "teca_table_reader.h"
#include "teca_table.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_test_util.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
using namespace std;

struct report
{
    teca_metadata operator()
        (unsigned int, const std::vector<teca_metadata> &)
    {
        teca_metadata md;
        md.insert("index_initializer_key", std::string("number_of_tables"));
        md.insert("index_request_key", std::string("table_id"));
        md.insert("number_of_tables", long(1));
        return md;
    }
};

struct execute
{
    const_p_teca_dataset operator()
        (unsigned int, const std::vector<const_p_teca_dataset> &,
        const teca_metadata &req)
    {
        long table_id = 0;
        if (req.get("table_id", table_id))
        {
            TECA_ERROR("request is missing \"table_id\"")
            return nullptr;
        }

        return teca_test_util::create_test_table(table_id);
    }
};

int main(int, char **)
{
    teca_system_interface::set_stack_trace_on_error();

    // Write a test table.
    p_teca_programmable_algorithm s = teca_programmable_algorithm::New();
    s->set_number_of_input_connections(0);
    s->set_number_of_output_ports(1);
    s->set_report_callback(report());
    s->set_execute_callback(execute());

    p_teca_table_writer w = teca_table_writer::New();
    w->set_input_connection(s->get_output_port());
    w->set_file_name("table_reader_test.bin");
    w->set_executive(teca_index_executive::New());

    w->update();

    // Set up reader to read it back in
    p_teca_table_reader r = teca_table_reader::New();
    r->set_file_name("table_reader_test.bin");

    // create the same table in memory
    s = teca_programmable_algorithm::New();
    s->set_number_of_input_connections(0);
    s->set_number_of_output_ports(1);
    s->set_report_callback(report());
    s->set_execute_callback(execute());

    // Set up the dataset diff algorithm
    p_teca_dataset_diff diff = teca_dataset_diff::New();
    diff->set_input_connection(0, s->get_output_port());
    diff->set_input_connection(1, r->get_output_port());
    diff->set_executive(teca_index_executive::New());

    // run the test
    diff->update();

    return 0;
}
