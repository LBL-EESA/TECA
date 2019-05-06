#include "teca_config.h"
#include "teca_programmable_algorithm.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_table.h"
#include "teca_test_util.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
using namespace std;

struct report
{
    long num_tables;

    report() : num_tables(0) {}
    explicit report(long n) : num_tables(n) {}

    teca_metadata operator()
        (unsigned int, const std::vector<teca_metadata> &)
    {
        teca_metadata md;
        md.insert("index_initializer_key", std::string("number_of_tables"));
        md.insert("index_request_key", std::string("table_id"));
        md.insert("number_of_tables", num_tables);
        return md;
    }
};

// This test should be executed after test_table_writer.cpp, which writes the
// files needed to test the dataset differ.
struct execute_create_test_table
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

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc < 2)
    {
        std::cerr << "Error: test_dataset_diff [compare same]" << std::endl;
        return -1;
    }

    int compare_same = atoi(argv[1]);

    // write 2 test files.
    p_teca_programmable_algorithm s = teca_programmable_algorithm::New();
    s->set_number_of_input_connections(0);
    s->set_number_of_output_ports(1);
    s->set_report_callback(report(2));
    s->set_execute_callback(execute_create_test_table());

    p_teca_table_writer w = teca_table_writer::New();
    w->set_input_connection(s->get_output_port());
    w->set_executive(teca_index_executive::New());
    w->set_file_name("dataset_diff_test_%t%.%e%");
    w->set_output_format_bin();

    w->update();


    // read two file and compare them. if compare_same is true then
    // read the first file on both readers, if not read both files
    p_teca_table_reader r1 = teca_table_reader::New();
    p_teca_table_reader r2 = teca_table_reader::New();

    if (compare_same)
    {
        // these datasets are the same, hence diff should pass
        r1->set_file_name("dataset_diff_test_0.bin");
        r2->set_file_name("dataset_diff_test_0.bin");
    }
    else
    {
        // these datasets are different, hence diff should fail
        r1->set_file_name("dataset_diff_test_0.bin");
        r2->set_file_name("dataset_diff_test_1.bin");
    }

    // diff the output of the readers
    p_teca_dataset_diff diff = teca_dataset_diff::New();
    diff->set_input_connection(0, r1->get_output_port());
    diff->set_input_connection(1, r2->get_output_port());

    diff->update();

    return 0;
}
