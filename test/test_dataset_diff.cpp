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
        md.set("index_initializer_key", std::string("number_of_tables"));
        md.set("index_request_key", std::string("table_id"));
        md.set("number_of_tables", num_tables);
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

void write_test_tables(const string& file_name, long num_tables)
{
    p_teca_programmable_algorithm s = teca_programmable_algorithm::New();
    s->set_number_of_input_connections(0);
    s->set_number_of_output_ports(1);
    s->set_report_callback(report(num_tables));
    s->set_execute_callback(execute_create_test_table());

    p_teca_table_writer w = teca_table_writer::New();
    w->set_input_connection(s->get_output_port());
    w->set_executive(teca_index_executive::New());
    w->set_file_name(file_name);
    w->set_output_format_bin();

    w->update();
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    teca_system_interface::set_stack_trace_on_error();

    // Write 2 test files.
    write_test_tables("dataset_diff_test_%t%.%e%", 2);

    {
      // Set up a pair of readers that read a single file. The resulting
      // datasets should be identical.
      p_teca_table_reader r1 = teca_table_reader::New();
      r1->set_file_name("dataset_diff_test_0.bin");

      p_teca_table_reader r2 = teca_table_reader::New();
      r2->set_file_name("dataset_diff_test_0.bin");

      // Set up the dataset diff algorithm to accept the two readers.
      p_teca_dataset_diff diff = teca_dataset_diff::New();
      diff->set_input_connection(0, r1->get_output_port());
      diff->set_input_connection(1, r2->get_output_port());

      diff->update();
    }

    // TODO -- disabled because the word ERROR in the test output causes the test
    // to fail, but in this case we expect it to fail, and that would actually
    // be success. Our cmake/ctest code does not yet handle that type of test
    // The following is a test of the dataset_diff algorithms ability to
    // correctly identify two different datasets.
    //
    //{
    //    // Set up a pair of readers that read different files. The resulting
    //    // datasets should be different.
    //    p_teca_table_reader r1 = teca_table_reader::New();
    //    r1->set_file_name("dataset_diff_test_0.bin");
    //    p_teca_table_reader r2 = teca_table_reader::New();
    //    r2->set_file_name("dataset_diff_test_1.bin");
    //
    //    // Set up the dataset diff algorithm to accept the two readers.
    //    p_teca_dataset_diff diff = teca_dataset_diff::New();
    //    diff->set_input_connection(0, r1->get_output_port());
    //    diff->set_input_connection(1, r2->get_output_port());
    //
    //    diff->update();
    //}

    return 0;
}
