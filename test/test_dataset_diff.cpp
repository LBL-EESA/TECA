#include "teca_config.h"
#include "teca_programmable_algorithm.h"
#include "teca_table_reader.h"
#include "teca_dataset_diff.h"
#include "teca_table.h"
#include "teca_time_step_executive.h"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
using namespace std;

// This test should be executed after test_table_writer.cpp, which writes the 
// files needed to test the dataset differ.

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    {
      // Set up a pair of readers that read a single file. The resulting
      // datasets should be identical.
      p_teca_table_reader r1 = teca_table_reader::New();
      r1->set_file_name("table_writer_test_0.bin");
      p_teca_table_reader r2 = teca_table_reader::New();
      r2->set_file_name("table_writer_test_0.bin");

      // Set up the dataset diff algorithm to accept the two readers.
      p_teca_dataset_diff diff = teca_dataset_diff::New();
      diff->set_input_connection(0, r1->get_output_port());
      diff->set_input_connection(1, r2->get_output_port());

      diff->update();
    }

    {
      // Set up a pair of readers that read different files. The resulting
      // datasets should be different.
      p_teca_table_reader r1 = teca_table_reader::New();
      r1->set_file_name("table_writer_test_0.bin");
      p_teca_table_reader r2 = teca_table_reader::New();
      r2->set_file_name("table_writer_test_1.bin");

      // Set up the dataset diff algorithm to accept the two readers.
      p_teca_dataset_diff diff = teca_dataset_diff::New();
      diff->set_input_connection(0, r1->get_output_port());
      diff->set_input_connection(1, r2->get_output_port());

      diff->update();
    }

    return 0;
}


