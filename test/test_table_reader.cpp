#include "teca_config.h"
#include "teca_programmable_algorithm.h"
#include "teca_table_reader.h"
#include "teca_table.h"
#include "teca_time_step_executive.h"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
using namespace std;

// This test should be paired with test_table_writer.cpp, which writes the 
// file that this reader reads.

struct execute
{
    const_p_teca_dataset operator()
        (unsigned int, const vector<const_p_teca_dataset> & tables,
         const teca_metadata &req)
    {
        (void)req;

        // The input should be a single teca table.
        if (tables.size() != 1)
        {
            cerr << "FAIL: input should be a single TECA table." << endl;
            return nullptr;
        }

        // Inspect the table.
        const_p_teca_table t = dynamic_pointer_cast<const teca_table>(tables[0]);
        if (t == nullptr)
        {
            cerr << "FAIL: input is not a table." << endl;
            return nullptr;
        }
        if (t->empty())
        {
            cerr << "FAIL: input is an empty table." << endl;
            return nullptr;
        }

        const_p_teca_variant_array step = t->get_column("step");
        if (step == nullptr)
        {
            cerr << "FAIL: input table has no 'step' column." << endl;
            return nullptr;
        }
        const_p_teca_variant_array name = t->get_column("name");
        if (name == nullptr)
        {
            cerr << "FAIL: input table has no 'name' column." << endl;
            return nullptr;
        }
        const_p_teca_variant_array age = t->get_column("age");
        if (age == nullptr)
        {
            cerr << "FAIL: input table has no 'age' column." << endl;
            return nullptr;
        }
        const_p_teca_variant_array skill = t->get_column("skill");
        if (skill == nullptr)
        {
            cerr << "FAIL: input table has no 'skill' column." << endl;
            return nullptr;
        }

        // Check the data.
        int num_rows = 5;
        int steps[] = {0, 0, 0, 0, 0};
        const char* names[] = {"James", "Jessie", "Frank", "Mike", "Tom"};
        int ages[] = {0, 2, 3, 1, 4};
        double skills[] = {0.0, 0.2, 3.1415, 0.1, 0.4};
        double fuzz = 1e-12;
        for (int row = 0; row < num_rows; ++row)
        {
          // Step number.
          int s;
          step->get(row, s);
          if (s != steps[row])
          {
              cerr << "FAIL: input table has incorrect step number " << s << " in row " << row << " (should be " << steps[row] << ")." << endl;
              return nullptr;
          }

          // Name.
          string n;
          name->get(row, n);
          if (n != names[row])
          {
              cerr << "FAIL: input table has incorrect name " << n << " in row " << row << " (should be " << names[row] << ")." << endl;
              return nullptr;
          }

          // Age.
          int a;
          age->get(row, a);
          if (a != ages[row])
          {
              cerr << "FAIL: input table has incorrect age " << a << " in row " << row << " (should be " << ages[row] << ")." << endl;
              return nullptr;
          }

          // Skill.
          double sk;
          skill->get(row, sk);
          if (abs(sk - skills[row]) > fuzz)
          {
              cerr << "FAIL: input table has incorrect skill " << sk << " in row " << row << " (should be " << skills[row] << ")." << endl;
              return nullptr;
          }
        }

        return nullptr;
    }
};

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    p_teca_table_reader r = teca_table_reader::New();
    r->set_file_name("table_writer_test_0.bin");

    p_teca_programmable_algorithm s = teca_programmable_algorithm::New();
    s->set_number_of_input_connections(1);
    s->set_number_of_output_ports(1);
    s->set_input_connection(r->get_output_port());
    s->set_execute_callback(execute());

    s->update();

    return 0;
}


