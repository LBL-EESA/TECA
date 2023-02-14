#include "teca_table.h"
#include "teca_table_join.h"
#include "teca_dataset_source.h"
#include "teca_table_writer.h"
#include "teca_index_executive.h"
#include "teca_dataset_diff.h"

int main(int argc, char **argv)
{
    (void) argc;
    (void) argv;

    // create 3 tables to join
    auto s1 = teca_dataset_source::New();
    auto t1 = teca_table::New();
    t1->declare_columns("col 1", double());
    t1 << 1.0 << 2.0 << 3.0;
    s1->set_dataset(t1);

    auto s2 = teca_dataset_source::New();
    auto t2 = teca_table::New();
    t2->declare_columns("col 2", int());
    t2 << 3 << 2 << 1;
    s2->set_dataset(t2);

    auto s3 = teca_dataset_source::New();
    auto t3 = teca_table::New();
    t3->declare_columns("col 3", float());
    t3 << 3.1415 << -3.1415 << 2.718;
    s3->set_dataset(t3);

    // join the 3 tables
    auto tj = teca_table_join::New();
    tj->set_number_of_input_connections(3);
    tj->set_input_connection(0, s1->get_output_port());
    tj->set_input_connection(1, s2->get_output_port());
    tj->set_input_connection(2, s3->get_output_port());

    auto exec = teca_index_executive::New();
    exec->set_arrays({"col 1", "col 2", "col 3"});

    bool write = false;
    if (write)
    {
        // dump to disk for inspection
        auto tw = teca_table_writer::New();
        tw->set_input_connection(tj->get_output_port());
        tw->set_executive(exec);
        tw->set_file_name("test_table_join.csv");

        tw->update();
    }

    // create the ground truth
    auto sb = teca_dataset_source::New();
    auto tb = teca_table::New();
    tb->declare_columns("col 1", double(),
         "col 2", int(), "col 3", float());

    tb   << 1.0 << 3 <<  3.1415
         << 2.0 << 2 << -3.1415
         << 3.0 << 1 <<  2.718;

    sb->set_dataset(tb);


    // compare the baseline against the joined table
    auto diff = teca_dataset_diff::New();
    diff->set_input_connection(0, sb->get_output_port());
    diff->set_input_connection(1, tj->get_output_port());
    diff->set_executive(exec);
    diff->set_verbose(1);

    diff->update();


    return 0;
}


