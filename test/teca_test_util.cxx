#include "teca_test_util.h"
#include "teca_common.h"
#include "teca_programmable_algorithm.h"
#include "teca_metadata.h"
#include "teca_metadata_util.h"

#include <string>
#include <vector>


namespace teca_test_util
{
// This creates a TECA table containing some basic test data that
// is used by the TECA table reader/writer tests and the dataset_diff
// test.
enum {base_table,
    break_string_col,
    break_int_col,
    break_float_col
    };
p_teca_table create_test_table(long step,
    int tid=teca_test_util::base_table);

// **************************************************************************
p_teca_table create_test_table(long step, int tid)
{
    p_teca_table t = teca_table::New();

    t->declare_columns(
        "step", long(), "name", std::string(),
        "age", int(), "skill", double());

    t->reserve(64);

    t->append(step, std::string("James"), 0, 0.0);
    t << step << std::string("Jessie") << 2 << 0.2
      << step << std::string("Frank") << 3 << 3.1415;
    t->append(step, std::string("Mike"), 1, 0.1);
    t << step << std::string("Tom") << 4 << 0.4;


    switch (tid)
    {
        case teca_test_util::base_table:
            return t;
            break;

        case teca_test_util::break_string_col:
            std::dynamic_pointer_cast<teca_variant_array_impl<std::string>>(
                t->get_column("name"))->set(2, std::string("Dave"));
            return t;
            break;

        case teca_test_util::break_int_col:
            std::dynamic_pointer_cast<teca_variant_array_impl<int>>(
                t->get_column("age"))->set(2, 6);
            return t;
            break;

        case teca_test_util::break_float_col:
            std::dynamic_pointer_cast<teca_variant_array_impl<double>>(
                t->get_column("skill"))->set(2, 2.71828183);
            return t;
    }

    TECA_ERROR("bad table id " << tid)
    return nullptr;
}

struct report_test_tables
{
    report_test_tables(long num = 4) : num_test_tables(num) {}

    teca_metadata operator()
        (unsigned int, const std::vector<teca_metadata> &)
    {
        teca_metadata md;
        md.set("index_initializer_key", std::string("number_of_tables"));
        md.set("index_request_key", std::string("table_id"));
        md.set("number_of_tables", this->num_test_tables);
        return md;
    }

    long num_test_tables;
};

struct generate_test_tables
{
    const_p_teca_dataset operator()
        (unsigned int, const std::vector<const_p_teca_dataset> &,
        const teca_metadata &req)
    {
        long table_id = 0;
        std::string request_key;
        if (teca_metadata_util::get_requested_index(req, request_key, table_id))
        {
            TECA_FATAL_ERROR("Failed to determine the requested index")
            return nullptr;
        }

        p_teca_dataset ods = teca_test_util::create_test_table(table_id);
        ods->set_request_index("table_id", table_id);

        return ods;
    }
};

// --------------------------------------------------------------------------
p_teca_algorithm test_table_server::New(long num_tables)
{
    p_teca_programmable_algorithm s = teca_programmable_algorithm::New();
    s->set_name("test_table_server");
    s->set_number_of_input_connections(0);
    s->set_number_of_output_ports(1);
    s->set_report_callback(report_test_tables(num_tables));
    s->set_execute_callback(generate_test_tables());
    return s;
}

// **************************************************************************
int bcast(MPI_Comm comm, std::string &str)
{
#if defined(TECA_HAS_MPI)
    int rank;
    MPI_Comm_rank(comm, &rank);
    long str_size = str.size();
    if (teca_test_util::bcast(comm, str_size))
        return -1;
    char *buf;
    if (rank == 0)
    {
        buf = const_cast<char*>(str.c_str());
    }
    else
    {
        buf = static_cast<char*>(malloc(str_size+1));
        buf[str_size] = '\0';
    }
    if (teca_test_util::bcast(comm, buf, str_size))
        return -1;
    if (rank != 0)
    {
        str = buf;
        free(buf);
    }
#else
    (void)comm;
    (void)str;
#endif
    return 0;
}
};
