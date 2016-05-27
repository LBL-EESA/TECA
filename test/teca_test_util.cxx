#include "teca_test_util.h"
#include "teca_common.h"

namespace teca_test_util
{

// **************************************************************************
const_p_teca_table create_test_table(long step, int tid)
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
                t->get_column("name"))->set(2, "Dave");
            return t;
            break;

        case teca_test_util::break_int_col:
            std::dynamic_pointer_cast<teca_variant_array_impl<std::string>>(
                t->get_column("age"))->set(2, 6);
            return t;
            break;

        case teca_test_util::break_float_col:
            std::dynamic_pointer_cast<teca_variant_array_impl<std::string>>(
                t->get_column("skill"))->set(2, 2.71828183);
            return t;
    }

    TECA_ERROR("bad table id " << tid)
    return nullptr;
}

// **************************************************************************
int bcast(std::string &str)
{
#if defined(TECA_HAS_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    long str_size = str.size();
    if (teca_test_util::bcast(str_size))
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
    if (teca_test_util::bcast(buf, str_size))
        return -1;
    if (rank != 0)
    {
        str = buf;
        free(buf);
    }
#else
    (void)str;
#endif
    return 0;
}
};
