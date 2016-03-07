#include "create_test_table.h"
using namespace std;

const_p_teca_dataset create_test_table(long step)
{
    p_teca_table t = teca_table::New();

    t->declare_columns(
        "step", long(), "name", string(),
        "age", int(), "skill", double());

    t->reserve(64);

    t->append(step, string("James"), 0, 0.0);
    t << step << string("Jessie") << 2 << 0.2
      << step << string("Frank") << 3 << 3.1415;
    t->append(step, string("Mike"), 1, 0.1);
    t << step << string("Tom") << 4 << 0.4;

    return t;
}

