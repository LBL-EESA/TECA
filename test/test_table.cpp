#include "teca_table.h"

#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    p_teca_table t = teca_table::New();

    t->declare_columns("c1", int(), "c2", float(), "c3", double());
    t->reserve(64);

    t->append(0, 0, 0);
    t->append(1, 1, 1);

    t << 2 << 2 << 2;

    t->to_stream(cerr);
    cerr << endl;

    return 0;
}
