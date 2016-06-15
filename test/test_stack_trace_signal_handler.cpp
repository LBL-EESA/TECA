#include "teca_system_interface.h"
#include "teca_common.h"

#include <iostream>
using std::cerr;
using std::endl;

struct dummy
{ int i; };

dummy *do_test(int frame, int segv)
{
    if (frame)
        return do_test(frame-1, segv);

    TECA_STATUS("The stack frame is currently" << std::endl
        << teca_system_interface::get_program_stack(0,0))

    // test the signal handler
    dummy *p = nullptr;
    if (segv)
    {
        TECA_STATUS("Initiating SEGV...")
        p->i = 0;
    }

    return p;
}

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 3)
    {
        cerr << "Usage:" << endl
            << "test_stack_trace_signal_handler [stack depth] [test segv]" << endl
            << endl;
        return -1;
    }

    int stack_depth = atoi(argv[1]);
    int test_segv = atoi(argv[2]);

    do_test(stack_depth, test_segv);

    return 0;
}
