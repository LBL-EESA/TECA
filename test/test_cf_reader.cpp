#include "teca_cf_reader.h"
#include "cf_reader_driver.h"

#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Usage error: set files regex on command line" << endl;
        return -1;
    }

    p_teca_cf_reader r = teca_cf_reader::New();
    r->set_files_regex(argv[1]);

    p_cf_reader_driver d = cf_reader_driver::New();
    d->set_input_connection(r->get_output_port());

    d->update();

    return 0;
}
