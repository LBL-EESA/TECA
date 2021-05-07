#include "teca_cf_reader.h"
#include "teca_calendar_util.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"

#include <string>
#include <iostream>

int main(int argc, char **argv)
{
    // process command line
    if (argc != 4)
    {
        std::cerr << "usage:" << std::endl
            << "test_interval_iterator [interval]"
            " [cf reader regex] [n expected]" << std::endl;
    }

    std::string interval = argv[1];
    std::string regex = argv[2];
    int n_expected = atoi(argv[3]);

    // get the time values
    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(regex);

    teca_metadata md = cfr->update_metadata();

    // iterate over the time axis, count the number of intervals and compare
    // against the expected number
    teca_calendar_util::p_interval_iterator it =
        teca_calendar_util::interval_iterator_factory::New(interval);

    if (!it || it->initialize(md))
    {
        TECA_ERROR("Failed to initialize the \"" << interval << "\" iterator")
        return -1;
    }

    int n_intervals = 0;
    long n_steps_total = 0;

    while (*it)
    {
        teca_calendar_util::time_point first_step;
        teca_calendar_util::time_point last_step;

        it->get_next_interval(first_step, last_step);

        long n_steps = last_step.index - first_step.index + 1;
        n_steps_total += n_steps;

        std::cerr << "From: " << first_step << " To: " << last_step
            << " steps=" << n_steps  << std::endl;

        n_intervals += 1;
    }

    std::cerr << n_steps_total << " steps located" << std::endl;

    // check the number of intervals
    if (n_intervals != n_expected)
    {
        TECA_ERROR("The \"" << interval << "\" iterator produced "
            << n_intervals << " where " << n_expected << " were expected")
        return -1;
    }

    return 0;
}

