#!/bin/bash

# Set to 0 to regenerate the baseline data, set to 1 to run the tests
do_test=1

if [[ $# -ne 3 ]]
then
    echo "usage: test_blocking_event_create_overlap_table.sh [app prefix] " \
         "[data root] [n threads]"
    exit -1
fi

app_prefix=${1}
data_root=${2}
n_threads=${3}

set -x

# run create overlap table app
${app_prefix}/teca_blocking_event_tracker \
    ${data_root}/MERRA2_BlockingEvent/test_MERRA2_blocking_event_overlap_table.csv \
    test_MERRA2_mapping.pkl \
    --elapsed-time-array-file ${data_root}/MERRA2_BlockingEvent/test_MERRA2_times.npy 

# Test the generated overlap table and times file against the reference
if [[ ${do_test} -eq 0 ]]
then
    # regenerate the baseline
    mv test_MERRA2_mapping.pkl \
        ${data_root}/MERRA2_BlockingEvent/
else
    # compare pickle file to baseline
    ${app_prefix}/teca_pickle_diff \
        ${data_root}/MERRA2_BlockingEvent/test_MERRA2_mapping.pkl \
        test_MERRA2_mapping.pkl

    # clean up
    rm test_MERRA2_mapping.pkl
fi
