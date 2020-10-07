#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_tc_detect_app.sh [app prefix] [data root] "     \
         "[mpi exec] [test cores]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

if [[ $# -eq 4 ]]
then
    mpi_exec=${3}
    test_cores=${4}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_tc_detect                              \
    --input_regex "${data_root}/test_tc_candidates_1990_07_0[12]\.nc" \
    --first_step 0 --last_step 3 --lowest_lat -20 --highest_lat 20    \
    --n_threads 1 --candidate_file test_tc_candidates_app_output.bin  \
    --track_file test_tc_track_app_output.bin

# run the diff
${app_prefix}/teca_table_diff                                         \
    "${data_root}/test_tc_candidates_app_ref.bin"                     \
    test_tc_candidates_app_output.bin "storm_id"

# run the diff
${app_prefix}/teca_table_diff                                         \
    "${data_root}/test_tc_track_app_ref.bin"                          \
    test_tc_track_app_output.bin "storm_id"

# clean up
rm test_tc_candidates_app_output.bin
rm test_tc_track_app_output.bin
