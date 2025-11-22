#!/bin/bash

# Set to 0 to regenerate the baseline data, set to 1 to run the tests
do_test=1

if [[ $# < 2 ]]
then
    echo "usage: test_blocking_event_create_overlap_table.sh [app prefix] " \
         "[data root] [n threads] [mpi exec] [n ranks]"
    exit -1
fi

app_prefix=${1}
data_root=${2}
n_threads=${3}

if [[ $# -eq 5 ]]
then
    mpi_exec=${4}
    test_cores=${5}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run create overlap table app
${launcher} ${app_prefix}/teca_blocking_event_create_overlap_table \
    --input_regex ${data_root}/MERRA2_BlockingEvent/MERRA2_blocktag\.198102..\.nc \
    --output_file test_MERRA2_blocking_event_overlap_table.csv \
    --output_times_file test_MERRA2_times.npy \
    --n_threads ${n_threads}

# Test the generated overlap table and times file against the reference
if [[ ${do_test} -eq 0 ]]
then
    # regenerate the baseline
    mv test_MERRA2_blocking_event_overlap_table.csv \
        ${data_root}/MERRA2_BlockingEvent/
    mv test_MERRA2_times.npy \
        ${data_root}/MERRA2_BlockingEvent/
else
    # run the diff
    test_table="test_MERRA2_blocking_event_overlap_table.csv"
    gt_table="${data_root}/MERRA2_BlockingEvent/test_MERRA2_blocking_event_overlap_table.csv"

    # Ignore lines starting with '#' since TECA adds git commit info in one header line
    # Sort table because order of lines may vary
    if diff -q <(grep -v '^#' "$test_table" | sort) <(grep -v '^#' "$gt_table" | sort) >/dev/null; then
        echo "SUCCESS: Match to reference table"
    else
        echo "ERROR: blocking event overlap table does not match reference"
    fi

    ${app_prefix}/teca_numpy_array_diff \
        ${data_root}/MERRA2_BlockingEvent/test_MERRA2_times.npy \
        test_MERRA2_times.npy

    # clean up
    rm test_MERRA2_blocking_event_overlap_table.csv test_MERRA2_times.npy
fi
