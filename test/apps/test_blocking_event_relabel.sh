#!/bin/bash

# Set to 0 to regenerate the baseline data, set to 1 to run the tests
do_test=1

if [[ $# < 2 ]]
then
    echo "usage: test_blocking_event_relabel.sh [app prefix] " \
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
${launcher} ${app_prefix}/teca_blocking_event_relabel \
    --input_regex ${data_root}/MERRA2_BlockingEvent/MERRA2_blocktag\.198102..\.nc \
    --first_step 0 --last_step 0 \
    --mapping_file ${data_root}/MERRA2_BlockingEvent/test_MERRA2_mapping.pkl \
    --output_file test_MERRA2_blockid.%t%.nc \
    --file_layout daily --date_format %Y%m%d \
    --n_threads ${n_threads}

# Test the generated overlap table and times file against the reference
if [[ ${do_test} -eq 0 ]]
then
    # regenerate the baseline
    mv test_MERRA2_blockid.*.nc \
        ${data_root}/MERRA2_BlockingEvent/
else
    # run the diff
    ${app_prefix}/teca_cartesian_mesh_diff                                   \
        --reference_dataset "${data_root}/MERRA2_BlockingEvent/test_MERRA2_blockid\.19810208\.nc" \
        --test_dataset "test_MERRA2_blockid\.19810208\.nc"             \
        --arrays global_blocking_event_id --relative_tolerance 1e-4      \
        --absolute_tolerance 1e-6 --verbose

    # clean up
    rm test_MERRA2_blockid.*.nc
fi
