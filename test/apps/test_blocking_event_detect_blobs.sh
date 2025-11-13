#!/bin/bash

# This script tests all blocking event apps (currently
# teca_blocking_event_detect_blobs, teca_blocking_event_create_overlap_table).
# These are tested in a single script since the later apps depend on the output
# of the former and CMake's testing framework does not support such dependencies.
# While this script tests the full pipeline, it also tests intermediate outputs
# against reference data to ensure correctness at each stage.

# Set to 0 to regenerate the baseline data, set to 1 to run the tests
do_test=1

if [[ $# < 2 ]]
then
    echo "usage: test_blocking_event_apps.sh [app prefix] " \
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

# run detect blobs app; discard unique region ids since they vary
# between CPU and GPU runs. Furthermore, this behavior mirrors what
# TempestExtremes DetectBlobs does.
${launcher} ${app_prefix}/teca_blocking_event_detect_blobs \
    --input_regex ${data_root}/MERRA2_BlockingEvent/MERRA2_100\.inst3_3d_asm_Np_Only3LayersH\.19810208\.nc \
    --geopotential_height_variable H --first_step 0 --last_step 0 \
    --thresholds_regex ${data_root}/MERRA2_BlockingEvent/MERRA2_threshold_H_filtered\.nc \
    --discard_unique_region_ids \
    --output_file test_MERRA2_blocktag.%t%.nc \
    --file_layout daily --date_format %Y%m%d \
    --n_threads ${n_threads}

# Test the generated blocking event tag file against the reference for 
# a specific date (sufficient to validate correctness)
if [[ ${do_test} -eq 0 ]]
then
    # regenerate the baseline
    mv test_MERRA2_blocktag.19810208.nc \
        ${data_root}/MERRA2_BlockingEvent/
else
    # run the diff
    ${app_prefix}/teca_cartesian_mesh_diff                                   \
        --reference_dataset "${data_root}/MERRA2_BlockingEvent/test_MERRA2_blocktag\.19810208\.nc" \
        --test_dataset "test_MERRA2_blocktag\.19810208\.nc"             \
        --arrays blocking_event_mask --relative_tolerance 1e-4      \
        --absolute_tolerance 1e-6 --verbose

    # clean up
    rm test_MERRA2_blocktag.19810208.nc
fi
