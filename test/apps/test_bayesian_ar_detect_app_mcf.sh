#!/bin/bash

if [[ $# < 3 ]]
then
    echo "usage: test_bayesian_ar_detect_app.sh [app prefix] "   \
         "[data root] [num threads] [mpiexec] [num ranks]"
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

# run the app
${launcher} ${app_prefix}/teca_bayesian_ar_detect                       \
    --input_file "${app_prefix}/../test/ECMWF-IFS-HR-SST-present.mcf"   \
    --compute_ivt --wind_u ua --wind_v va --specific_humidity hus       \
    --segment_ar_probability --write_ivt --write_ivt_magnitude          \
    --output_file test_bayesian_ar_detect_app_mcf_output_%t%.nc         \
    --steps_per_file 365 --first_step 8 --last_step 23                  \
    --n_threads ${n_threads} --verbose


# run the diff
${app_prefix}/teca_cartesian_mesh_diff                                              \
    --reference_dataset ${data_root}/test_bayesian_ar_detect_app_mcf_ref'.*\.nc'    \
    --test_dataset test_bayesian_ar_detect_app_mcf_output'.*\.nc'                   \
    --arrays IVT IVT_U IVT_V ar_probability ar_binary_tag                           \
    --verbose

# clean up
rm test_bayesian_ar_detect_app_mcf_output*.nc
