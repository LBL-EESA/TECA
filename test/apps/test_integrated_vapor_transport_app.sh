#!/bin/bash

if [[ $# < 3 ]]
then
    echo "usage: test_integrated_vapor_transport_app.sh [app prefix] "   \
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
set -e

# run the app
${launcher} ${app_prefix}/teca_integrated_vapor_transport               \
    --input_file "${app_prefix}/../test/ECMWF-IFS-HR-SST-present.mcf"   \
    --wind_u ua --wind_v va --specific_humidity hus                     \
    --write_ivt 1 --write_ivt_magnitude 1 --n_threads ${n_threads}      \
    --output_file test_integrated_vapor_transport_app_mcf_output_%t%.nc \
    --verbose --steps_per_file 365



# run the diff
${app_prefix}/teca_cartesian_mesh_diff                                                      \
    --reference_dataset ${data_root}/test_integrated_vapor_transport_app_mcf_ref'.*\.nc'    \
    --test_dataset test_integrated_vapor_transport_app_mcf_output'.*\.nc'                   \
    --arrays IVT_U IVT_V IVT --absolute_tolerance 5e-5 --verbose

# clean up
rm test_integrated_vapor_transport_app_mcf_output*.nc
