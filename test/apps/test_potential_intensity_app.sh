#!/bin/bash

if [[ $# < 3 ]]
then
    echo "usage: test_potential_intensity_app.sh [app prefix] "   \
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
${launcher} ${app_prefix}/teca_potential_intensity                              \
    --input_regex ${data_root}/tcpypi_sample_data_1980-01-31-10Z'\.nc$'         \
    --psl_variable msl --sst_variable sst --air_temperature_variable t          \
    --mixing_ratio_variable q --t_axis_variable month --z_axis_variable p       \
    --output_file test_potential_intensity_app_%t%.nc                           \
    --file_layout number_of_steps --steps_per_file 12 --verbose 2

# run the diff
${app_prefix}/teca_cartesian_mesh_diff                                          \
    --reference_dataset ${data_root}/test_potential_intensity_app'.*\.nc'       \
    --ref_reader::t_axis_variable month                                         \
    --test_dataset test_potential_intensity_app'.*\.nc'                         \
    --test_reader::t_axis_variable month                                        \
    --arrays V_max P_min IFL T_o OTL --verbose

# clean up
#rm test_potential_intensity_app*.nc
