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
${launcher} ${app_prefix}/teca_integrated_vapor_transport                           \
    --input_regex "${data_root}/ERAinterim_1979-01-0.*\.nc$"                        \
    --x_axis_variable longitude --y_axis_variable latitude --z_axis_variable level  \
    --wind_u uwnd --wind_v vwnd --specific_humidity shum --write_ivt 1              \
    --write_ivt_magnitude 1 --steps_per_file 256 --n_threads ${n_threads} --verbose \
    --output_file test_integrated_vapor_transport_app_packed_data_output_%t%.nc

do_test=1
if [[ $do_test -eq 0 ]]
then
    echo "regenerating baseline..."
    for f in `ls test_integrated_vapor_transport_app_packed_data_output_*.nc`
    do
        ff=`echo $f | sed s/output/ref/g`
        cp -vd $f ${data_root}/$ff
    done
else
    # run the diff
    ${app_prefix}/teca_cartesian_mesh_diff                                                              \
        --reference_dataset ${data_root}/test_integrated_vapor_transport_app_packed_data_ref'.*\.nc'    \
        --test_dataset test_integrated_vapor_transport_app_packed_data_output'.*\.nc'                   \
        --test_reader::x_axis_variable longitude --test_reader::y_axis_variable latitude                \
        --ref_reader::x_axis_variable longitude --ref_reader::y_axis_variable latitude                  \
        --arrays IVT_U IVT_V IVT --absolute_tolerance 1e-6 --verbose

    # clean up
    rm test_integrated_vapor_transport_app_packed_data_output*.nc
fi
