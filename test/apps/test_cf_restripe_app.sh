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

# run the app to split a single file into one per array
 for array in PSL T200 T500 U850 UBOT V850 VBOT Z1000 Z200;
 do
     echo $array;
     mkdir -p test_cf_restripe/${array};
     ${launcher} ${app_prefix}/teca_cf_restripe \
        --input_regex "${data_root}/test_tc_candidates_1990_07_0[0-9]\.nc" \
        --output_file test_cf_restripe/${array}/${array}_%t%.nc --steps_per_file 2 \
        --n_threads ${n_threads} --point_arrays ${array};
done

set +x
# make the config file
mcf_file=`pwd`/test_cf_restripe.mcf
echo "# TECA multi_cf_reader configuration"      > ${mcf_file}
echo "# `date`"                                 >> ${mcf_file}
echo "" >> ${mcf_file}
echo "data_root = `pwd`/test_cf_restripe"       >> ${mcf_file}
echo "" >> ${mcf_file}
echo "[cf_reader]" >> ${mcf_file}
echo "regex = %data_root%/PSL/PSL.*\.nc$"       >> ${mcf_file}
echo "variables = PSL"                          >> ${mcf_file}
echo "provides_time"                            >> ${mcf_file}
echo "provides_geometry"                        >> ${mcf_file}
echo "" >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/T200/T200.*\.nc$"     >> ${mcf_file}
echo "variables = T200"                         >> ${mcf_file}
echo "" >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/T500/T500.*\.nc$"     >> ${mcf_file}
echo "variables = T500"                         >> ${mcf_file}
echo "" >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/U850/U850.*\.nc$"     >> ${mcf_file}
echo "variables = U850"                         >> ${mcf_file}
echo "" >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/V850/V850.*\.nc$"     >> ${mcf_file}
echo "variables = V850"                         >> ${mcf_file}
echo ""                                         >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/UBOT/UBOT.*\.nc$"     >> ${mcf_file}
echo "variables = UBOT"                         >> ${mcf_file}
echo ""                                         >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/VBOT/VBOT.*\.nc$"     >> ${mcf_file}
echo "variables = VBOT"                         >> ${mcf_file}
echo ""                                         >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/Z1000/Z1000.*\.nc$"   >> ${mcf_file}
echo "variables = Z1000"                        >> ${mcf_file}
echo ""                                         >> ${mcf_file}
echo "[cf_reader]"                              >> ${mcf_file}
echo "regex = %data_root%/Z200/Z200.*\.nc$"     >> ${mcf_file}
echo "variables = Z200"                         >> ${mcf_file}
set -x

# run the app to merge the split up dataset back into its original form
${launcher} ${app_prefix}/teca_cf_restripe                          \
    --input_file ${mcf_file} --output_file test_cf_restripe-%t%.nc  \
    --point_arrays PSL T200 T500 U850 UBOT V850 VBOT Z1000 Z200     \
    --steps_per_file 365 --n_threads ${n_threads}

# run the diff
${app_prefix}/teca_cartesian_mesh_diff                                          \
    --reference_dataset "${data_root}/test_tc_candidates_1990_07_0[0-9]\.nc"    \
    --test_dataset test_cf_restripe-'.*\.nc'                                    \
    --arrays PSL T200 T500 U850 UBOT V850 VBOT Z1000 Z200                       \
    --verbose

# clean up
rm -rf ${mcf_file} test_cf_restripe test_cf_restripe-*.nc
