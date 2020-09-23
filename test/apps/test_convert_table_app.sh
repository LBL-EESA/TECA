#!/bin/bash

if [[ $# -ne 4 ]]
then
    echo "usage: test_convert_table_app.sh [app prefix] " \
         "[data root] [in file] [out file]"
    exit -1
fi

set -x

ierr=0

app_prefix=${1}
data_root=${2}
in_file=${3}
out_file=${4}
out_ext=${4##*.}

${app_prefix}/teca_convert_table ${data_root}/${in_file} ${out_file}

# check that the file was written
if [[ ! -f ${out_file} ]]
then
    echo "ERROR: output file ${out_file} does not exist"
    ierr=-1
fi

# do a diff
if [[ "${out_ext}" != "nc" ]]
then
    ${app_prefix}/teca_table_diff "${out_file}" "${data_root}/${in_file}"
    res=$?
    if [[ ${res} -ne 0 ]]
    then
        echo "ERROR: the converted table differs from the baseline"
        ierr=-1
    fi
fi

# clean output file.
# rm ${out_file}

exit ${ierr}
