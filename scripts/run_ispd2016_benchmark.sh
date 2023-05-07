#!/bin/bash
# File              : run_ispd2016_benchmark.sh
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 08.31.2021
# Last Modified Date: 09.06.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
# Usage:
#   Run this script under `install` directory.

host=$(hostname)
date_str="${host}_$(date +"%Y.%m.%d_%H.%M.%S")"
log_dir="../ispd2016_log/log_${date_str}"
echo "host: $host"
echo "log_dir = ${log_dir}"

read -p "Description: "
desp=$REPLY

mkdir -p "${log_dir}/results"
echo "desp: $desp"
echo $desp > "${log_dir}/description.txt"

for i in $(seq -f '%02g' 1 12)
do
	configuration_file_path="./unittest/regression/ispd2016/FPGA${i}.json"
	screen_output_file_path="${log_dir}/FPGA${i}.log"
	echo "================================"
	echo "Running Benchmark ${configuration_file_path}"
	python ./openparf.py --config ${configuration_file_path} 2>&1 | tee ${screen_output_file_path}
	if [ $? -eq 0 ]; then
	  cp -r "./results/FPGA${i}" "${log_dir}/results"
	else
	  echo "Benchmark ${configuration_file_path} failed."
	fi
done
