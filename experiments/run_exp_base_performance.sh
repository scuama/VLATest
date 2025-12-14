#!/bin/bash

# 项目根目录
PROJECT_ROOT="/mnt/disk1/decom/VLATest"

models=(openvla-7b)

datasets=(t-grasp_n-1000_o-m3_s-2498586606.json t-move_n-1000_o-m3_s-2263834374.json t-put-on_n-1000_o-m3_s-2593734741.json t-put-in_n-1000_o-m3_s-2905191776.json)

timeout_duration="1h"  # Adjust the timeout duration as needed

for data in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running ${model} on ${data}"
    output_dir="${PROJECT_ROOT}/results/${data%.json}/${model}_2024/"

    # Check if the output directory exists and if all folders have log.json
    if [ ! -d "$output_dir" ] || [ "$(find "$output_dir" -type d | wc -l)" -lt 1000 ] || [ "$(find "$output_dir" -name 'log.json' | wc -l)" -ne 1000 ]; then
      while true; do
        # Run the Python script with a timeout
        cd "${PROJECT_ROOT}/experiments"
        timeout "${timeout_duration}" bash -c "PYTHONPATH=${PROJECT_ROOT} python3 run_fuzzer.py -s 2024 -m '${model}' -d ${PROJECT_ROOT}/data/'${data}' -io /data/"

        # Re-check if the output directory has the expected folders and log.json files
        log_count=$(find "$output_dir" -name 'log.json' 2>/dev/null | wc -l)

        if [ "$log_count" -eq 1000 ]; then
          echo "${model} on ${data} completed successfully with all log.json files."
          break
        else
          echo "Script failed, timed out, or got stuck; restarting ${model} on ${data} from ${log_count}..."
          pkill -f run_fuzzer.py  # Ensure the process is killed if timeout didn't do it
          sleep 5  # Wait a bit before retrying
        fi
      done
    else
      echo "Skipping ${model} on ${data}, already completed and verified."
    fi
  done
done